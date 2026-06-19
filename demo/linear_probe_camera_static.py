# encoding: utf-8
import numpy as np
import torch
import sys
import argparse
import warnings
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader


class LinearProbeAnalyzer:
    def __init__(self, cfg):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        cfg.defrost()
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg = cfg

        print("Building model...")
        self.model = build_model(cfg)
        self.model.eval()
        self.model.to(self.device)
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

        self.features_cache = {'x_cls': None, 'q_view': None, 'f_inv': None}
        self._register_hooks()

    def _register_hooks(self):
        def hook_pavd(module, inputs, output):
            self.features_cache['x_cls'] = inputs[0].detach().cpu()
            if 'prompts' in output and output['prompts'] is not None:
                self.features_cache['q_view'] = output['prompts'].mean(dim=1).detach().cpu()
            else:
                self.features_cache['q_view'] = None
            self.features_cache['f_inv'] = output['view_invariant_feats'].detach().cpu()

        pavd_module = self.model.module.pavd if hasattr(self.model, 'module') else self.model.pavd
        pavd_module.register_forward_hook(hook_pavd)

    def extract_dataset_features(self, dataset_name, max_samples=-1, random_seed=42):
        print(f"Loading dataset: {dataset_name} for Feature Extraction...")
        test_loader, _ = build_reid_test_loader(self.cfg, dataset_name=dataset_name)

        all_x_cls, all_q_view, all_f_inv, all_camids = [], [], [], []
        extracted_count = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch['images'].to(self.device)
                camids = batch['camids'].cpu().numpy()
                inputs = {
                    "images": images,
                    "targets": torch.zeros_like(batch['camids']),
                    "camids": batch['camids'],
                    "viewids": ['Aerial'] * len(camids)
                }
                self.model(inputs)

                all_x_cls.append(self.features_cache['x_cls'].numpy())
                if self.features_cache['q_view'] is not None:
                    all_q_view.append(self.features_cache['q_view'].numpy())
                all_f_inv.append(self.features_cache['f_inv'].numpy())
                all_camids.extend(camids)

                extracted_count += len(camids)
                if extracted_count % 500 == 0:
                    print(f"Extracted {extracted_count} samples...")
                # 不提前 break，而是收集全部数据（否则顺序取前 N 会有偏差）
                # 但为了控制内存，可以先收集全部，后续再随机采样

        # 拼接所有特征
        X_cls = np.concatenate(all_x_cls, axis=0)
        X_finv = np.concatenate(all_f_inv, axis=0)
        y_cam = np.array(all_camids)
        X_qview = np.concatenate(all_q_view, axis=0) if all_q_view else None

        total = len(y_cam)
        print(f"Total samples extracted: {total}")

        # 随机采样 max_samples 个样本（打乱顺序后取前 max_samples）
        if max_samples > 0 and max_samples < total:
            rng = np.random.RandomState(random_seed)
            indices = rng.permutation(total)[:max_samples]
            X_cls = X_cls[indices]
            X_finv = X_finv[indices]
            y_cam = y_cam[indices]
            if X_qview is not None:
                X_qview = X_qview[indices]
            print(f"Randomly selected {max_samples} samples (seed={random_seed})")
        elif max_samples > 0:
            print(f"Using all {total} samples (max_samples >= total)")

        print(f"Feature extraction complete. Final samples: {len(y_cam)}")
        return X_cls, X_qview, X_finv, y_cam

    @staticmethod
    def majority_baseline(y):
        counts = Counter(y)
        majority_ratio = max(counts.values()) / len(y)
        return majority_ratio * 100.0

    def run_linear_probe(self, X, y, feature_name):
        if X is None:
            return "N/A"
        var = np.var(X)
        print(f"  [Diagnostic] {feature_name} variance: {var:.6f}")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        clf = LogisticRegression(max_iter=2000, n_jobs=-1, multi_class='multinomial')
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100.0
        return acc

    def run(self, dataset_name, max_samples):
        X_cls, X_qview, X_finv, y_cam = self.extract_dataset_features(dataset_name, max_samples)

        majority_acc = self.majority_baseline(y_cam)
        print(f"\n[Baseline] Majority class accuracy: {majority_acc:.2f}%")

        print("\n" + "=" * 50)
        print("Linear Probe Results (Camera ID Prediction)")
        print("=" * 50)

        acc_cls = self.run_linear_probe(X_cls, y_cam, "x_cls (Baseline)")
        print(f"1. Baseline Features (x_cls):        {acc_cls:.2f}%")

        acc_qview = self.run_linear_probe(X_qview, y_cam, "Q_view (Learnable Queries)") if X_qview is not None else "N/A"
        if acc_qview != "N/A":
            print(f"2. View-aware Queries (Q_view):      {acc_qview:.2f}%")
        else:
            print("2. View-aware Queries (Q_view):      Not Available")

        acc_finv = self.run_linear_probe(X_finv, y_cam, "f_inv (Final Modulated)")
        print(f"3. Final Identity Features (f_inv):  {acc_finv:.2f}%")
        print("=" * 50)

        if X_qview is not None and np.var(X_qview) < 1e-4:
            print(f"\n[Note] Q_view has near-zero variance (static prompts).")
            print(f"       Its accuracy ({acc_qview:.2f}%) should be compared to majority baseline ({majority_acc:.2f}%).")
        print()


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Linear Probe for Camera ID prediction (single model)")
    parser.add_argument("--config-file", required=True, help="Model config file")
    parser.add_argument("--dataset-name", default="CARGO", help="Dataset name registered in fastreid")
    parser.add_argument("--max-samples", type=int, default=8000, help="Number of samples to extract (randomly sampled). Set -1 for whole dataset.")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER, help="Modify config options using the command-line")
    args = parser.parse_args()

    setup_logger(name="fastreid")
    cfg = setup_cfg(args)
    analyzer = LinearProbeAnalyzer(cfg)
    analyzer.run(args.dataset_name, args.max_samples)

"""
CUDA_VISIBLE_DEVICES=0 python3 demo/linear_probe_camera_static.py \
--config-file 之前的消融/CARGO参数分析/补充实验/VAD不与patch_token交互/config.yml \
--dataset-name CARGO \
--max-samples 8000 \
MODEL.WEIGHTS 之前的消融/CARGO参数分析/补充实验/VAD不与patch_token交互/model_best.pth
"""