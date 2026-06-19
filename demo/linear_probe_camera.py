# encoding: utf-8
import os
import numpy as np
import torch
import torchvision.transforms as T
import sys
import argparse
import warnings

# 导入 sklearn 相关库用于 Linear Probe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image as PILImage
    BICUBIC = PILImage.BICUBIC

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
        """
        利用 Hook 截获 PAVD 的输入和输出，提取三种核心特征
        """
        def hook_pavd(module, inputs, output):
            self.features_cache['x_cls'] = inputs[0].detach().cpu()
            if 'prompts' in output and output['prompts'] is not None:
                self.features_cache['q_view'] = output['prompts'].mean(dim=1).detach().cpu()
            else:
                self.features_cache['q_view'] = None
                
            self.features_cache['f_inv'] = output['view_invariant_feats'].detach().cpu()

        pavd_module = self.model.module.pavd if hasattr(self.model, 'module') else self.model.pavd
        pavd_module.register_forward_hook(hook_pavd)

    def extract_dataset_features(self, dataset_name, max_samples=-1):
        print(f"Loading dataset: {dataset_name} for Feature Extraction...")
        test_loader, _ = build_reid_test_loader(self.cfg, dataset_name=dataset_name)

        all_x_cls = []
        all_q_view = []
        all_f_inv = []
        all_camids = []

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

        # 拼接成整块 numpy 数组 (这里先不截断，拿到全部数据再进行平衡)
        X_cls = np.concatenate(all_x_cls, axis=0)
        X_finv = np.concatenate(all_f_inv, axis=0)
        y_cam = np.array(all_camids)
        
        X_qview = None
        if len(all_q_view) > 0:
            X_qview = np.concatenate(all_q_view, axis=0)

        # ========================================================
        # 新增逻辑：确保各类 Camera ID 样本数量绝对平均
        # ========================================================
        unique_cams, counts = np.unique(y_cam, return_counts=True)
        print("\n[Balance] Camera ID distribution before balancing:")
        for c, count in zip(unique_cams, counts):
            print(f"  Camera {c:2d}: {count} samples")

        # 计算每个相机能分配到的最大平衡数量
        min_count = counts.min()
        target_per_class = min_count
        
        # 如果用户设置了总样本上限，则调整每个类的抽取数量
        if max_samples > 0:
            target_per_class = min(target_per_class, max_samples // len(unique_cams))

        print(f"\n[Balance] Selecting strictly {target_per_class} samples per camera...")

        balanced_indices = []
        for c in unique_cams:
            c_idx = np.where(y_cam == c)[0]
            # 固定随机种子保证结果可复现，随机选取指定数量的索引
            np.random.seed(42)
            selected_idx = np.random.choice(c_idx, target_per_class, replace=False)
            balanced_indices.extend(selected_idx)

        balanced_indices = np.array(balanced_indices)
        np.random.shuffle(balanced_indices) # 打乱整体顺序

        # 根据平衡后的索引重新过滤数据
        X_cls = X_cls[balanced_indices]
        X_finv = X_finv[balanced_indices]
        y_cam = y_cam[balanced_indices]
        if X_qview is not None:
            X_qview = X_qview[balanced_indices]

        print(f"[Balance] Feature extraction complete. Final balanced samples: {len(y_cam)}")
        # ========================================================

        return X_cls, X_qview, X_finv, y_cam

    def run_linear_probe(self, X, y, feature_name):
        """
        核心评估逻辑：训练逻辑回归预测 Camera ID
        """
        if X is None:
            return "N/A"
            
        # ========================================================
        # 修改逻辑：加入 stratify=y 确保训练集和测试集中各种相机的数量绝对一致
        # ========================================================
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # 归一化 (对 Linear Probe 非常重要)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 训练逻辑回归分类器
        print(f"Training Linear Probe for [{feature_name}] ...")
        clf = LogisticRegression(max_iter=2000, n_jobs=-1)
        clf.fit(X_train, y_train)

        # 评估准确率
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred) * 100
        return acc

    def run(self, dataset_name, max_samples):
        X_cls, X_qview, X_finv, y_cam = self.extract_dataset_features(dataset_name, max_samples)

        # 调试：打印 Q_view 的方差，检查是否为常数
        if X_qview is not None:
            qvar = np.var(X_qview)
            print(f"\n[Debug] Q_view variance across all samples: {qvar:.8f}")
            if qvar < 1e-8:
                print("[Debug] Q_view is constant (all samples identical).")
            else:
                print("[Debug] Q_view has variation across samples.")
        else:
            print("\n[Debug] Q_view is None (no prompts output).")

        print("\n" + "="*50)
        print("🚀 Linear Probe Results (Camera ID Prediction)")
        print("="*50)
        
        # 1. 评估 Baseline 特征
        acc_cls = self.run_linear_probe(X_cls, y_cam, "x_cls (Baseline)")
        print(f"1. Baseline Features (x_cls):        {acc_cls:.2f}%")

        # 2. 评估 Q_view 特征
        acc_qview = self.run_linear_probe(X_qview, y_cam, "Q_view (Learnable Queries)")
        if acc_qview != "N/A":
            print(f"2. View-aware Queries (Q_view):      {acc_qview:.2f}%")
        else:
            print("2. View-aware Queries (Q_view):      Not Available (Static Mode)")

        # 3. 评估 Final Modulated 特征
        acc_finv = self.run_linear_probe(X_finv, y_cam, "f_inv (Final Modulated)")
        print(f"3. Final Identity Features (f_inv):  {acc_finv:.2f}%")
        print("="*50)


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-name", default="CARGO", help="Dataset name registered in fastreid")
    # 注意：现在这里的 max_samples 意味着“总共最多取多少个样本”，
    # 程序会自动将其平分给 13 个相机，比如设为 6500，则每个相机严格抽取 500 个
    parser.add_argument("--max-samples", type=int, default=13000, help="Max total images to extract after balancing (-1 for maximum possible balanced size).")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")
    
    analyzer = LinearProbeAnalyzer(cfg)
    analyzer.run(args.dataset_name, args.max_samples)

"""
CUDA_VISIBLE_DEVICES=1 python3 demo/linear_probe_camera.py \
--config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml \
--dataset-name CARGO \
--max-samples 50000 \
MODEL.WEIGHTS 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/model_best.pth


CUDA_VISIBLE_DEVICES=0 python3 demo/linear_probe_camera.py \
--config-file 之前的消融/CARGO参数分析/补充实验/VAD不与patch_token交互/config.yml \
--dataset-name CARGO \
--max-samples 50000 \
MODEL.WEIGHTS 之前的消融/CARGO参数分析/补充实验/VAD不与patch_token交互/model_best.pth
"""