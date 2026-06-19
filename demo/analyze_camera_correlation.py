# encoding: utf-8
import os
import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import pearsonr
import sys
import argparse
import matplotlib.pyplot as plt
import seaborn as sns   # 需要安装：pip install seaborn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

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
from PIL import Image

class CameraCorrelationAnalyzer:
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

        self.avd_weights = None
        self._register_hooks()

        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST, interpolation=BICUBIC),
            T.ToTensor(),
        ])

    def _register_hooks(self):
        def hook_avd(module, input, output):
            if isinstance(output, tuple) and len(output) >= 5:
                w = output[4].detach().cpu()
                if w.dim() == 3:
                    w = w.squeeze(1)
                self.avd_weights = w.numpy()

        if hasattr(self.model, 'module'):
            base_avd = self.model.module.pavd.avd
        else:
            base_avd = self.model.pavd.avd
        if hasattr(base_avd, 'adaptive_view_disentangle'):
            base_avd.adaptive_view_disentangle[-1]['cross_attn'].register_forward_hook(hook_avd)
        else:
            # 备用方案，适用于版本差异较大的情况
            base_avd.register_forward_hook(hook_avd)

    def analyze_dataset(self, dataset_name, max_samples=-1, aerial_only=False):
        print(f"Loading dataset: {dataset_name} ...")
        test_loader, _ = build_reid_test_loader(self.cfg, dataset_name=dataset_name)

        img_paths = []
        camids = []
        for batch in test_loader:
            img_paths.extend(batch['img_paths'])
            camids.extend(batch['camids'].cpu().numpy())

        if aerial_only:
            keep = [i for i, cid in enumerate(camids) if cid in [1, 2, 3, 4, 5]]
            img_paths = [img_paths[i] for i in keep]
            camids = [camids[i] for i in keep]
            print(f"Analyzing Aerial-only cameras (1-5): {len(img_paths)} images.")
        else:
            print(f"Analyzing all cameras (1-13): {len(img_paths)} images.")

        if max_samples > 0 and max_samples < len(img_paths):
            indices = np.random.choice(len(img_paths), max_samples, replace=False)
            img_paths = [img_paths[i] for i in indices]
            camids = [camids[i] for i in indices]

        # 存储分析数据
        all_weights = []
        valid_camids = []

        for idx, (path, cid) in enumerate(zip(img_paths, camids)):
            if idx % 200 == 0:
                print(f"Processing {idx}/{len(img_paths)}...")

            pil = Image.open(path).convert('RGB')
            tensor = self.transform(pil).unsqueeze(0).to(self.device)

            self.avd_weights = None
            with torch.no_grad():
                inputs = {"images": tensor}
                inputs.update({'targets': torch.tensor([0]), 'camids': torch.tensor([0]), 'viewids': ['Aerial']})
                self.model(inputs)

            if self.avd_weights is not None:
                w = self.avd_weights.squeeze()
                all_weights.append(w)
                valid_camids.append(cid)

        if len(all_weights) == 0:
            print("No valid weights collected! Check hook compatibility.")
            return

        all_weights = np.array(all_weights)
        valid_camids = np.array(valid_camids)
        num_queries = all_weights.shape[1]

        print("\n" + "=" * 60)
        print(f"Pearson Correlation with Camera ID (Aerial-only: {aerial_only})")
        print("=" * 60)
        best_r = 0
        best_q = -1
        for q in range(num_queries):
            r, p = pearsonr(valid_camids, all_weights[:, q])
            print(f"Query {q:2d}: r = {r:6.4f}, p = {p:.2e}")
            if abs(r) > abs(best_r):
                best_r = r
                best_q = q

        print("-" * 60)
        print(f"Strongest correlation: Query {best_q} (r = {best_r:.4f})")

        # 绘制散点图
        self.plot_correlation(valid_camids, all_weights[:, best_q],
                              xlabel='Camera ID (Proxy for Viewpoint Continuum)',
                              ylabel=f'Query {best_q} Weight',
                              title=f'Correlation with Camera ID\n(Query {best_q}, r={best_r:.3f})',
                              save_path=f"correlation_camera_q{best_q}.pdf")

    @staticmethod
    def plot_correlation(x, y, xlabel, ylabel, title, save_path):
        plt.figure(figsize=(8, 5))
        sns.set_style("whitegrid")
        plt.scatter(x, y, alpha=0.6, s=20, c='steelblue', edgecolors='w')

        # 线性回归
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_sorted = np.linspace(min(x), max(x), 100)
        plt.plot(x_sorted, p(x_sorted), "r--", linewidth=2.5, label=f'Linear fit (slope = {z[0]:.3f})')

        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=12)
        plt.legend()
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path, dpi=200)
        plt.close()
        print(f"Saved correlation plot to {save_path}")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--dataset-name", required=True,
                        help="Dataset name registered in fastreid (should be CARGO)")
    parser.add_argument("--max-samples", type=int, default=2000,
                        help="Max images to process (-1 for all)")
    parser.add_argument("--aerial-only", action="store_true",
                        help="Only use aerial camera images (cam 1-5)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")
    analyzer = CameraCorrelationAnalyzer(cfg)
    analyzer.analyze_dataset(dataset_name=args.dataset_name,
                             max_samples=args.max_samples,
                             aerial_only=args.aerial_only)
    
"""
# 分析全部相机(ID 1-13)
CUDA_VISIBLE_DEVICES=0 python3 demo/analyze_camera_correlation.py \
    --config-file /path/to/your/qvam_cargo_config.yml \
    --dataset-name CARGO \
    --max-samples 3000 \
    MODEL.WEIGHTS /path/to/qvam_cargo_model.pth

# 分析航拍相机(ID 1-5),结果应该更明显
CUDA_VISIBLE_DEVICES=0 python3 demo/analyze_camera_correlation.py \
    --config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml \
    --dataset-name CARGO \
    --max-samples 2000 \
    --aerial-only \
    MODEL.WEIGHTS 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/model_best.pth
"""