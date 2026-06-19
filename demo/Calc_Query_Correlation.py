# encoding: utf-8
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from scipy.stats import pearsonr
import sys
import argparse
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

class ViewpointCorrelationAnalyzer:
    def __init__(self, cfg):
        self.device = torch.device("cuda")
        cfg.defrost()
        # cfg.MODEL.DEVICE = "cpu"
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
                self.avd_weights = output[4].detach().cpu().squeeze().numpy()

        if hasattr(self.model, 'module'): 
            base_avd = self.model.module.pavd.avd
        else: 
            base_avd = self.model.pavd.avd
            
        base_avd.register_forward_hook(hook_avd)

    def analyze_images(self, img_paths, max_samples=-1):
        """通用分析函数：输入图像路径列表，输出相关性结果"""
        # 随机打乱并限制数量
        np.random.shuffle(img_paths)
        max_samples = len(img_paths) if max_samples == -1 else max_samples
        img_paths = img_paths[:max_samples]

        aspect_ratios = []
        query_weights_history = []

        for idx, path in enumerate(img_paths):
            if idx % 200 == 0:
                print(f"Processing {idx}/{len(img_paths)}...")
                
            raw_img = cv2.imread(path)
            if raw_img is None:
                continue
            h, w = raw_img.shape[:2]
            aspect_ratio = h / float(w)
            
            pil = Image.open(path).convert('RGB')
            tensor = self.transform(pil).unsqueeze(0).to(self.device)
            
            self.avd_weights = None
            with torch.no_grad():
                inputs = {"images": tensor}
                if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
                     inputs.update({'targets': torch.tensor([0]), 
                                    'camids': torch.tensor([0]), 
                                    'viewids': ['Aerial']})
                     self.model(inputs)
                else:
                     self.model(tensor)
            
            if self.avd_weights is not None:
                aspect_ratios.append(aspect_ratio)
                query_weights_history.append(self.avd_weights)

        if len(aspect_ratios) == 0:
            print("No valid images processed!")
            return

        aspect_ratios = np.array(aspect_ratios)
        query_weights_history = np.array(query_weights_history)
        num_queries = query_weights_history.shape[1]
        
        print("\n" + "="*50)
        print("Pearson Correlation Analysis (Aspect Ratio vs. Query Weight)")
        print("="*50)
        
        best_r = 0
        best_q = -1
        
        for q_idx in range(num_queries):
            weights = query_weights_history[:, q_idx]
            r, p_value = pearsonr(aspect_ratios, weights)
            print(f"Query P{q_idx}: r = {r:.4f}, p-value = {p_value:.2e}")
            
            if abs(r) > abs(best_r):
                best_r = r
                best_q = q_idx
                
        print("-" * 50)
        print(f"STRONGEST correlation found at Query P{best_q} (r = {best_r:.4f})")
        print("You can use this value in your rebuttal.")

    def analyze_dataset(self, dataset_name, max_samples=-1):
        """通过数据集名称加载所有测试图像（query + gallery）"""
        print(f"Loading dataset: {dataset_name} ...")
        test_loader, num_query = build_reid_test_loader(self.cfg, dataset_name=dataset_name)
        
        # 收集所有图像路径（先收集整个数据集的路径，不依赖特征提取）
        img_paths = []
        for batch in test_loader:
            img_paths.extend(batch['img_paths'])
        
        print(f"Loaded {len(img_paths)} images from dataset.")
        self.analyze_images(img_paths, max_samples)

    def analyze_directory(self, img_dir, max_samples=-1):
        """原有的文件夹模式"""
        print(f"Scanning directory: {img_dir} ...")
        valid_exts = {'.jpg', '.jpeg', '.png'}
        img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
                     if os.path.splitext(f)[1].lower() in valid_exts]
        print(f"Found {len(img_paths)} images.")
        self.analyze_images(img_paths, max_samples)


def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    # 新增数据集模式参数
    parser.add_argument("--dataset-name", help="Dataset name registered in fastreid (e.g., AG_ReID_v2_A2W)")
    parser.add_argument("--img-dir", help="Directory containing images (folder mode)")
    parser.add_argument("--max-samples", type=int, default=2000, help="Maximum number of images to analyze")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if not args.dataset_name and not args.img_dir:
        parser.error("Either --dataset-name or --img-dir must be provided.")

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")
    analyzer = ViewpointCorrelationAnalyzer(cfg)

    if args.dataset_name:
        analyzer.analyze_dataset(args.dataset_name, args.max_samples)
    else:
        analyzer.analyze_directory(args.img_dir, args.max_samples)

"""
CUDA_VISIBLE_DEVICES=0 \
python3 demo/Calc_Query_Correlation.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--img-dir /path/to/your/AGReID/bounding_box_test \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth

------------------------AGReIDv2------------------------------
CUDA_VISIBLE_DEVICES=1 \
python3 demo/Calc_Query_Correlation.py \
--config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
--dataset-name AG_ReID_v2_A2W \
--max-samples 2000 \
MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth

==================================================
Pearson Correlation Analysis (Aspect Ratio vs. Query Weight)
==================================================
Query P0: r = -0.0503, p-value = 6.17e-10
Query P1: r = -0.1348, p-value = 2.81e-62
Query P2: r = 0.1168, p-value = 4.40e-47
Query P3: r = 0.0128, p-value = 1.14e-01
--------------------------------------------------
STRONGEST correlation found at Query P1 (r = -0.1348)

------------------------AGReID------------------------------
CUDA_VISIBLE_DEVICES=1 \
python3 demo/Calc_Query_Correlation.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--dataset-name AG_ReID \
--max-samples -1 \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth

------------------------CARGO------------------------------
CUDA_VISIBLE_DEVICES=1 \
python3 demo/Calc_Query_Correlation.py \
--config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml \
--dataset-name CARGO_AG \
--max-samples -1 \
MODEL.WEIGHTS 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/model_best.pth
"""