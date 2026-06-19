# encoding: utf-8
import argparse
import sys
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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
from fastreid.data.transforms import *

class SingleRowAnalyzer:
    def __init__(self, cfg):
        self.device = torch.device("cpu")
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg = cfg
        
        print("Building model...")
        self.model = build_model(cfg)
        self.model.eval()
        self.model.to(self.device)
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

        self.hooks = {'vad': None}
        self._register_hooks()

        self.input_size = cfg.INPUT.SIZE_TEST
        self.transform = T.Compose([
            T.Resize(self.input_size, interpolation=BICUBIC),
            ToTensor(), 
        ])

    def _register_hooks(self):
        def hook_vad(module, input, output):
            if isinstance(output, tuple):
                self.hooks['vad'] = output[1].detach().cpu()
        
        if hasattr(self.model, 'module'):
            base = self.model.module.pavd
        else:
            base = self.model.pavd
        base.vad.register_forward_hook(hook_vad)

    def process_image(self, img_path):
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise ValueError(f"Error loading: {img_path}")
        raw_resized = cv2.resize(raw_img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)
        pil = Image.open(img_path).convert('RGB')
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        self.hooks['vad'] = None
        with torch.no_grad():
            inputs = {"images": tensor}
            if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
                inputs.update({'targets': torch.tensor([0]), 'camids': torch.tensor([0]), 'viewids': ['Aerial']})
                self.model(inputs)
            else:
                self.model(tensor)
        return {'rgb': rgb, 'vad': self.hooks['vad']}

    def generate_avg_heatmap(self, res):
        """生成平均热力图（所有Query的平均注意力）"""
        attn_vad = res['vad']
        if attn_vad.shape[0] == 1:
            attn_vad = attn_vad.squeeze(0)
        avg_attn = torch.mean(attn_vad, dim=0)  # [N]
        H, W, _ = res['rgb'].shape
        grid_h, grid_w = self.input_size[0]//16, self.input_size[1]//16
        attn_grid = avg_attn.reshape(1, 1, grid_h, grid_w)
        attn_large = F.interpolate(attn_grid, size=(H, W), mode='bicubic', align_corners=False).squeeze()
        att = attn_large - attn_large.min()
        att = att / (att.max() + 1e-8)
        heatmap = cv2.applyColorMap(np.uint8(255 * att.numpy()), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        original_img = res['rgb'].astype(np.float32) / 255.0
        overlay = 0.6 * original_img + 0.4 * heatmap
        return original_img, overlay

    def analyze_and_plot(self, img_paths, output_path):
        """
        img_paths: 同一ID的多个视角图像，按顺序 (ground → 逐步升高 → top-down)
        输出：一行，列 = 2*len(img_paths)，分别为：原始图, 热力图, 原始图, 热力图, ...
        原始图像下方添加视角标签，热力图无标签。
        """
        num_imgs = len(img_paths)
        # 存储每张图的信息
        rgb_list = []
        heatmap_list = []

        for img_path in img_paths:
            res = self.process_image(img_path)
            rgb, overlay = self.generate_avg_heatmap(res)
            rgb_list.append(rgb)
            heatmap_list.append(overlay)

        # 定义视角标签（用于原始图像）
        view_labels = []
        for i in range(num_imgs):
            if i == 0:
                view_labels.append("Ground")
            else:
                view_labels.append(f"Aerial {i}")

        cols = 2 * num_imgs
        fig = plt.figure(figsize=(cols * 1.5, 3.5), dpi=300)
        gs = gridspec.GridSpec(1, cols, wspace=0.05)

        for i in range(num_imgs):
            # 原始图像
            ax_img = fig.add_subplot(gs[0, 2*i])
            ax_img.imshow(rgb_list[i])
            ax_img.axis('off')
            ax_img.set_xlabel(view_labels[i], fontsize=10, fontweight='bold')
            
            # 热力图
            ax_heat = fig.add_subplot(gs[0, 2*i + 1])
            ax_heat.imshow(heatmap_list[i])
            ax_heat.axis('off')

        plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.12)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved single-row analysis to {output_path}")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--images", nargs='+', required=True,
                        help="List of images of the same ID in order (ground to aerial top-down)")
    parser.add_argument("--output", default="demo/Attn_Map_Output/single_row_no_line.pdf")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")
    analyzer = SingleRowAnalyzer(cfg)
    analyzer.analyze_and_plot(args.images, args.output)

"""
CUDA_VISIBLE_DEVICES=0 python3 demo/query_to_viewpoint_change.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--images demo/view_point_change/0.jpg demo/view_point_change/1.jpg demo/view_point_change/2.jpg demo/view_point_change/3.jpg \
--output ./demo/Attn_Map_Output/controlled_evolution.pdf \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth
"""