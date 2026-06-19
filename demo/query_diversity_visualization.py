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
import matplotlib.patches as mpatches

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

# 颜色表 (RGB)
PART_COLORS = [
    (228, 26, 28),   # Soft Red
    (55, 126, 184),  # Soft Blue
    (77, 175, 74),   # Soft Green
    (152, 78, 163),  # Soft Purple
    (255, 127, 0),   # Orange
    (255, 255, 51),  # Yellow
    (166, 86, 40),   # Brown
    (247, 129, 191)  # Pink
]

class PaperFinalVisualizer:
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

        self.hooks = {'vad': None, 'avd_weights': None}
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
        
        def hook_avd(module, input, output):
            if isinstance(output, tuple) and len(output) >= 5:
                self.hooks['avd_weights'] = output[4].detach().cpu()

        if hasattr(self.model, 'module'): 
            base_vad = self.model.module.pavd.vad
            base_avd = self.model.module.pavd.avd
        else: 
            base_vad = self.model.pavd.vad
            base_avd = self.model.pavd.avd
        
        base_vad.register_forward_hook(hook_vad)
        base_avd.register_forward_hook(hook_avd)

    def process_image(self, img_path):
        raw_img = cv2.imread(img_path)
        raw_resized = cv2.resize(raw_img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_CUBIC)
        rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)
        
        pil = Image.open(img_path).convert('RGB')
        tensor = self.transform(pil).unsqueeze(0).to(self.device)
        
        self.hooks['vad'] = None
        self.hooks['avd_weights'] = None
        
        with torch.no_grad():
            inputs = {"images": tensor}
            if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
                 inputs.update({'targets': torch.tensor([0]), 'camids': torch.tensor([0]), 'viewids': ['Aerial']})
                 self.model(inputs)
            else:
                 self.model(tensor)
                 
        return {'rgb': rgb, 'vad': self.hooks['vad'], 'avd_weights': self.hooks['avd_weights']}

    def run(self, img_paths, output_dir, filename, num_queries=4, threshold=0.55):
        os.makedirs(output_dir, exist_ok=True)
        self.plot_paper_figure(img_paths, output_dir, filename, num_queries, threshold)

    def plot_paper_figure(self, img_paths, output_dir, filename, num_queries, threshold):
        num_images = len(img_paths)
        # 列数：Input, Overlay, K个Query热力图, 图例列
        cols = 2 + num_queries + 1
        
        # 宽度比例：前 (2+num_queries) 列宽度为1，最后一列（图例）宽度设为 1.8（可调整）
        width_ratios = [1.0] * (2 + num_queries) + [0.5]
        fig_width = 2.0 * (2 + num_queries) + 2.0 * 1.8
        fig_height = 4.0 * num_images
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        gs = gridspec.GridSpec(num_images, cols, width_ratios=width_ratios, wspace=0.1, hspace=0.1)

        for row_idx, img_path in enumerate(img_paths):
            res = self.process_image(img_path)
            
            attn_vad = res['vad'].squeeze(0)  # [L, N]
            avd_weights = res['avd_weights'].squeeze(0).squeeze(0) if res['avd_weights'] is not None else torch.ones(attn_vad.shape[0])
            
            if attn_vad.shape[1] == (self.input_size[0]//16) * (self.input_size[1]//16) + 1:
                attn_vad = attn_vad[:, 1:]
                
            H, W, _ = res['rgb'].shape
            grid_h, grid_w = self.input_size[0]//16, self.input_size[1]//16
            
            top_indices = torch.argsort(avd_weights, descending=True)[:num_queries].numpy()
            
            L = attn_vad.shape[0]
            all_maps_large = []
            for i in range(L):
                grid = attn_vad[i].reshape(1, 1, grid_h, grid_w)
                large = F.interpolate(grid, size=(H, W), mode='bicubic', align_corners=False).squeeze().numpy()
                all_maps_large.append(large)
            all_maps_large = np.stack(all_maps_large)  # [L, H, W]
            
            # --- Overlay 图: argmax per pixel ---
            argmax_idx = np.argmax(all_maps_large, axis=0)
            max_vals = np.max(all_maps_large, axis=0)
            global_max = np.max(max_vals)
            bg_threshold = global_max * threshold 
            
            overlay_img = res['rgb'].astype(np.float32) / 255.0
            colored_mask = np.zeros_like(overlay_img)
            for i in range(L):
                mask = (argmax_idx == i) & (max_vals > bg_threshold)
                color = np.array(PART_COLORS[i % len(PART_COLORS)]) / 255.0
                colored_mask[mask] = color
                
            alpha = 0.55
            overlay_final = overlay_img.copy()
            active_pixels = max_vals > bg_threshold
            overlay_final[active_pixels] = overlay_img[active_pixels] * (1-alpha) + colored_mask[active_pixels] * alpha
            
            # ----- 绘图 -----
            # 第0列: 原始图像
            ax_in = fig.add_subplot(gs[row_idx, 0])
            ax_in.imshow(res['rgb'])
            ax_in.axis('off')
            if row_idx == 0: 
                ax_in.set_title("Input", fontsize=14, pad=10)
            # 固定视角标签：第一行 Aerial View，第二行 Ground View
            view_title = "Aerial View" if row_idx == 0 else "Ground View"
            ax_in.text(-0.15, 0.5, view_title, transform=ax_in.transAxes, rotation=90,
                       va='center', ha='center', fontsize=14, fontweight='bold')

            # 第1列: Overlay
            ax_ov = fig.add_subplot(gs[row_idx, 1])
            ax_ov.imshow(overlay_final)
            ax_ov.axis('off')
            if row_idx == 0: 
                ax_ov.set_title("Overlay", fontsize=14, pad=10)

            # 第2列至第2+num_queries-1列: 每个 Top-K Query 的热力图
            pixel_max = np.max(all_maps_large, axis=0, keepdims=True) + 1e-8
            for col_offset, q_idx in enumerate(top_indices):
                att = all_maps_large[q_idx]
                abs_norm = att / (global_max + 1e-8)
                rel_dominance = att / pixel_max[0]
                att_final = abs_norm * (rel_dominance ** 2.0)
                att_final = np.clip(att_final ** 1.2, 0, 1)
                
                heatmap = cv2.applyColorMap(np.uint8(255 * att_final), cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                blend = 0.4 * overlay_img + 0.6 * heatmap
                
                ax_q = fig.add_subplot(gs[row_idx, 2 + col_offset])
                ax_q.imshow(blend)
                ax_q.axis('off')
                if row_idx == 0: 
                    ax_q.set_title(f"Query{q_idx}", fontsize=14, pad=10)

            # 最后一列: 图例（仅第一行绘制，第二行留白）
            if row_idx == 0:
                ax_leg = fig.add_subplot(gs[row_idx, -1])
                ax_leg.axis('off')
                # 不设置标题
                patches = []
                for idx, q_idx in enumerate(top_indices):
                    color = np.array(PART_COLORS[q_idx % len(PART_COLORS)]) / 255.0
                    patch = mpatches.Patch(color=color, label=f"Query {q_idx}")
                    patches.append(patch)
                if L > num_queries:
                    patches.append(mpatches.Patch(color='gray', label=f"... + {L - num_queries} more"))
                ax_leg.legend(handles=patches, loc='center', fontsize=12,
                              handlelength=1.5, handleheight=1.5, frameon=False)
            else:
                # 第二行图例列为空白
                ax_leg = fig.add_subplot(gs[row_idx, -1])
                ax_leg.axis('off')

        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
        save_path = os.path.join(output_dir, filename + ".pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Success! Figure saved to {save_path}")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--input", nargs='+', required=True) 
    parser.add_argument("--output", default="vis_output")
    parser.add_argument("--filename", default="rebuttal_fig_2")
    parser.add_argument("--num_queries", type=int, default=4, help="Top K queries to show")
    parser.add_argument("--threshold", type=float, default=0.55, help="Background threshold for Overlay (0.0 to 1.0)")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")

    vis = PaperFinalVisualizer(cfg)
    vis.run(args.input, args.output, args.filename, args.num_queries, args.threshold)
"""
python3 demo/query_diversity_visualization.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--input \
demo/input/P0004T02140A0C0F1321.jpg \
demo/input/P0004T02140A0C3F991.jpg \
--output ./demo/Attn_Map_Output \
--filename rebuttal_fig_distinct_1 \
--num_queries 4 \
--threshold 0.55 \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth

python3 demo/query_diversity_visualization.py \
--config-file 之前的消融/AGReID参数分析/num_prompts/8/config.yml \
--input \
demo/input/P0004T02140A0C0F1321.jpg \
demo/input/P0004T02140A0C3F991.jpg \
--output ./demo/Attn_Map_Output \
--filename rebuttal_fig_distinct_8 \
--num_queries 8 \
--threshold 0.4 \
MODEL.WEIGHTS 之前的消融/AGReID参数分析/num_prompts/8/model_best.pth
"""