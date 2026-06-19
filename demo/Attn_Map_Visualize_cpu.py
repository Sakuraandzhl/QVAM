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

        self.hooks = {'vad': None}
        self._register_hooks()

        self.input_size = cfg.INPUT.SIZE_TEST
        self.transform = T.Compose([
            T.Resize(self.input_size, interpolation=BICUBIC),
            ToTensor(), 
        ])

    def _register_hooks(self):
        def hook_vad(module, input, output):
            if isinstance(output, tuple): self.hooks['vad'] = output[1].detach().cpu()
        
        if hasattr(self.model, 'module'): base = self.model.module.pavd
        else: base = self.model.pavd
        
        base.vad.register_forward_hook(hook_vad)

    def process_image(self, img_path):
        if not os.path.exists(img_path):
            raise ValueError(f"Image not found: {img_path}")
            
        raw_img = cv2.imread(img_path)
        if raw_img is None: raise ValueError(f"Err loading: {img_path}")
        # Resize raw image for visualization
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

    # -------------------------------------------------------------------------
    # [核心] 生成平均热力图
    # -------------------------------------------------------------------------
    def generate_avg_heatmap(self, res):
        attn_vad = res['vad'] # [B, L, N] or [L, N]
        if attn_vad.shape[0] == 1: attn_vad = attn_vad.squeeze(0)
        
        # 1. 对所有 Prompt 求平均 -> [N]
        # 这一步融合了所有 Prompt 的关注点，展示模型的整体注意力
        avg_attn = torch.mean(attn_vad, dim=0) 
        
        H, W, _ = res['rgb'].shape
        grid_h, grid_w = 16, 8
        
        # 2. 还原空间维度 [H_grid, W_grid]
        attn_grid = avg_attn.reshape(1, 1, grid_h, grid_w)
        
        # 3. 插值回原图大小 [1, 1, H, W]
        attn_large = F.interpolate(attn_grid, size=(H, W), mode='bicubic', align_corners=False).squeeze()
        
        # 4. 归一化 (0~1)
        att = attn_large - attn_large.min()
        att = att / (att.max() + 1e-8)
        
        # 5. 生成热力图
        heatmap = cv2.applyColorMap(np.uint8(255 * att.numpy()), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # 6. 叠加
        original_img = res['rgb'].astype(np.float32) / 255.0
        overlay = 0.6 * original_img + 0.4 * heatmap
        
        return original_img, overlay

    def run(self, img_paths, output_dir, filename="vis_comparison"):
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析输入对：[A1, G1, A2, G2, ...]
        if len(img_paths) % 2 != 0:
            raise ValueError("Input images must be in pairs (Aerial, Ground)")
            
        pairs = []
        for i in range(0, len(img_paths), 2):
            pairs.append((img_paths[i], img_paths[i+1]))
            
        self.plot_comparison(pairs, output_dir, filename)

    def plot_comparison(self, pairs, output_dir, filename):
        num_pairs = len(pairs)
        # 列数 = 2 * ID数量 (每组由 原图 + 热力图 组成)
        cols = num_pairs * 2
        
        # 动态计算画布大小，保持单个图像的宽高比
        # 假设单个图像宽高比约 0.5 (128/256)
        # 每个单元格宽 2, 高 4 -> 画布宽 2*cols, 高 4*2(rows)
        fig_width = 2.0 * cols
        fig_height = 8.0 
        
        fig = plt.figure(figsize=(fig_width, fig_height), dpi=300)
        
        # wspace=0.05: 组内紧凑
        # 组间可以通过插入空列或者手动调整，这里先用统一间距
        gs = gridspec.GridSpec(2, cols, wspace=0.05, hspace=0.02)

        for idx, (path_aerial, path_ground) in enumerate(pairs):
            # 处理 Aerial
            res_a = self.process_image(path_aerial)
            img_a, map_a = self.generate_avg_heatmap(res_a)
            
            # 处理 Ground
            res_g = self.process_image(path_ground)
            img_g, map_g = self.generate_avg_heatmap(res_g)
            
            # 基础列索引
            base_col = idx * 2
            
            # === Row 1: Aerial ===
            # Image
            ax1 = fig.add_subplot(gs[0, base_col])
            ax1.imshow(img_a)
            ax1.axis('off')
            # 仅在第一个ID显示 View 标签
            if idx == 0:
                ax1.text(-0.15, 0.5, "Aerial View", transform=ax1.transAxes, 
                         rotation=90, va='center', ha='right', fontsize=14, fontweight='bold')
            # 标题
            ax1.set_title(f"ID {idx+1}", fontsize=12)

            # Heatmap
            ax2 = fig.add_subplot(gs[0, base_col + 1])
            ax2.imshow(map_a)
            ax2.axis('off')
            ax2.set_title("Attention", fontsize=12)

            # === Row 2: Ground ===
            # Image
            ax3 = fig.add_subplot(gs[1, base_col])
            ax3.imshow(img_g)
            ax3.axis('off')
            if idx == 0:
                ax3.text(-0.15, 0.5, "Ground View", transform=ax3.transAxes, 
                         rotation=90, va='center', ha='right', fontsize=14, fontweight='bold')
            
            # Heatmap
            ax4 = fig.add_subplot(gs[1, base_col + 1])
            ax4.imshow(map_g)
            ax4.axis('off')

        # 调整边距
        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.05)
        
        save_path = os.path.join(output_dir, filename + ".pdf")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved Comparison to {save_path}")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    # 接受任意数量的输入，必须成对：A1 G1 A2 G2 ...
    parser.add_argument("--input", nargs='+', required=True) 
    parser.add_argument("--output", default="vis_comparison")
    parser.add_argument("--filename", default="vis_avg_heatmap", help="output file name")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")

    vis = PaperFinalVisualizer(cfg)
    vis.run(args.input, args.output, args.filename)
'''
python3 demo/Attn_Map_Visualize_cpu.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--input \
demo/input/AGReID/P0004T04041A0C0F7591.jpg demo/input/AGReID/P0004T04041A0C3F601.jpg \
demo/input/AGReID/P0004T02140A0C0F1231.jpg demo/input/AGReID/P0004T02140A0C3F841.jpg \
demo/input/AGReID/P0006T02140A0C0F2371.jpg demo/input/AGReID/P0006T02140A0C0F1951.jpg \
demo/input/AGReID/P0116T04041A1C0F3901.jpg demo/input/AGReID/P0116T04041A1C3F481.jpg \
--output ./demo/Attn_Map_Output \
--filename heat_map_cargo \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth
'''

'''CARGO
python3 demo/Attn_Map_Visualize_cpu.py \
--config-file logs_6/34597178_7.1/True/1.0_0.01_1.0/config.yml \
--input \
demo/input/Cam1_day_3_55.jpg \
demo/input/Cam12_day_3_22433.jpg \
--output ./demo/Attn_Map_Output \
--filename heat_map_cargo \
--threshold 0.5 \
MODEL.WEIGHTS logs_6/34597178_7.1/True/1.0_0.01_1.0/model_best.pth
'''

"""AGReIDv2
python3 demo/Attn_Map_Visualize_cpu.py \
--config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
--input \
demo/input/AGReIDv2/P0013T04070A1C0F00601.jpg \
demo/input/AGReIDv2/P0013T04070A1C2F00091.jpg \
--output ./demo/Attn_Map_Output \
--filename heat_map_cargo \
--threshold 0.45 \
MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth

"""


# # encoding: utf-8
# import argparse
# import sys
# import os
# import cv2
# import numpy as np
# import torch
# import torch.nn.functional as F
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import matplotlib.gridspec as gridspec

# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     from PIL import Image as PILImage
#     BICUBIC = PILImage.BICUBIC

# sys.path.append('.') 

# from fastreid.config import get_cfg
# from fastreid.modeling.meta_arch import build_model
# from fastreid.utils.checkpoint import Checkpointer
# from fastreid.utils.logger import setup_logger
# from fastreid.data.transforms import *

# # =========================================================
# # [修复] 字体回退策略
# # 优先找 Times New Roman，找不到就找 Times，再找不到就用通用的 serif
# # 这样在没有安装微软字体的 Linux 服务器上也不会报错
# # =========================================================
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'Times', 'DejaVu Serif', 'Liberation Serif', 'serif']
# plt.rcParams['mathtext.fontset'] = 'stix' # 让数学公式字体也像 Times

# class PaperFinalVisualizer:
#     def __init__(self, cfg):
#         self.device = torch.device("cpu")
#         cfg.defrost()
#         cfg.MODEL.DEVICE = "cpu"
#         cfg.MODEL.BACKBONE.PRETRAIN = False
#         self.cfg = cfg
        
#         print("Building model...")
#         self.model = build_model(cfg)
#         self.model.eval()
#         self.model.to(self.device)
#         Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

#         self.hooks = {'vad': None, 'avd': None}
#         self._register_hooks()

#         self.input_size = cfg.INPUT.SIZE_TEST
#         self.transform = T.Compose([
#             T.Resize(self.input_size, interpolation=BICUBIC),
#             ToTensor(), 
#         ])
        
#         # 扩展调色板 (应对 Top-K 筛选)
#         self.fixed_palette = [
#             '#00897B', '#D35400', '#C0392B', '#2874A6', 
#             '#7D3C98', '#566573', '#1E8449', '#B03A2E',
#             '#2E4053', '#CA6F1E', '#17A589', '#6C3483' 
#         ]
#         self.line_colors = {'ground': '#2E86C1', 'aerial': '#C0392B'}

#     def _register_hooks(self):
#         def hook_vad(module, input, output):
#             if isinstance(output, tuple): self.hooks['vad'] = output[1].detach().cpu()
#         def hook_avd(module, input, output):
#             if isinstance(output, tuple): self.hooks['avd'] = output[1].detach().cpu()

#         if hasattr(self.model, 'module'): base = self.model.module.pavd
#         else: base = self.model.pavd
        
#         base.vad.register_forward_hook(hook_vad)
#         if hasattr(base.avd, 'adaptive_view_disentangle'):
#             base.avd.adaptive_view_disentangle[-1]['cross_attn'].register_forward_hook(hook_avd)

#     def process_image(self, img_path):
#         raw_img = cv2.imread(img_path)
#         if raw_img is None: raise ValueError(f"Err: {img_path}")
#         raw_resized = cv2.resize(raw_img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_CUBIC)
#         rgb = cv2.cvtColor(raw_resized, cv2.COLOR_BGR2RGB)
        
#         pil = Image.open(img_path).convert('RGB')
#         tensor = self.transform(pil).unsqueeze(0).to(self.device)
        
#         self.hooks['vad'] = None
#         self.hooks['avd'] = None
#         with torch.no_grad():
#             inputs = {"images": tensor}
#             if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
#                  inputs.update({'targets': torch.tensor([0]), 'camids': torch.tensor([0]), 'viewids': ['Aerial']})
#                  self.model(inputs)
#             else:
#                  self.model(tensor)
#         return {'rgb': rgb, 'vad': self.hooks['vad'], 'avd': self.hooks['avd']}

#     def get_overlay_topk(self, res, threshold=0.4, topk=4):
#         # 1. 获取 VAD
#         attn_vad = res['vad'] 
#         if attn_vad.shape[0] == 1: attn_vad = attn_vad.squeeze(0) 
#         if attn_vad.ndim == 3: attn_vad = torch.mean(attn_vad, dim=0) 
        
#         H, W, _ = res['rgb'].shape
#         grid_h, grid_w = 16, 8
#         num_prompts = attn_vad.shape[0]

#         # 2. 获取 AVD (重要性权重)
#         attn_avd = res['avd']
#         if attn_avd.shape[-1] != num_prompts: pass
#         if attn_avd.dim() > 1:
#             attn_avd = attn_avd.reshape(-1, num_prompts)
#             attn_avd = torch.mean(attn_avd, dim=0)
#         raw_weights = attn_avd 

#         # [核心逻辑] Top-K 筛选
#         k = min(topk, num_prompts)
#         topk_vals, topk_indices = torch.topk(raw_weights, k=k)
        
#         # [核心逻辑] Softmax 加权计算 Alpha
#         softmax_weights = F.softmax(topk_vals, dim=0)
#         prompt_alphas = 0.3 + 0.6 * softmax_weights

#         # 3. 制作 Top-K 的空间 Mask
#         attn_grid = attn_vad.reshape(num_prompts, grid_h, grid_w)
#         topk_attn_grid = attn_grid[topk_indices].unsqueeze(0) 
#         attn_large = torch.nn.functional.interpolate(topk_attn_grid, size=(H, W), mode='bicubic').squeeze(0)
#         max_vals, local_indices = torch.max(attn_large, dim=0) 
#         norm = (max_vals - max_vals.min()) / (max_vals.max() - max_vals.min() + 1e-8)
#         is_bg = norm < threshold
        
#         # 4. 绘制
#         colored = np.zeros((H, W, 3), dtype=np.uint8)
#         alpha_mask = np.zeros((H, W), dtype=np.float32)
        
#         active_prompts = []
#         for i, p_idx in enumerate(topk_indices.numpy()):
#             locs = (local_indices == i) & (~is_bg)
#             if not locs.any(): continue
            
#             active_prompts.append(p_idx)
            
#             hex_color = self.fixed_palette[p_idx % len(self.fixed_palette)].lstrip('#')
#             rgb_color = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
            
#             p_alpha = prompt_alphas[i].item()
#             colored[locs] = rgb_color
#             alpha_mask[locs] = p_alpha

#         # 5. 混合
#         overlay = res['rgb'].astype(np.float32)
#         colored = colored.astype(np.float32)
#         alpha_3ch = np.stack([alpha_mask]*3, axis=-1)
#         mixed = overlay * (1 - alpha_3ch) + colored * alpha_3ch
#         mixed = np.clip(mixed, 0, 255).astype(np.uint8)
#         bg_locs = is_bg.numpy()
#         mixed[bg_locs] = res['rgb'][bg_locs]
        
#         return mixed, active_prompts, topk_vals.numpy(), topk_indices.numpy()

#     def run(self, img_paths, output_dir, threshold=0.4):
#         os.makedirs(output_dir, exist_ok=True)
#         res_ground = self.process_image(img_paths[0])
#         res_aerial = self.process_image(img_paths[1])
#         self.plot_layout(res_ground, res_aerial, output_dir, threshold)

#     def plot_layout(self, r_g, r_a, output_dir, thresh):
#         fig = plt.figure(figsize=(4.8, 5.5), dpi=300) 
#         gs = gridspec.GridSpec(2, 2, height_ratios=[2.5, 1], width_ratios=[1, 1])
#         gs.update(wspace=0.08, hspace=0.12) 

#         img_g, p_g, vals_g, idx_g = self.get_overlay_topk(r_g, thresh)
#         img_a, p_a, vals_a, idx_a = self.get_overlay_topk(r_a, thresh)
        
#         y_min = min(vals_g.min(), vals_a.min())
#         y_max = max(vals_g.max(), vals_a.max())
#         margin = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 0.01
#         y_lims = (y_min - margin, y_max + margin)

#         # === Row 0: Images ===
#         ax_img_g = fig.add_subplot(gs[0, 0])
#         ax_img_g.imshow(img_g)
#         ax_img_g.set_title("Ground View", fontsize=14, fontweight='normal', pad=5)
#         ax_img_g.axis('off')

#         ax_img_a = fig.add_subplot(gs[0, 1])
#         ax_img_a.imshow(img_a)
#         ax_img_a.set_title("Aerial View", fontsize=14, fontweight='normal', pad=5)
#         ax_img_a.axis('off')

#         # === Row 1: Charts ===
#         ax_chart_g = fig.add_subplot(gs[1, 0])
#         self.plot_compact_chart(ax_chart_g, vals_g, idx_g, self.line_colors['ground'], y_lims)
        
#         ax_chart_a = fig.add_subplot(gs[1, 1])
#         self.plot_compact_chart(ax_chart_a, vals_a, idx_a, self.line_colors['aerial'], y_lims)

#         # === Legend ===
#         all_active_prompts = set(p_g) | set(p_a)
#         patches = []
#         for p in sorted(list(all_active_prompts)):
#             patches.append(mpatches.Patch(
#                 color=self.fixed_palette[p % len(self.fixed_palette)], 
#                 label=f'Prompt {p}'
#             ))
        
#         fig.legend(handles=patches, loc='lower center', ncol=len(patches), 
#                    bbox_to_anchor=(0.5, 0.01), fontsize=9, frameon=False,
#                    columnspacing=0.5, handletextpad=0.3, borderpad=0)

#         plt.subplots_adjust(left=0.12, right=0.95, top=0.92, bottom=0.10)
        
#         save_path = os.path.join(output_dir, args.filename + ".pdf")
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         plt.close()
#         print(f"✅ Saved Top-4 Vis (Font Fixed) to {save_path}")

#     def plot_compact_chart(self, ax, vals, indices, color, y_lims):
#         x = range(len(vals))
#         ax.plot(x, vals, marker='o', linestyle='-', color=color, 
#                 linewidth=1.8, markersize=5, alpha=0.9)
        
#         ax.set_box_aspect(0.9) 
        
#         ax.set_ylim(y_lims)
#         ax.set_xticks(x)
#         ax.set_xticklabels([f"P{i}" for i in indices], fontsize=9)
#         ax.tick_params(axis='y', labelsize=8)
        
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)
#         ax.grid(True, linestyle=':', alpha=0.4)
        
#         max_i = np.argmax(vals)
#         ax.plot(max_i, vals[max_i], marker='o', color='#F9D367', 
#                 markersize=6, markeredgecolor=color, zorder=10)

# def setup_cfg(args):
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     return cfg

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config-file", required=True)
#     parser.add_argument("--input", nargs=2, required=True)
#     parser.add_argument("--output", default="vis_paper_final")
#     parser.add_argument("--threshold", type=float, default=0.4)
#     parser.add_argument("--filename", default="paper_vis_topk", help="output file name")
#     parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
#     args = parser.parse_args()

#     cfg = setup_cfg(args)
#     setup_logger(name="fastreid")

#     vis = PaperFinalVisualizer(cfg)
#     vis.run(args.input, args.output, args.threshold)









# # encoding: utf-8
# import argparse
# import sys
# import os
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image
# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches

# # 尝试导入插值模式
# try:
#     from torchvision.transforms import InterpolationMode
#     BICUBIC = InterpolationMode.BICUBIC
# except ImportError:
#     from PIL import Image as PILImage
#     BICUBIC = PILImage.BICUBIC

# # 添加项目根目录到路径
# sys.path.append('.') 

# from fastreid.config import get_cfg
# from fastreid.modeling.meta_arch import build_model
# from fastreid.utils.checkpoint import Checkpointer
# from fastreid.utils.logger import setup_logger
# from fastreid.data.transforms import *


# class AttentionVisualizer:
#     def __init__(self, cfg):
#         # 1. 强制设置为 CPU
#         self.device = torch.device("cpu")
        
#         cfg.defrost()
#         cfg.MODEL.DEVICE = "cpu"
#         cfg.MODEL.BACKBONE.PRETRAIN = False
#         self.cfg = cfg
        
#         # 2. 构建模型
#         print("Building model (on CPU)...")
#         self.model = build_model(cfg)
#         self.model.eval()
#         self.model.to(self.device)

#         # 3. 加载权重
#         print(f"Loading checkpoint from {cfg.MODEL.WEIGHTS}...")
#         Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

#         # 4. 注册 Hook 捕获 Attention Weights
#         self.captured_attn = None       # VAD (Prompt -> Patch)
#         self.captured_mask_attn = None  # AVD (Mask -> Prompt) [新增]
#         self._register_hook()           # 原有的 Hook
#         self._register_avd_hook()       # [新增] AVD Hook

#         # 5. 图像预处理
#         self.input_size = cfg.INPUT.SIZE_TEST 
        
#         self.transform = T.Compose([
#             T.Resize(self.input_size, interpolation=3),
#             ToTensor(), 
#         ])
        
#         self.palette = self._generate_palette(64)

#     def _generate_palette(self, n):
#         colors = []
#         cmaps = ['tab20', 'Set1', 'Set2', 'Set3', 'tab20b', 'tab20c']
#         idx = 0
#         while len(colors) < n:
#             cmap_name = cmaps[idx % len(cmaps)]
#             cmap = plt.get_cmap(cmap_name)
#             num_c = cmap.N
#             for i in range(num_c):
#                 if len(colors) >= n: break
#                 rgba = cmap(i)
#                 color = (int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255))
#                 colors.append(color)
#             idx += 1
#         return colors

#     def _register_hook(self):
#         """原有的 VAD Hook"""
#         def hook_fn(module, input, output):
#             if isinstance(output, tuple):
#                 attn_weights = output[1]
#                 self.captured_attn = attn_weights.detach().cpu()
#             elif isinstance(output, torch.Tensor):
#                 self.captured_attn = output.detach().cpu()
#             else:
#                 print("❌ Unknown output type in hook.")

#         try:
#             if hasattr(self.model, 'module'):
#                 target_module = self.model.module.pavd.vad
#             else:
#                 target_module = self.model.pavd.vad
            
#             target_module.register_forward_hook(hook_fn)
#             print("✅ Successfully hooked into PAVD.VAD module.")
#         except AttributeError as e:
#             print(f"❌ Error hooking VAD module: {e}")
#             sys.exit(1)

#     # ---------------- [新增] AVD Hook 函数 ----------------
#     def _register_avd_hook(self):
#         """Hook 到 AVD 模块的 Cross-Attention 以获取 Mask 对 Prompts 的注意力"""
#         def hook_fn_avd(module, input, output):
#             # AVD 的 CrossAttn 输出通常是 (output, weights)
#             if isinstance(output, tuple):
#                 self.captured_mask_attn = output[1].detach().cpu()
#             else:
#                 print("❌ AVD Hook output is not tuple")

#         try:
#             # 定位 AVD 模块
#             if hasattr(self.model, 'module'):
#                 avd_module = self.model.module.pavd.avd
#             else:
#                 avd_module = self.model.pavd.avd
            
#             # Hook 到最后一层 AVD Block 的 Cross Attention
#             # 根据提供的代码结构: self.adaptive_view_disentangle 是个 ModuleList
#             # 我们取最后一层 [-1]
#             if hasattr(avd_module, 'adaptive_view_disentangle'):
#                 target_layer = avd_module.adaptive_view_disentangle[-1]['cross_attn']
#                 target_layer.register_forward_hook(hook_fn_avd)
#                 print("✅ Successfully hooked into PAVD.AVD Cross-Attention.")
#             else:
#                 print("⚠️ AVD module does not have 'adaptive_view_disentangle'. Skip AVD hooking.")
#         except Exception as e:
#             print(f"❌ Error hooking AVD module: {e}")
#     # -----------------------------------------------------

#     def parse_filename(self, img_path):
#         filename = os.path.basename(img_path)
#         parts = filename.split('_')
#         cam_str = parts[0]
#         try:
#             camid = int(cam_str[3:])
#             pid = int(parts[2])
#             viewid_str = 'Aerial' if camid <= 5 else 'Ground'
#             camid_idx = camid - 1
#         except Exception:
#             print("⚠️ Filename parsing failed, using default dummy values.")
#             pid = 0
#             camid = 0
#             viewid_str = 'Unknown'
#             camid_idx = 0
#         return pid, camid_idx, viewid_str

#     def preprocess_image(self, img_path):
#         raw_img = cv2.imread(img_path)
#         if raw_img is None:
#             raise ValueError(f"Could not load image: {img_path}")
            
#         raw_img_resized = cv2.resize(
#             raw_img, 
#             (self.input_size[1], self.input_size[0]), 
#             interpolation=cv2.INTER_CUBIC
#         )
#         raw_img_rgb = cv2.cvtColor(raw_img_resized, cv2.COLOR_BGR2RGB)

#         pil_img = Image.open(img_path).convert('RGB')
#         img_tensor = self.transform(pil_img) 
        
#         # 保持你的原逻辑，不做任何 *255 修改
        
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
#         return img_tensor, raw_img_rgb

#     def run(self, img_path, output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         img_name = os.path.basename(img_path).split('.')[0]

#         pid, camid, viewid_str = self.parse_filename(img_path)
#         print(f"Processing {img_name} | PID: {pid} | View: {viewid_str}")

#         img_tensor, raw_img_rgb = self.preprocess_image(img_path)

#         self.captured_attn = None 
#         self.captured_mask_attn = None # 重置
        
#         with torch.no_grad():
#             inputs = {
#                 "images": img_tensor,
#                 "targets": torch.tensor([pid], device=self.device),
#                 "camids": torch.tensor([camid], device=self.device),
#                 "viewids": [viewid_str] 
#             }
#             if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
#                  self.model(inputs)
#             else:
#                  self.model(img_tensor)

#         # 1. 原有的 VAD 可视化
#         if self.captured_attn is not None:
#             self._process_vad_attn(self.captured_attn, raw_img_rgb, output_dir, img_name, viewid_str)
#         else:
#             print("❌ Failed to capture VAD attention map.")

#         # 2. [新增] AVD Mask-Prompt 可视化
#         if self.captured_mask_attn is not None:
#             self._plot_mask_prompt_attn(self.captured_mask_attn, output_dir, img_name, viewid_str)
#         else:
#             print("⚠️ Failed to capture AVD mask attention (Or AVD not running).")

#     def _process_vad_attn(self, attn, raw_img_rgb, output_dir, img_name, viewid_str):
#         # 你的原 VAD 处理逻辑封装在这里
#         if attn.shape[0] == 1:
#             attn = attn.squeeze(0)
#         if attn.ndim == 3: 
#              if attn.shape[0] < attn.shape[1]: 
#                 attn = torch.mean(attn, dim=0)
        
#         num_prompts, num_patches = attn.shape
#         H_in, W_in = self.input_size
#         patch_w = W_in // 16 
#         patch_h = num_patches // patch_w
        
#         print(f"VAD Attention Shape: {attn.shape} | Grid: {patch_h}x{patch_w}")
        
#         if patch_h * patch_w != num_patches:
#             print(f"⚠️ Warning: Calculated grid mismatch.")

#         self.visualize_segmentation_style(attn, raw_img_rgb, patch_h, patch_w, output_dir, img_name, view_tag=viewid_str)

# # ---------------- [修改版] 放大差异的折线图 ----------------
#     def _plot_mask_prompt_attn(self, attn_tensor, output_dir, img_name, view_tag):
#         """
#         绘制 Mask Token 对各个 Prompt 的注意力权重折线图 (自动缩放 Y 轴以放大差异)
#         """
#         # 1. 数据处理
#         if attn_tensor.shape[0] == 1:
#             attn_tensor = attn_tensor.squeeze(0)
#         if attn_tensor.ndim == 3:
#             attn_tensor = torch.mean(attn_tensor, dim=0)
#         attn_vals = attn_tensor.squeeze().numpy()
#         num_prompts = len(attn_vals)
        
#         print(f"AVD Mask Attention Shape: {attn_vals.shape}")
        
#         # 计算统计量
#         y_min, y_max = attn_vals.min(), attn_vals.max()
#         y_mean = attn_vals.mean()
#         y_range = y_max - y_min
        
#         # 2. 绘图
#         plt.figure(figsize=(10, 5))
        
#         # 绘制主折线
#         plt.plot(range(num_prompts), attn_vals, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
#         # 根据是否高于平均值，给点上不同的颜色
#         colors = ['red' if v >= y_mean else 'dodgerblue' for v in attn_vals]
#         sizes = [80 if v >= y_mean else 40 for v in attn_vals] # 重要的点画大一点
        
#         plt.scatter(range(num_prompts), attn_vals, c=colors, s=sizes, zorder=5)
        
#         # 连线 (用插值平滑曲线可能会误导，还是用直线连接但加粗高亮部分)
#         plt.plot(range(num_prompts), attn_vals, color='black', linewidth=1.5, alpha=0.7)

#         plt.xlabel('Prompt ID', fontsize=12)
#         plt.ylabel('Attention Weight (Zoomed In)', fontsize=12)
#         plt.title(f'{view_tag} View - Mask Attention Focus\n(Dashed Line = Mean: {y_mean:.4f})', fontsize=14)
        
#         # 画平均线
#         plt.axhline(y=y_mean, color='green', linestyle=':', label='Mean', alpha=0.6)
        
#         plt.grid(True, linestyle='--', alpha=0.3)
        
#         # --- 核心修改：动态调整 Y 轴范围 (显微镜模式) ---
#         # 如果方差极小，这就很重要。为了防止 range=0 报错，加个极小值
#         margin = max(y_range * 0.5, 0.001) 
#         plt.ylim(y_min - margin, y_max + margin)
        
#         # 标注每个点的数值
#         for i, val in enumerate(attn_vals):
#             offset = margin * 0.1
#             font_weight = 'bold' if val == y_max else 'normal'
#             plt.text(i, val + offset, f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight=font_weight)

#         # 标注最大值
#         max_idx = np.argmax(attn_vals)
#         plt.annotate(f'Max Focus', 
#                      xy=(max_idx, attn_vals[max_idx]), 
#                      xytext=(max_idx, attn_vals[max_idx] + margin * 0.5),
#                      arrowprops=dict(facecolor='red', shrink=0.05, width=1, headwidth=6),
#                      ha='center', fontsize=10, color='red')

#         plt.tight_layout()
#         save_path = os.path.join(output_dir + '/line/', f"{img_name}_mask_attn_line_zoomed.jpg")
#         plt.savefig(save_path, dpi=150)
#         plt.close()
#         print(f"✅ Saved Zoomed-in Line Chart to {save_path}")
#     # -------------------------------------------------------------
#     def visualize_segmentation_style(self, attn_map, raw_img_rgb, grid_h, grid_w, output_dir, img_name, view_tag, bg_threshold=0.4):
#         """你的原可视化逻辑"""
#         H_img, W_img, _ = raw_img_rgb.shape
        
#         attn_grid = attn_map.reshape(attn_map.shape[0], grid_h, grid_w).unsqueeze(0)
#         attn_large = torch.nn.functional.interpolate(
#             attn_grid, size=(H_img, W_img), mode='bicubic', align_corners=False
#         ).squeeze(0)
        
#         max_vals, prompt_mask = torch.max(attn_large, dim=0) 
#         max_vals = max_vals.numpy()
#         prompt_mask = prompt_mask.numpy()
        
#         v_min, v_max = max_vals.min(), max_vals.max()
#         if v_max - v_min > 1e-8:
#             norm_vals = (max_vals - v_min) / (v_max - v_min)
#         else:
#             norm_vals = np.zeros_like(max_vals)

#         is_background = norm_vals < bg_threshold

#         colored_mask = np.zeros((H_img, W_img, 3), dtype=np.uint8)
#         unique_prompts = np.unique(prompt_mask)
        
#         for p_idx in unique_prompts:
#             color = self.palette[p_idx % len(self.palette)]
#             pixel_locs = (prompt_mask == p_idx) & (~is_background)
#             colored_mask[pixel_locs] = color

#         alpha = 0.55
#         overlay = raw_img_rgb.copy()
#         fg_indices = ~is_background
        
#         # 加个保护，防止全背景报错
#         if np.sum(fg_indices) > 0:
#             fg_raw = raw_img_rgb[fg_indices]
#             fg_mask = colored_mask[fg_indices]
#             blended_fg = (fg_raw * 0.45 + fg_mask * 0.55).astype(np.uint8)
#             overlay[fg_indices] = blended_fg

#         plt.figure(figsize=(12, 6))
        
#         plt.subplot(1, 2, 1)
#         plt.imshow(overlay)
#         plt.axis('off')
#         plt.title(f"{view_tag} View\nPrompt Regions (Thresh={bg_threshold})")
        
#         plt.subplot(1, 2, 2)
#         vis_mask_white_bg = colored_mask.copy()
#         vis_mask_white_bg[is_background] = [255, 255, 255]
#         plt.imshow(vis_mask_white_bg)
#         plt.axis('off')
#         plt.title("Filtered ID Map")
        
#         valid_mask = prompt_mask.copy()
#         valid_mask[is_background] = -1
#         prompt_counts = {p: np.sum(valid_mask == p) for p in unique_prompts if p != -1}
#         sorted_prompts = sorted(prompt_counts, key=prompt_counts.get, reverse=True)[:10]
        
#         legend_patches = []
#         for p_idx in sorted_prompts:
#             c_norm = [x/255.0 for x in self.palette[p_idx % len(self.palette)]]
#             patch = mpatches.Patch(color=c_norm, label=f'Prompt {p_idx}')
#             legend_patches.append(patch)
            
#         plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
#         plt.tight_layout()
#         save_path = os.path.join(output_dir + '/img/', f"{img_name}_segmentation_thresh1.jpg")
#         plt.savefig(save_path, dpi=150, bbox_inches='tight')
#         plt.close()
        
#         print(f"✅ Saved THRESHOLDED visualization to {save_path}")

# def setup_cfg(args):
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     return cfg

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config-file", required=True)
#     parser.add_argument("--input", required=True, help="Path to single image")
#     parser.add_argument("--output", default="vis_prompts")
#     parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
#     args = parser.parse_args()

#     cfg = setup_cfg(args)
#     setup_logger(name="fastreid")

#     vis = AttentionVisualizer(cfg)
#     vis.run(args.input, args.output)

# # encoding: utf-8
# import argparse
# import sys
# import os
# import cv2
# import numpy as np
# import torch
# import torchvision.transforms as T
# from PIL import Image

# # 添加项目根目录到路径
# sys.path.append('.') 

# from fastreid.config import get_cfg
# from fastreid.modeling.meta_arch import build_model
# from fastreid.utils.checkpoint import Checkpointer
# from fastreid.utils.logger import setup_logger

# # 定义不同 Part 的颜色 (BGR 格式)
# PART_COLORS = [
#     (0, 0, 255),    # Red
#     (0, 255, 0),    # Green
#     (255, 0, 0),    # Blue
#     (0, 255, 255),  # Yellow
#     (255, 255, 0),  # Cyan
#     (255, 0, 255),  # Magenta
#     (0, 165, 255),  # Orange
#     (128, 0, 128),  # Purple
# ]

# class AttentionVisualizer:
#     def __init__(self, cfg):
#         # 1. 强制设置为 CPU
#         self.device = torch.device("cpu")
        
#         cfg.defrost()
#         cfg.MODEL.DEVICE = "cpu"       # 关键：告诉 fastreid 使用 CPU
#         cfg.MODEL.BACKBONE.PRETRAIN = False
#         self.cfg = cfg
        
#         # 2. 构建模型
#         print("Building model (on CPU)...")
#         self.model = build_model(cfg)
#         self.model.eval()
#         self.model.to(self.device)

#         # 3. 加载权重
#         # fastreid 的 Checkpointer 会自动处理 map_location，只要 model 在 cpu 上
#         print(f"Loading checkpoint from {cfg.MODEL.WEIGHTS}...")
#         Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

#         # 4. 注册 Hook
#         self.captured_attn = None 
#         self._register_hook()

#         # 5. 图像预处理
#         # 保持之前的修复：只做 Resize 和 ToTensor，不做 Normalize
#         self.transform = T.Compose([
#             T.Resize(cfg.INPUT.SIZE_TEST),
#             T.ToTensor(), 
#         ])
        
#         self.input_size = cfg.INPUT.SIZE_TEST 

#     def _register_hook(self):
#         def hook_fn(module, input, output):
#             # PAD.forward 返回 (part_feats, attn_weights)
#             _, attn_weights = output
#             self.captured_attn = attn_weights.detach().cpu()

#         try:
#             # 兼容 DataParallel (虽然 CPU 一般不用，但为了健壮性保留)
#             if hasattr(self.model, 'module'):
#                 target_module = self.model.module.pavd.vad
#             else:
#                 target_module = self.model.pavd.vad
            
#             target_module.register_forward_hook(hook_fn)
#             print("✅ Successfully hooked into PGVD.PAD module.")
#         except AttributeError as e:
#             print(f"❌ Error hooking module: {e}")
#             sys.exit(1)

#     def parse_filename(self, img_path):
#         """
#         根据 CARGO 数据集格式解析文件名
#         Example: Cam8_day_1_50970.jpg
#         """
#         filename = os.path.basename(img_path)
#         parts = filename.split('_')
        
#         # 解析 Cam ID (例如 "Cam8" -> 8)
#         cam_str = parts[0]
#         camid = int(cam_str[3:])
        
#         # 解析 PID (例如 parts[2] 是 "1")
#         pid = int(parts[2])
        
#         # 解析 View ID
#         # 根据 CARGO 代码: camid <= 5 为 Aerial, 否则为 Ground
#         viewid_str = 'Aerial' if camid <= 5 else 'Ground'
        
#         # 调整 camid (0-indexed)
#         camid_idx = camid - 1
        
#         print(f"🔍 Parsed Info - PID: {pid}, CamID: {camid} ({viewid_str})")
#         return pid, camid_idx, viewid_str

#     def preprocess_image(self, img_path):
#         raw_img = cv2.imread(img_path)
#         if raw_img is None:
#             raise ValueError(f"Could not load image: {img_path}")
#         raw_img_resized = cv2.resize(raw_img, (self.input_size[1], self.input_size[0]))

#         pil_img = Image.open(img_path).convert('RGB')
        
#         # T.ToTensor() 会把范围变成 [0, 1]
#         img_tensor = self.transform(pil_img) 
        
#         # 🟢【核心修复】检查配置中的均值，决定是否还原到 0-255
#         # FastReID 默认均值通常是 [123.675, 116.280, 103.530]
#         if self.cfg.MODEL.PIXEL_MEAN[0] > 1:
#             img_tensor = img_tensor * 255.0
            
#         img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
#         return img_tensor, raw_img_resized

#     def run(self, img_path, output_dir):
#         os.makedirs(output_dir, exist_ok=True)
#         img_name = os.path.basename(img_path).split('.')[0]

#         # 1. 解析信息
#         pid, camid, viewid_str = self.parse_filename(img_path)

#         # 2. 准备数据
#         img_tensor, raw_img = self.preprocess_image(img_path)

#         # 3. 模型推理
#         self.captured_attn = None 
#         with torch.no_grad():
#             # 构造完整的输入字典，所有 tensor 显式指定 device=self.device (CPU)
#             inputs = {
#                 "images": img_tensor,
#                 "targets": torch.tensor([pid], device=self.device),
#                 "camids": torch.tensor([camid], device=self.device),
#                 "viewids": [viewid_str] 
#             }
            
#             # 运行模型
#             if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
#                  self.model(inputs)
#             else:
#                  self.model(img_tensor)

#         if self.captured_attn is None:
#             print("❌ Failed to capture attention map.")
#             return

#         # 4. 解析 Attention Map
#         attn_map = self.captured_attn.squeeze(0) 
#         num_parts, num_patches = attn_map.shape
        
#         H, W = self.input_size
#         patch_w = W // 16 
#         patch_h = num_patches // patch_w 
        
#         print(f"Detected: {num_parts} parts. Grid: ({patch_h}, {patch_w})")

#         # 5. 可视化
#         self.visualize_parts(attn_map, raw_img, patch_h, patch_w, output_dir, img_name)

#     def visualize_parts(self, attn_map, raw_img, grid_h, grid_w, output_dir, img_name):
#         num_parts = attn_map.shape[0]
#         H, W, _ = raw_img.shape
#         heatmaps = []

#         # 生成热力图
#         for i in range(num_parts):
#             att = attn_map[i]
#             # 简单的 Min-Max 归一化用于显示
#             att = (att - att.min()) / (att.max() - att.min() + 1e-8)
#             att_grid = att.reshape(grid_h, grid_w).numpy()
#             att_resized = cv2.resize(att_grid, (W, H), interpolation=cv2.INTER_CUBIC)
#             heatmaps.append(att_resized)

#         # 1. 独立显示
#         for i in range(num_parts):
#             heatmap = heatmaps[i]
#             color = PART_COLORS[i % len(PART_COLORS)]
#             colored_map = np.zeros_like(raw_img)
#             for c in range(3): 
#                 colored_map[:, :, c] = heatmap * color[c]
            
#             overlay = cv2.addWeighted(raw_img, 0.6, colored_map.astype(np.uint8), 0.8, 0)
#             cv2.imwrite(os.path.join(output_dir, f"{img_name}_part_{i}.jpg"), overlay)

#         # 2. 综合显示 (Argmax)
#         stack = np.array(heatmaps) 
#         part_indices = np.argmax(stack, axis=0)
#         max_vals = np.max(stack, axis=0)

#         mask = np.zeros_like(raw_img)
#         for r in range(H):
#             for c in range(W):
#                 if max_vals[r, c] > 0.1: # 阈值过滤背景
#                     pid_idx = part_indices[r, c]
#                     mask[r, c] = PART_COLORS[pid_idx % len(PART_COLORS)]

#         composite_result = cv2.addWeighted(raw_img, 0.7, mask, 0.6, 0)
#         cv2.imwrite(os.path.join(output_dir, f"{img_name}_composite.jpg"), composite_result)
#         print(f"✅ Saved to {output_dir}")

# def setup_cfg(args):
#     cfg = get_cfg()
#     cfg.merge_from_file(args.config_file)
#     cfg.merge_from_list(args.opts)
#     cfg.freeze()
#     return cfg
# #python3 demo/Attn_Map_Visualize_cpu.py --config-file ./demo/demo.yml --input demo/input/Cam8_day_1_50970.jpg --output ./demo/Attn_Map_Output MODEL.WEIGHTS 
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config-file", required=True)
#     parser.add_argument("--input", required=True)
#     parser.add_argument("--output", default="vis_output")
#     parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
#     args = parser.parse_args()

#     cfg = setup_cfg(args)
#     setup_logger(name="fastreid")

#     vis = AttentionVisualizer(cfg)
#     vis.run(args.input, args.output)