# encoding: utf-8
import argparse
import sys
import os
import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

# 添加项目根目录到路径
sys.path.append('.') 

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.utils.logger import setup_logger

# 定义不同 Part 的颜色 (BGR 格式)
PART_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 255, 0),  # Cyan
    (255, 0, 255),  # Magenta
    (0, 165, 255),  # Orange
    (128, 0, 128),  # Purple
]

class AttentionVisualizer:
    def __init__(self, cfg):
        self.cfg = cfg
        # self.device = torch.device(cfg.MODEL.DEVICE)
        # cfg.defrost()
        # cfg.MODEL.BACKBONE.PRETRAIN = False

        self.device = torch.device("cpu")
        cfg.defrost()
        cfg.MODEL.DEVICE = "cpu"       # 关键：告诉 fastreid 使用 CPU
        cfg.MODEL.BACKBONE.PRETRAIN = False
        self.cfg = cfg
        # 强制关闭预训练加载，避免覆盖我们的权重
        
        
        # 1. 构建模型
        print("Building model...")
        self.model = build_model(cfg)
        self.model.eval()
        self.model.to(self.device)

        # 2. 加载权重
        print(f"Loading checkpoint from {cfg.MODEL.WEIGHTS}...")
        Checkpointer(self.model).load(cfg.MODEL.WEIGHTS)

        # 3. 注册 Hook
        self.captured_attn = None 
        self._register_hook()

        # 4. 图像预处理 (核心修复点)
        # ❌ 移除了 T.Normalize，因为模型内部 Baseline_multiview.preprocess_image 会做归一化
        # 如果这里再做一次，数值就错了，导致 Attention Map 失效
        self.transform = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TEST),
            T.ToTensor(), 
        ])
        
        self.input_size = cfg.INPUT.SIZE_TEST 

    def _register_hook(self):
        def hook_fn(module, input, output):
            # PAD.forward 返回 (part_feats, attn_weights)
            prompts, attn_weights = output
            self.captured_attn = attn_weights.detach().cpu()

        try:
            if hasattr(self.model, 'module'):
                target_module = self.model.module.pavd.vad
            else:
                target_module = self.model.pavd.vad
            
            target_module.register_forward_hook(hook_fn)
            print("✅ Successfully hooked into PGVD.PAD module.")
        except AttributeError as e:
            print(f"❌ Error hooking module: {e}")
            sys.exit(1)

    def parse_filename(self, img_path):
        """
        根据 CARGO 数据集格式解析文件名
        Example: Cam8_day_1_50970.jpg
        """
        filename = os.path.basename(img_path)
        parts = filename.split('_')
        
        # 解析 Cam ID (例如 "Cam8" -> 8)
        cam_str = parts[0]
        camid = int(cam_str[3:])
        
        # 解析 PID (例如 parts[2] 是 "1")
        pid = int(parts[2])
        
        # 解析 View ID
        # 根据 CARGO 代码: camid <= 5 为 Aerial, 否则为 Ground
        viewid_str = 'Aerial' if camid <= 5 else 'Ground'
        
        # 调整 camid (0-indexed)
        camid_idx = camid - 1
        
        print(f"🔍 Parsed Info - PID: {pid}, CamID: {camid} ({viewid_str})")
        return pid, camid_idx, viewid_str

    def preprocess_image(self, img_path):
        raw_img = cv2.imread(img_path)
        if raw_img is None:
            raise ValueError(f"Could not load image: {img_path}")
        raw_img_resized = cv2.resize(raw_img, (self.input_size[1], self.input_size[0]))

        pil_img = Image.open(img_path).convert('RGB')
        
        # T.ToTensor() 会把范围变成 [0, 1]
        img_tensor = self.transform(pil_img) 
        
        # 🟢【核心修复】检查配置中的均值，决定是否还原到 0-255
        # FastReID 默认均值通常是 [123.675, 116.280, 103.530]
        if self.cfg.MODEL.PIXEL_MEAN[0] > 1:
            img_tensor = img_tensor * 255.0
            
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, raw_img_resized

    def run(self, img_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.basename(img_path).split('.')[0]

        # 1. 解析信息
        pid, camid, viewid_str = self.parse_filename(img_path)

        # 2. 准备数据
        img_tensor, raw_img = self.preprocess_image(img_path)

        # 3. 模型推理
        self.captured_attn = None 
        with torch.no_grad():
            # 构造完整的输入字典
            inputs = {
                "images": img_tensor,
                "targets": torch.tensor([pid], device=self.device),
                "camids": torch.tensor([camid], device=self.device),
                "viewids": [viewid_str] # 传入 list of strings
            }
            
            # 运行模型
            if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
                 self.model(inputs)
            else:
                 self.model(img_tensor)

        if self.captured_attn is None:
            print("❌ Failed to capture attention map.")
            return

        attn_map = self.captured_attn.squeeze(0) 
        
        # 🟢【关键步骤】移除 CLS Token (如果有)
        # 假设 attn_map 是 [Num_Prompts, Num_Patches]
        # 如果 patch 数量无法整除 (例如 129)，通常第一个是 CLS
        total_pixels = self.input_size[0] * self.input_size[1]
        patch_area = 16 * 16
        expected_patches = total_pixels // patch_area
        
        if attn_map.shape[1] == expected_patches + 1:
            print("✂️ Removing CLS token for visualization.")
            attn_map = attn_map[:, 1:]
            
        num_parts, num_patches = attn_map.shape
        H, W = self.input_size
        
        # 计算 Grid 形状 (例如 256x128 -> 16x8)
        grid_w = W // 16
        grid_h = H // 16
        
        print(f"Grid Size: ({grid_h}, {grid_w})")

        # 5. 可视化 (调用新函数)
        self.visualize_patches_blocky(attn_map, raw_img, grid_h, grid_w, output_dir, img_name)
    def visualize_patches_blocky(self, attn_map, raw_img, grid_h, grid_w, output_dir, img_name):
        """
        不进行插值，直接将 16x16 的 patch 染色
        """
        num_parts = attn_map.shape[0]
        raw_H, raw_W, _ = raw_img.shape
        
        # 存储调整大小后的 Mask，用于 Argmax
        resized_masks = []

        # -----------------------------
        # 1. 独立显示 (Blocky Style)
        # -----------------------------
        for i in range(num_parts):
            att = attn_map[i] # Shape: [N_Patches]
            
            # 归一化 (依然需要归一化以便人眼观察，但你可以选择是否 MinMax)
            # 建议：为了看清真实分布，使用 Global 归一化或者局部 MinMax
            att_norm = (att - att.min()) / (att.max() - att.min() + 1e-8)
            
            # Reshape 成网格 [grid_h, grid_w]
            att_grid = att_norm.reshape(grid_h, grid_w).numpy()
            
            # 🟢【核心修改】使用最近邻插值 (NEAREST) 放大到原图尺寸
            # 这会保持方块形状，不会模糊
            att_resized = cv2.resize(att_grid, (raw_W, raw_H), interpolation=cv2.INTER_NEAREST)
            resized_masks.append(att_resized)
            
            # 生成颜色层
            color = PART_COLORS[i % len(PART_COLORS)]
            colored_map = np.zeros_like(raw_img)
            
            # 这种写法会让关注度高的地方颜色深，关注度低的地方透明
            for c in range(3):
                colored_map[:, :, c] = (att_resized * color[c]).astype(np.uint8)
            
            # 叠加: 图像 + 颜色层
            # 只有 att_resized > 0 的地方才会有颜色
            overlay = cv2.addWeighted(raw_img, 0.7, colored_map, 0.8, 0)
            
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_part_{i}_blocky.jpg"), overlay)

        # -----------------------------
        # 2. 综合显示 (Argmax Blocky)
        # -----------------------------
        stack = np.array(resized_masks) # Shape: [K, H, W]
        
        # 找出每个像素点（其实是每个Patch）归属哪个 Prompt
        part_indices = np.argmax(stack, axis=0) 
        max_vals = np.max(stack, axis=0)

        mask_img = np.zeros_like(raw_img)
        
        # 设定阈值，避免把背景噪声也染色
        # 因为我们用了 Min-Max，背景通常是 0，所以阈值设小一点即可
        threshold = 0.1 
        
        for r in range(raw_H):
            for c in range(raw_W):
                if max_vals[r, c] > threshold:
                    pid = part_indices[r, c]
                    mask_img[r, c] = PART_COLORS[pid % len(PART_COLORS)]
        
        # 叠加
        composite_result = cv2.addWeighted(raw_img, 0.6, mask_img, 0.6, 0)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_composite_blocky.jpg"), composite_result)
        
        print(f"✅ Saved blocky visualizations to {output_dir}")
    def visualize_parts(self, attn_map, raw_img, grid_h, grid_w, output_dir, img_name):
        num_parts = attn_map.shape[0]
        H, W, _ = raw_img.shape
        heatmaps = []
        # for i in range(1, num_parts):
        #     print(i, (attn_map[0] - attn_map[i]).abs().mean().item())
        # 生成热力图
        for i in range(num_parts):
            att = attn_map[i]
            # 简单的 Min-Max 归一化用于显示
            att = (att - att.min()) / (att.max() - att.min() + 1e-8)
            # print(att)
            att_grid = att.reshape(grid_h, grid_w).numpy()
            # att = attn_map[i]
            # att_grid = att.reshape(grid_h, grid_w).numpy()
            att_resized = cv2.resize(att_grid, (W, H), interpolation=cv2.INTER_CUBIC)
            heatmaps.append(att_resized)

        # 1. 独立显示
        for i in range(num_parts):
            heatmap = heatmaps[i]
            color = PART_COLORS[i % len(PART_COLORS)]
            colored_map = np.zeros_like(raw_img)
            for c in range(3): 
                colored_map[:, :, c] = heatmap * color[c]
            
            overlay = cv2.addWeighted(raw_img, 0.6, colored_map.astype(np.uint8), 0.8, 0)
            cv2.imwrite(os.path.join(output_dir, f"{img_name}_part_{i}.jpg"), overlay)

        # 2. 综合显示 (Argmax)
        stack = np.array(heatmaps) 
        part_indices = np.argmax(stack, axis=0)
        max_vals = np.max(stack, axis=0)

        mask = np.zeros_like(raw_img)
        for r in range(H):
            for c in range(W):
                if max_vals[r, c] > 0.15: # 阈值过滤背景
                    pid_idx = part_indices[r, c]
                    mask[r, c] = PART_COLORS[pid_idx % len(PART_COLORS)]

        composite_result = cv2.addWeighted(raw_img, 0.7, mask, 0.6, 0)
        cv2.imwrite(os.path.join(output_dir, f"{img_name}_composite.jpg"), composite_result)
        print(f"✅ Saved to {output_dir}")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="vis_output")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()

    cfg = setup_cfg(args)
    setup_logger(name="fastreid")

    vis = AttentionVisualizer(cfg)
    vis.run(args.input, args.output)
#CUDA_VISIBLE_DEVICES=2 python3 demo/Attn_Map_Visualize.py --config-file ./demo/demo.yml --input demo/input/Cam8_day_1_50970.jpg --output demo/Attn_Map_Output MODEL.WEIGHTS 
# python3 demo/Attn_Map_Visualize.py --config-file ./demo/demo.yml --input demo/input/Cam8_day_2_51424.jpg --output demo/Attn_Map_Output MODEL.WEIGHTS 
