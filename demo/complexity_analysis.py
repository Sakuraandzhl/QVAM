# encoding: utf-8
import argparse
import sys
import os
import time
import torch
import torch.nn as nn
from thop import profile, clever_format

# 添加项目根目录到路径
sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.modeling.meta_arch import build_model
from fastreid.utils.logger import setup_logger
def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # 强制在 GPU 上运行以测量 FPS
    cfg.MODEL.DEVICE = "cuda"
    cfg.freeze()
    return cfg

class ModelWrapper(nn.Module):
    """
    包装器类：适配 thop 的输入格式，并构造 Baseline_multiview 需要的字典。
    """
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # 预先构造好 dummy 的辅助信息，避免在 forward 里重复创建
        # 假设 batch size = 1
        self.dummy_targets = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dummy_camids = torch.zeros(1, dtype=torch.long, device=self.device)
        self.dummy_viewids = ["Aerial"] # 列表形式

    def forward(self, x):
        # x: [1, 3, H, W]
        # 注意：thop 传入的 x 通常是未归一化的随机数。
        # Baseline_multiview 内部会做 sub mean div std。
        # 为了保证流程顺畅，我们构造字典。
        
        if self.cfg.MODEL.META_ARCHITECTURE == 'Baseline_multiview':
            inputs = {
                "images": x,
                "targets": self.dummy_targets,
                "camids": self.dummy_camids,
                "viewids": self.dummy_viewids
            }
            # eval 模式下，Baseline_multiview 返回特征 Tensor
            return self.model(inputs)
        else:
            # 对于纯 ViT 等标准模型
            return self.model(x)

def main(args):
    cfg = setup_cfg(args)
    setup_logger(name="fastreid")
    
    # 1. 构建模型
    print(f"Building model: {cfg.MODEL.META_ARCHITECTURE}")
    model = build_model(cfg)
    model.eval()
    device = torch.device("cuda")
    model.to(device)

    # 2. 包装模型
    wrapped_model = ModelWrapper(model, cfg)
    wrapped_model.eval()

    # 3. 准备 Dummy Input
    # 按照 config 中的测试尺寸
    h, w = cfg.INPUT.SIZE_TEST
    # 创建一个随机输入，模拟 [0, 255] 的图像数据（虽然这里是 float，但在 preprocess 里会被处理）
    # 注意：FastReID 的 preprocess 期望输入是 float tensor
    input_tensor = torch.randn(1, 3, h, w).to(device)

    print(f"Input size: (1, 3, {h}, {w})")

    # ---------------------------------------------------------
    # 计算 Params 和 GFLOPs (使用 thop)
    # ---------------------------------------------------------
    print("Running thop profile...")
    try:
        # thop.profile 会自动递归 tracing 所有的子模块
        # inputs 必须是 tuple
        macs, params = profile(wrapped_model, inputs=(input_tensor, ), verbose=False)
        
        # 格式化输出
        # MACs -> GFLOPs: 通常 1 MAC = 2 FLOPs，但在论文中常直接用 MACs 数值代表 GFLOPs
        # 这里我们遵循 thop 的输出习惯，macs 是乘加次数。
        # 如果你想严格对应 GFLOPs，可以 * 2，但对比时保持一致即可。
        # 现在的顶会论文大多直接汇报 thop 算出来的 macs 作为 GFLOPs。
        macs_str, params_str = clever_format([macs, params], "%.3f")
        
        print("-" * 30)
        print(f"Parameters: {params_str}")
        print(f"GFLOPs (MACs): {macs_str}")
        print("-" * 30)
        
        # 保存数值用于 LaTeX
        gflops_val = macs / 1e9
        params_val = params / 1e6
        
    except Exception as e:
        print(f"❌ Error calculating FLOPs: {e}")
        import traceback
        traceback.print_exc()
        gflops_val = 0.0
        params_val = 0.0

    # ---------------------------------------------------------
    # 计算 FPS / Latency (推理速度)
    # ---------------------------------------------------------
    print("Measuring Inference Speed...")
    
    # 预热
    print("Warmup...")
    with torch.no_grad():
        for _ in range(50):
            wrapped_model(input_tensor)
    
    # 计时
    iterations = 1000  # 增加循环次数以获得更稳定的结果
    torch.cuda.synchronize()
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(iterations):
            wrapped_model(input_tensor)
            
    torch.cuda.synchronize()
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / iterations) * 1000 # 毫秒
    fps = iterations / total_time
    
    print("-" * 30)
    print(f"Batch Size: 1")
    print(f"Iterations: {iterations}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Latency: {avg_latency:.2f} ms/img")
    print(f"FPS: {fps:.2f} img/s")
    print("-" * 30)

    # ---------------------------------------------------------
    # 输出 LaTeX 表格行
    # ---------------------------------------------------------
    print("\nCopy this row to your LaTeX table:")
    print(f"QVAM (Ours) & ViT-B/16 & {params_val:.1f} & {gflops_val:.1f} & {avg_latency:.1f} \\\\")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Params, FLOPs, and FPS")
    parser.add_argument("--config-file", required=True, help="path to config file")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main(args)

'''
QVAM
CUDA_VISIBLE_DEVICES=0 \
python3 demo/complexity_analysis.py \
--config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml

ViT
CUDA_VISIBLE_DEVICES=0 \
python3 demo/complexity_analysis.py \
--config-file 之前的消融/CARGO消融/34597178/ViT/config.yml
'''