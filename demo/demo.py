# encoding: utf-8
"""
用来提取任意类型图像，只关注特征不计算相似度可视化等等
但是extract_feature.py和visualize_results.py需要ReID标准数据集且要有pid和camid标签并且可以进行可视化
主要用于实际进行检索，比如提取两百张图像特征进行匹配
作用:
1.配置参数(通过命令行)
2.通过input output获取输入图像并提取特征放入output
核心目标:提取单个 / 批量图像的特征，保存为 .npy 文件供后续使用
使用场景:实际应用中获取图像特征（如 “给 100 张行人图提特征，用于后续检索”）
用户关注:每张图对应的特征向量文件
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import argparse  # 处理命令行参数
import glob       # 批量匹配文件路径（如 "dir/*.jpg"）
import os         # 路径操作
import sys        # 系统路径配置
import torch
import torch.nn.functional as F  # PyTorch 函数库（用于特征归一化）
import cv2        # OpenCV，读取/处理图像
import numpy as np # 数值计算（特征保存为 numpy 数组）
import tqdm       # 显示训练/处理进度条
from torch.backends import cudnn  # PyTorch GPU 加速配置

sys.path.append('.')  # 将当前目录加入系统路径，确保能导入本地模块

# FastReID 框架核心工具
from fastreid.config import get_cfg  # 加载 FastReID 配置
from fastreid.utils.logger import setup_logger  # 初始化日志系统
from fastreid.utils.file_io import PathManager  # FastReID 路径管理工具（兼容多环境）

from predictor import FeatureExtractionDemo  # 自定义特征提取类（核心依赖，来自之前分析的 predictor.py）

# import some modules added in project like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True#后续会启用，让 CuDNN 自动选择最优卷积算法，提升 GPU 推理速度
setup_logger(name="fastreid")


def setup_cfg(args):
    # 1. 初始化 FastReID 配置节点（默认包含基础参数）
    cfg = get_cfg()#cfg 是 FastReID 的「配置节点」，包含模型、输入、数据集等所有参数，后续 FeatureExtractionDemo 需用它初始化模型
    # 2. 从配置文件加载参数（如模型结构、预训练权重路径、输入尺寸等）
    cfg.merge_from_file(args.config_file)
    # 3. 用命令行参数覆盖配置文件（优先级：命令行 > 配置文件，方便临时调整）
    cfg.merge_from_list(args.opts)
    # 4. 冻结配置：防止后续代码误修改参数（确保配置稳定）
    cfg.freeze()
    return cfg


def get_parser():
    # 初始化参数解析器，描述工具功能
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")

    # 必选参数：模型配置文件路径（如 logs/cargo/config.yaml）
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",  # 参数说明
    )

    # 可选参数：是否启用多进程特征提取（加速批量处理）
    parser.add_argument(
        "--parallel",
        action='store_true',  # 无需传值，加此参数即表示 "启用"
        help='If use multiprocess for feature extraction.'
    )

    # 必选参数：输入图像路径（支持单图、多图、目录/glob 模式）
    parser.add_argument(
        "--input",
        nargs="+",  # 支持传入多个值（如 "img1.jpg img2.jpg" 或 "dir/*.jpg"）
        help="A list of space separated input images; "
             "or a single glob pattern such as 'directory/*.jpg'",
    )

    # 可选参数：特征保存目录（默认 "demo_output"）
    parser.add_argument(
        "--output",
        default='demo_output',
        help='path to save features'
    )

    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER  # 接收所有剩余的参数
    )
    return parser


def postprocess(features):
    # features = F.normalize(features)
    features = features.cpu().data.numpy()
    return features


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)
    PathManager.mkdirs(args.output)  # FastReID 工具，兼容本地/分布式文件系统

    if args.input:
        if PathManager.isdir(args.input[0]):
            args.input = glob.glob(os.path.expanduser(args.input[0]))
            assert args.input, "The input path(s) was not found"
        for path in tqdm.tqdm(args.input):
            img_name = os.path.basename(path)  # 例如：返回 "Cam1_xxx_yyy.jpg"
            pid = torch.tensor([int(img_name.split('_')[2])])  # 从文件名分割出第 3 部分（索引 2）作为 pid
            camid = torch.tensor([int(img_name.split('_')[0][3:])])  # 从文件名分割出第 1 部分（CamX），截取 X 作为 camid
            viewid = 'Aerial' if camid <= 5 else 'Ground'
            img = cv2.imread(path)
            feat = demo.run_on_image(img, camid, viewid)
            feat = postprocess(feat)
            save_path = os.path.join(args.output, os.path.basename(path).split('.')[0] + '.npy')
            np.save(save_path, feat)
