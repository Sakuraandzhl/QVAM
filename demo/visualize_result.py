# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
import argparse
import logging
import sys
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
import torch
import tqdm
from torch.backends import cudnn

# 解决 OpenMP 冲突 (防止 Segfault)
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

sys.path.append('.')
from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader, build_reid_train_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer
import plotly.express as px
import plotly.graph_objs as go

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')

# tsne画图时随机选择初始点迭代，设置seed可让byid和byview图位置一样
seed = 42
np.random.seed(seed)

def visualize_multiview_features(
    feats,
    labels,
    views,
    mode='2d',            # '2d' | '3d' | 'compare'
    method='tsne',        # 'tsne' | 'pca'
    save_dir='outputs/mv_vis',
    sample_num=400,
    show=False
):
    os.makedirs(save_dir, exist_ok=True)
    print(f"🔹 Mode = {mode.upper()}, Method = {method.upper()}")

    # === Step 1️⃣ 检查维度是否对齐 ===
    total_dim = feats.shape[1]
    if total_dim % 2 != 0:
        print(f"⚠️ Warning: Feature dimension {total_dim} is ODD! Truncating last dimension.")
        feats = feats[:, :-1]
        total_dim -= 1
    
    # === Step 2️⃣ 拆分视角特征 ===
    C = total_dim // 2
    inv_feat = feats[:, :C]
    global_feat = feats[:, C:]

    # === Step 3️⃣ 基础统计 ===
    def stat_info(name, f):
        f = f.detach().cpu()
        mean_val = f.mean().item()
        std_val = f.std().item()
        abs_mean = f.abs().mean().item()
        print(f"\n==== {name} ====")
        print(f"Mean: {mean_val:.4f}, Std: {std_val:.4f}, AbsMean: {abs_mean:.4f}")

    stat_info("invariant feature", inv_feat)
    stat_info("global feature", global_feat)

    # === Step 4️⃣ 整理数据 ===
    all_feats, all_labels, all_tags, all_div = [], [], [], []
    
    for i in range(feats.shape[0]):
        invf = inv_feat[i].detach().cpu().numpy()
        glof = global_feat[i].detach().cpu().numpy()
        pid = labels[i].item()
        
        # 处理 View 标签
        v = views[i].decode() if isinstance(views[i], bytes) else views[i]
        if v.lower().startswith('a'):
            v_ori = 'Aerial'
        else:
            v_ori = 'Ground'
            
        # 添加 Invariant 特征
        all_feats.append(invf)
        all_labels.append(pid)
        all_tags.append(v_ori)
        all_div.append('invariant')
        
        # 添加 Global 特征
        all_feats.append(glof)
        all_labels.append(pid)
        all_tags.append(v_ori)
        all_div.append('global')

    # 转换为 Numpy 数组并确保连续性
    all_feats = np.array(all_feats, dtype=np.float64)
    all_feats = np.ascontiguousarray(all_feats)
    
    # 将标签列表也转换为 Numpy 数组，方便索引
    all_labels = np.array(all_labels)
    all_tags = np.array(all_tags)
    all_div = np.array(all_div)

    print(f"📊 Total valid features: {all_feats.shape}")
    
    if not np.isfinite(all_feats).all():
        print("❌ Error: Features contain NaN or Infinity. Cleaning data...")
        all_feats = np.nan_to_num(all_feats)

    # === Step 5️⃣ 降维 ===
    n_samples = all_feats.shape[0]
    safe_perplexity = min(30, max(5, n_samples // 4))
    
    if mode.lower() in ['2d', 'compare']:
        reducer = TSNE(
            n_components=2, 
            init='random',  # 推荐 random 防止 crash
            perplexity=safe_perplexity, 
            random_state=seed,
            learning_rate='auto'
        ) if method.lower() == 'tsne' else PCA(n_components=2)
    else:
        reducer = PCA(n_components=3)

    print(f"🔹 Running dimensionality reduction (Perplexity={safe_perplexity})...")
    try:
        reduced = reducer.fit_transform(all_feats)
    except Exception as e:
        print(f"⚠️ Reduction failed ({str(e)}), fallback to PCA")
        reduced = PCA(n_components=2).fit_transform(all_feats)

    # === Step 6️⃣ 绘图函数 (修复版) ===

    def plot_2d_by_view(reduced, suffix='2d'):
        plt.figure(figsize=(10, 9))
        
        # 颜色映射: 区分 Aerial / Ground
        colors_map = {
            'Aerial': 'tab:blue',
            'Ground': 'tab:red',
        }
        
        # 形状映射: 区分 Invariant / Global
        markers_map = {
            'invariant': 'o',  # 圆点
            'global': 'x',     # 叉号
        }
        
        # 我们需要画 4 种组合: (Aerial, Inv), (Aerial, Glob), (Ground, Inv), (Ground, Glob)
        # 这样图例才能正确显示
        combinations = [
            ('Aerial', 'invariant'),
            ('Aerial', 'global'),
            ('Ground', 'invariant'),
            ('Ground', 'global')
        ]
        
        for view_tag, div_tag in combinations:
            # 找到同时满足 View 和 Type 的索引
            mask = (all_tags == view_tag) & (all_div == div_tag)
            if not np.any(mask):
                continue
                
            plt.scatter(
                reduced[mask, 0], reduced[mask, 1],
                s=30, alpha=0.7,
                c=colors_map[view_tag],
                marker=markers_map[div_tag],
                label=f"{view_tag}-{div_tag}"
            )

        plt.legend(loc='best', fontsize=10)
        plt.title(f"Multiview Feature Distribution ({method.upper()})\nColor=View, Marker=FeatType")
        plt.tight_layout()
        path = os.path.join(save_dir, f"multiview_{method}_{suffix}_byView.png")
        plt.savefig(path, dpi=300)
        print(f"✅ Saved {path}")

    def plot_2d_by_id(reduced, suffix='2d'):
        plt.figure(figsize=(12, 10))
        
        unique_ids = np.unique(all_labels)
        # 为每个 ID 生成不同的颜色
        cmap = plt.cm.get_cmap('nipy_spectral', len(unique_ids))
        
        markers_map = {
            'invariant': 'o',
            'global': '^',  # 用三角形区分，因为 x 有时候看不清颜色
        }
        
        # 只画前 10 个 ID 避免混乱，或者全部画
        # 这里画全部，但要注意点可能很密
        for i, uid in enumerate(unique_ids):
            # 获取该 ID 的颜色
            color = cmap(i)
            
            # 分别画 invariant 和 global
            for div_tag in ['invariant', 'global']:
                mask = (all_labels == uid) & (all_div == div_tag)
                if not np.any(mask):
                    continue
                
                plt.scatter(
                    reduced[mask, 0], reduced[mask, 1],
                    s=40, alpha=0.8,
                    color=color,
                    marker=markers_map[div_tag],
                    # 不加 label 防止图例爆炸
                )
        
        # 手动添加形状图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='k', label='Invariant'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='k', label='Global')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        plt.title(f"Feature Distribution by ID ({method.upper()})\nColor=ID, Marker=FeatType")
        plt.tight_layout()
        path = os.path.join(save_dir, f"multiview_{method}_{suffix}_byID.png")
        plt.savefig(path, dpi=300)
        print(f"✅ Saved {path}")

    # 3D 绘图保持原样，Plotly 会自动处理 Pandas/Dict 结构
    def plot_3d_interactive(reduced, suffix='3d'):
        # 构造 DataFrame
        import pandas as pd
        df = pd.DataFrame({
            'x': reduced[:, 0],
            'y': reduced[:, 1],
            'z': reduced[:, 2],
            'View': all_tags,
            'FeatType': all_div,
            'ID': all_labels.astype(str)
        })
        
        # By View
        fig_view = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='View', symbol='FeatType',
            hover_data=['ID', 'FeatType'],
            title=f"3D Distribution ({method.upper()} by View)",
            color_discrete_map={'Aerial': 'blue', 'Ground': 'green'}
        )
        fig_view.update_traces(marker=dict(size=4, opacity=0.7))
        fig_view.write_html(os.path.join(save_dir, f"multiview_{method}_{suffix}_byView.html"))
        
        # By ID
        fig_id = px.scatter_3d(
            df, x='x', y='y', z='z',
            color='ID', symbol='FeatType',
            hover_data=['View'],
            title=f"3D Distribution ({method.upper()} by ID)"
        )
        fig_id.update_traces(marker=dict(size=4, opacity=0.7))
        fig_id.write_html(os.path.join(save_dir, f"multiview_{method}_{suffix}_byID.html"))
        
        print(f"✅ Saved 3D HTML files")

    # === Step 7️⃣ 执行 ===
    if mode == '2d':
        plot_2d_by_view(reduced)
    elif mode == '3d':
        reduced3d = PCA(n_components=3).fit_transform(all_feats)
        plot_3d_interactive(reduced3d)
    elif mode == 'compare':
        # 2D 部分
        plot_2d_by_view(reduced, suffix='2d')
        plot_2d_by_id(reduced, suffix='2d')
        # 3D 部分
        reduced3d = PCA(n_components=3).fit_transform(all_feats)
        plot_3d_interactive(reduced3d, suffix='3d')
    else:
        raise ValueError("mode should be '2d', '3d', or 'compare'")


# ... (setup_cfg, get_parser 等代码保持不变) ...
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--config-file",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='if use multiprocess for feature extraction.'
    )
    parser.add_argument(
        "--dataset-name",
        help="a test dataset name for visualizing ranking list."
    )
    parser.add_argument(
        "--output",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",

    )
    parser.add_argument(
        "--vis-label",
        action='store_true',
        help="if visualize label of query instance"
    )
    parser.add_argument(
        "--num-vis",
        default=100,
        help="number of query images to be visualized",
    )
    parser.add_argument(
        "--rank-sort",
        default="ascending",
        help="rank order of visualization images by AP metric",
    )
    parser.add_argument(
        "--label-sort",
        default="ascending",
        help="label order of visualization images by cosine similarity metric",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,             # 参数类型为整数
        default=1,            # 默认使用 1 张 GPU
        help="number of gpus *per machine*"
    )
    parser.add_argument(
        "--max-rank",
        default=10,
        help="maximum number of rank list to be visualized",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )

    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    dataset_name = cfg.DATASETS.TESTS[0]
    test_loader = build_reid_train_loader(cfg, dataset_name=dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []
    viewids = []  

    # 稍微增加采样数以确保 t-SNE 效果
    max_batches = 10 
    
    for i, (feat, pid, camid, viewid) in enumerate(tqdm.tqdm(demo.run_on_loader(test_loader))):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
        viewids.extend(viewid)
        if i >= max_batches:
            break
            
    feats = torch.cat(feats, dim=0)
    pids = np.asarray(pids)
    camids = np.asarray(camids)

    # ==============================
    # ✅ 多视角特征可视化分析
    # ==============================
    # 如果提取太多，随机采样 400 个点（即 200 张图片）
    if len(feats) > 200:
        idx = np.random.choice(len(feats), 200, replace=False)
        feats = feats[idx]
        pids = pids[idx]
        # 注意 viewids 是 list，不能直接用 numpy 索引，需要转换
        viewids = np.array(viewids)[idx]

    logger.info(f"Visualizing {len(feats)} images...")

    visualize_multiview_features(
        feats,                
        torch.tensor(pids),   
        viewids,              
        mode='compare',
        method='tsne',
        save_dir='outputs/vis_results',
        show=False
    )