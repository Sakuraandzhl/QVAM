# encoding: utf-8
import argparse
import logging
import sys
import os
import numpy as np
import torch
import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random
import time # 引入 time 用于重置种子
from collections import defaultdict

# 添加项目路径
sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo

setup_logger(name="fastreid")
logger = logging.getLogger('fastreid.visualize_result')

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Strict Balanced t-SNE (Paper Style)")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument('--parallel', action='store_true', help='if use multiprocess')
    parser.add_argument("--output", default="./demo/tsne_output", help="output dir")
    parser.add_argument("--file_name", default="tSNE", help="output file name")
    
    # 严格采样参数
    parser.add_argument("--target-ids", type=int, default=10, help="Number of IDs to select")
    parser.add_argument("--samples-per-view", type=int, default=10, help="Strict samples per view per ID")
    
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
    
    # 🟢 [修改1] 关键步骤：重置随机种子
    # 因为 setup_cfg 可能会固定种子，导致 random.sample 每次结果一样
    # 这里用当前时间戳重置种子，保证每次运行选出不同的人
    random.seed('2309482') #230492342  2309482
    
    # 1. 构建 Loader (num_workers=0)
    dataset_name = cfg.DATASETS.TESTS[0]
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=dataset_name, num_workers=0)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting features (Collecting pool)...")
    
    data_pool = defaultdict(lambda: {'Aerial': [], 'Ground': []})
    max_batches = 50
    
    for i, (feat, pid, camid, viewid) in enumerate(tqdm.tqdm(demo.run_on_loader(test_loader), total=max_batches)):
        if i >= max_batches: break
        feat = feat.cpu().numpy()
        for j in range(len(pid)):
            p = int(pid[j])
            v = viewid[j] 
            f = feat[j]
            data_pool[p][v].append(f)

    # 2. 严格筛选 ID
    required_cnt = args.samples_per_view
    valid_pids = []
    for pid, views in data_pool.items():
        if len(views['Aerial']) >= required_cnt and len(views['Ground']) >= required_cnt:
            valid_pids.append(pid)
            
    print(f"IDs meeting criteria: {len(valid_pids)}")
    
    if len(valid_pids) < args.target_ids:
        print(f"⚠️ Warning: Only found {len(valid_pids)} IDs, less than target {args.target_ids}.")
        selected_pids = valid_pids
    else:
        # 🟢 [修改2] 真正的随机采样
        # 每次运行都会从 valid_pids 中随机挑选出 target_ids 个不同的身份
        # selected_pids = random.sample(valid_pids, args.target_ids)
        selected_pids = valid_pids[:args.target_ids]
    print(f"Selected IDs (Randomly Sampled): {selected_pids}")

    # 3. 构建最终数据集
    final_feats = []
    final_pids = []
    final_views = []
    
    for pid in selected_pids:
        # 这里的 random.sample 也会因为上面的 random.seed 重置而变得随机
        samples_a = random.sample(data_pool[pid]['Aerial'], required_cnt)
        for f in samples_a:
            final_feats.append(f)
            final_pids.append(pid)
            final_views.append('Aerial')
            
        samples_g = random.sample(data_pool[pid]['Ground'], required_cnt)
        for f in samples_g:
            final_feats.append(f)
            final_pids.append(pid)
            final_views.append('Ground')

    final_feats = np.array(final_feats)
    final_pids = np.array(final_pids)
    final_views = np.array(final_views)

    # 4. t-SNE
    logger.info("Running t-SNE...")
    # 注意：t-SNE 内部的 random_state 依然固定为 42，
    # 这意味着“对于同一组数据，生成的形状是确定的”。
    # 但因为我们上面的 input data (final_feats) 每次都不同（人不同），
    # 所以最终生成的图每次都会大变样。
    tsne = TSNE(n_components=2, learning_rate='auto', metric='cosine', perplexity=30, init='pca', random_state=42, n_jobs=-1)
    tsne_results = tsne.fit_transform(final_feats)

    # 5. 绘图 (Paper Style with Black Border)
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    
    unique_pids_sorted = sorted(list(set(final_pids)))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_pids_sorted)))
    pid2color = {pid: colors[i] for i, pid in enumerate(unique_pids_sorted)}
    
    for pid in unique_pids_sorted:
        mask_id = (final_pids == pid)
        mask_a = mask_id & (final_views == 'Aerial')
        ax.scatter(tsne_results[mask_a, 0], tsne_results[mask_a, 1], 
                    c=[pid2color[pid]], marker='o', s=60, alpha=0.9, 
                    edgecolors='white', linewidth=0.5)
        mask_g = mask_id & (final_views == 'Ground')
        ax.scatter(tsne_results[mask_g, 0], tsne_results[mask_g, 1], 
                    c=[pid2color[pid]], marker='P', s=80, alpha=0.9, 
                    edgecolors='white', linewidth=0.5)

    # Legend with Frame
    # from matplotlib.lines import Line2D
    # legend_elements = [
    #     Line2D([0], [0], marker='o', color='w', label='Aerial', 
    #            markerfacecolor='#AAAAAA', markersize=10),
    #     Line2D([0], [0], marker='P', color='w', label='Ground', 
    #            markerfacecolor='#AAAAAA', markersize=12)
    # ]
    # ax.legend(handles=legend_elements, loc='upper right', fontsize=12, 
    #           frameon=True, edgecolor='black', fancybox=False, framealpha=1.0, borderpad=0.8)
    
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Aerial', 
               markerfacecolor='#AAAAAA', markersize=9), # 点稍微改小一点点
        Line2D([0], [0], marker='P', color='w', label='Ground', 
               markerfacecolor='#AAAAAA', markersize=11)
    ]
    
    ax.legend(handles=legend_elements, 
              loc='upper right', 
              fontsize=20, 
              
              # 🟢 修改部分：图例框样式
              frameon=True, 
              edgecolor='#CCCCCC', # 浅灰色边框 (Light Grey)
              facecolor='white',   # 白色背景
              framealpha=0.9,      # 稍微有一点点透明度
              borderpad=0.6        # 内部留白稍微小一点
    )

    # Remove ticks but keep border (spines)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set black border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.5)

    plt.tight_layout()

    os.makedirs(args.output, exist_ok=True)
    filepath = os.path.join(args.output, args.file_name + ".pdf")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"✅ Saved to {filepath}")


'''
tSNE for vit  要注释掉baseline_multiview中的memory_bank

CUDA_VISIBLE_DEVICES=1 \
python3 demo/my_tSNE.py \
--config-file demo/demo.yml \
--output demo/tsne \
--target-ids 10 \
--file_name vit_30_resize \
--samples-per-view 10 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS /mnt/sda/sakura/Projects/纯净版ViT/logs/CARGO/VIT_base/model_final.pth
'''

'''
tSNE for pavd

CUDA_VISIBLE_DEVICES=0 \
python3 demo/my_tSNE.py \
--config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml \
--output demo/tsne \
--target-ids 10 \
--file_name pavd_20_resize \
--samples-per-view 10 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs_6/34597178_7.1/True/1.0_0.01_1.0/model_best.pth
'''

"""
测试集最多只有50个batch,所以无论种子是多少都能把每个id的测试集图像全部读完,每次改种子仅仅改的是随机挑选的20张图像,而id的选择是基于字典读取,所以每次选择的id相同但是选择某个id的图像不同
"""