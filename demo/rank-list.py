# encoding: utf-8
import argparse
import sys
import torch
import numpy as np
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.evaluation.rank import evaluate_rank
from fastreid.utils.visualizer import Visualizer # 注意：这里用标准的 Visualizer

cudnn.benchmark = True
setup_logger(name="fastreid")

def setup_cfg(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction and Rank-list visualization")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--dataset-name", help="a test dataset name for visualizing ranking list.")
    parser.add_argument("--output", default="./vis_results", help="directory to save visualization results.")
    parser.add_argument("--num-vis", type=int, default=20, help="number of query images to be visualized")
    parser.add_argument("--max-rank", type=int, default=10, help="maximum number of rank list to be visualized")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER)
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = setup_cfg(args)
        
    # 1. 加载数据和模型
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=False)
    
    # 2. 提取特征
    print(">>> Extracting features...")
    feats = []
    pids = []
    camids = []
    for (feat, pid, camid, viewid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)
    
    feats = torch.cat(feats, dim=0)
    pids = np.array(pids)
    camids = np.array(camids)
    
    q_feat = feats[:num_query]
    g_feat = feats[num_query:]
    q_feat = torch.nn.functional.normalize(q_feat, dim=1, p=2)
    g_feat = torch.nn.functional.normalize(g_feat, dim=1, p=2)
    distmat = 1 - torch.mm(q_feat, g_feat.t()).cpu().numpy()
    
    cmc, all_ap, all_inp = evaluate_rank(distmat, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])

    visualizer = Visualizer(test_loader.dataset)
    visualizer.get_model_output(all_ap, distmat, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])
    print(f">>> Visualizing Top-{args.num_vis} worst cases (AP ascending)...")
    visualizer.vis_rank_list(
        output=args.output,
        vis_label=False,
        num_vis=args.num_vis,
        rank_sort="ascending",
        label_sort="ascending",
        max_rank=args.max_rank,
    )
    
    print(f">>> Done! Results saved in {args.output}")

"""
AG-ReIDv2  AG_ReID_v2_A2W  AG_ReID_v2_G2A

CUDA_VISIBLE_DEVICES=2 \
python3 demo/rank-list.py \
--config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
--dataset-name AG_ReID_v2_A2W \
--output worst_case_analysis \
--num-vis 50 \
--max-rank 10 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth
"""