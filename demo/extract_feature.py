# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
"""
核心目标：评估 ReID 模型在测试集上的性能，生成可视化结果（排名列表、ROC 曲线）
使用场景：模型训练后验证效果（如 “我的模型在 DukeMTMC 数据集上表现如何？”）
用户关注：模型性能指标（CMC、AP）、结果可视化图表
"""
import argparse
import logging
import pdb
import sys

import numpy as np
import torch
import tqdm
from torch.backends import cudnn

sys.path.append('.')

from fastreid.evaluation.rank import evaluate_rank
from fastreid.config import get_cfg
from fastreid.utils.logger import setup_logger
from fastreid.data import build_reid_test_loader
from predictor import FeatureExtractionDemo
from fastreid.utils.visualizer import Visualizer

# import some modules added in project
# for example, add partial reid like this below
# sys.path.append("projects/PartialReID")
# from partialreid import *

cudnn.benchmark = True
setup_logger(name="fastreid")

logger = logging.getLogger('fastreid.visualize_result')


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
        type=int,
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
        "--max-rank",
        type=int,
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
    test_loader, num_query = build_reid_test_loader(cfg, dataset_name=args.dataset_name)
    demo = FeatureExtractionDemo(cfg, parallel=args.parallel)

    logger.info("Start extracting image features")
    feats = []
    pids = []
    camids = []

    for (feat, pid, camid, viewid) in tqdm.tqdm(demo.run_on_loader(test_loader), total=len(test_loader)):
        feats.append(feat)
        pids.extend(pid)
        camids.extend(camid)


    pdb.set_trace()

    # 将批次特征拼接成完整特征矩阵（shape: [总样本数, 特征维度]）
    feats = torch.cat(feats, dim=0)
    # 按 num_query 拆分查询集（前 num_query 个）和图库集（剩余）
    q_feat = feats[:num_query]  # 查询集特征（shape: [num_query, 特征维度]）
    g_feat = feats[num_query:]  # 图库集特征（shape: [num_gallery, 特征维度]）
    # 拆分 PID 和 CamID（numpy 数组，用于后续评估）
    q_pids = np.asarray(pids[:num_query])  # 查询集身份ID
    g_pids = np.asarray(pids[num_query:])  # 图库集身份ID
    q_camids = np.asarray(camids[:num_query])  # 查询集相机ID
    g_camids = np.asarray(camids[num_query:])  # 图库集相机ID
    # 计算余弦距离：1 - 余弦相似度（相似度越高，距离越小）
    distmat = 1 - torch.mm(q_feat, g_feat.t())
    distmat = distmat.numpy()  # 转为 numpy 数组（方便后续评估函数处理）

    
    logger.info("Computing APs for all query images ...")
    # 评估排名性能：输入距离矩阵和标签，返回 CMC、AP、inp（平均精度）
    cmc, all_ap, all_inp = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
    logger.info("Finish computing APs for all query images!")

    visualizer = Visualizer(test_loader.dataset)
    visualizer.get_model_output(all_ap, distmat, q_pids, g_pids, q_camids, g_camids)

    logger.info("Start saving ROC curve ...")
    fpr, tpr, pos, neg = visualizer.vis_roc_curve(args.output)
    visualizer.save_roc_info(args.output, fpr, tpr, pos, neg)
    logger.info("Finish saving ROC curve!")

    logger.info("Saving rank list result ...")
    query_indices = visualizer.vis_rank_list(args.output, args.vis_label, args.num_vis,
                                             args.rank_sort, args.label_sort, args.max_rank)
    logger.info("Finish saving rank list results!")

'''
AG-ReID

CUDA_VISIBLE_DEVICES=1 \
python3 demo/extract_feature.py \
--config-file logs/AG_ReID/1_16/85.53_88.46/config.yml \
--dataset-name AG_ReID \
--output ./AG-ReID_Rank_List_vitvd \
--num-vis 10 \
--label-sort descending \
--rank-sort descending \
--max-rank 8 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs/AG_ReID/1_16/85.53_88.46/model_best.pth
'''

'''
CARGO

CUDA_VISIBLE_DEVICES=0 \
python3 demo/extract_feature.py \
--config-file 之前的消融/CARGO消融/34597178/True/1.0_0.01_1.0/config.yml \
--dataset-name CARGO_AG \
--output ./CARGO_Rank_List \
--num-vis 50 \
--label-sort descending \
--rank-sort descending \
--max-rank 10 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs_6/34597178_7.1/True/1.0_0.01_1.0/model_best.pth
'''

'''
AG-ReIDv2  AG_ReID_v2_A2W  AG_ReID_v2_G2A

CUDA_VISIBLE_DEVICES=2 \
python3 demo/extract_feature.py \
--config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
--dataset-name AG_ReID_v2_A2W \
--output ./AG-ReIDv2_Rank_List \
--num-vis 2000 \
--label-sort descending \
--rank-sort descending \
--max-rank 10 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth
'''