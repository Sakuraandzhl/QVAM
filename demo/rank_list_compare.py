# encoding: utf-8
"""
@author:  xingyu liao
@contact: sherlockliao01@gmail.com
"""
"""
核心目标：同时获取两个模型的测试结果，并在visualizer中实现可视化一个模型的rank1为0而另一个模型的rank1为1的结果，解释QVAM的rank1低于SeCap但是整体排名效果优于SeCap
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
    feats = torch.cat(feats, dim=0)
    np.save(args.output+"/QVAM/feats_qvam.npy", feats.cpu().numpy())
    np.save(args.output+"/QVAM/pids.npy", pids)
    np.save(args.output+"/QVAM/camids.npy", camids)
    # 还需要保存 image paths 以便 visualizer 读图
    # test_loader.dataset 是一个 list of tuples (path, pid, camid)
    img_paths = [x['img_paths'] for x in test_loader.dataset]
    np.save(args.output+"/QVAM/img_paths.npy", img_paths)


'''
AG-ReIDv2  AG_ReID_v2_A2W  AG_ReID_v2_G2A

CUDA_VISIBLE_DEVICES=2 \
python3 demo/rank_list_compare.py \
--config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
--dataset-name AG_ReID_v2_A2W \
--output ./AGv2-rank_list_compare \
--num-vis 2209 \
--label-sort descending \
--rank-sort descending \
--max-rank 8 \
DATALOADER.NUM_WORKERS 0 \
MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth
'''