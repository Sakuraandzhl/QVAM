import numpy as np
import torch
import sys
import argparse
sys.path.append('.')
from fastreid.evaluation.rank import evaluate_rank
from fastreid.utils.visualizer_compare import Visualizer
def get_parser():
    parser = argparse.ArgumentParser(description="Feature extraction with reid models")
    parser.add_argument(
        "--npdir",
        default="./vis_rank_list",
        help="a file or directory to save rankling list result.",
    )
    parser.add_argument(
        "--num_query",
        type=int,
        default=100,
        help="a file or directory to save rankling list result.",
    )
    parser.add_argument(
        "--max_rank",
        type=int,
        default=10,
        help="a file or directory to save rankling list result.",
    )
    parser.add_argument(
        "--num_vis",
        type=int,
        default=10000,
        help="a file or directory to save rankling list result.",
    )
    return parser
if __name__ == '__main__':
    args = get_parser().parse_args()
    # 1. 加载数据
    feats_qvam = torch.from_numpy(np.load(args.npdir + "/QVAM/feats_qvam.npy"))
    feats_secap = torch.from_numpy(np.load(args.npdir + "/SeCap/feats_secap.npy"))
    pids = np.load(args.npdir + "/QVAM/pids.npy")
    camids = np.load(args.npdir + "/QVAM/camids.npy")
    img_paths = np.load(args.npdir + "/QVAM/img_paths.npy")

    # 构造一个假的 dataset 对象给 Visualizer 用
    # Visualizer 需要 dataset[i] 返回 (img_path, pid, camid)
    class FakeDataset:
        def __init__(self, paths, pids, camids):
            self.data = list(zip(paths, pids, camids))
        def __getitem__(self, idx):
            return self.data[idx]

    dataset = FakeDataset(img_paths, pids, camids)
    # 2. 计算距离
    # 这里你需要知道 num_query 是多少。可以手动指定或者从文件名解析。
    # AG-ReID v2 A2W: query=2209
    num_query = args.num_query

    q_f1, g_f1 = feats_qvam[:num_query], feats_qvam[num_query:]
    dist1 = 1 - torch.mm(q_f1, g_f1.t()).numpy()

    q_f2, g_f2 = feats_secap[:num_query], feats_secap[num_query:]
    dist2 = 1 - torch.mm(q_f2, g_f2.t()).numpy()
    cmc1, all_ap1, all_inp1 = evaluate_rank(dist1, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])
    cmc2, all_ap2, all_inp2 = evaluate_rank(dist2, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])
    # 3. 画图
    vis = Visualizer(dataset)
    vis.get_model_output(dist1, dist2, all_ap1, all_ap2, pids[:num_query], pids[num_query:], camids[:num_query], camids[num_query:])
    vis.vis_compare_rank_list(args.npdir, max_rank=args.max_rank, num_vis=args.num_vis)