# fastreid/utils/visualizer_compare.py
# encoding: utf-8

import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class Visualizer:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_model_output(self, dist1, dist2, ap1, ap2, q_pids, g_pids, q_camids, g_camids):
        """
        dist1: Distance matrix of Model 1 (QVAM/Ours)
        dist2: Distance matrix of Model 2 (SeCap/Baseline)
        """
        self.dist1 = dist1
        self.dist2 = dist2
        self.ap1 = ap1
        self.ap2 = ap2
        self.q_pids = q_pids
        self.g_pids = g_pids
        self.q_camids = q_camids
        self.g_camids = g_camids
        self.num_query = len(q_pids)
        self.num_gallery = len(g_pids)

    def get_matched_result(self, q_index, dist):
        q_pid = self.q_pids[q_index]
        q_camid = self.q_camids[q_index]

        # 1. 排序
        order = np.argsort(dist[q_index])
        
        # 2. 过滤掉同一摄像头下的同一ID (Junk)
        remove = (self.g_pids[order] == q_pid) & (self.g_camids[order] == q_camid)
        keep = np.invert(remove)
        
        # 3. 计算匹配情况 (1为对, 0为错)
        matches = (self.g_pids[order] == q_pid).astype(np.int32)
        
        cmc = matches[keep]
        sort_idx = order[keep]
        return cmc, sort_idx

    def vis_compare_rank_list(self, output, max_rank=10, num_vis=50):
        """
        可视化对比：筛选 SeCap Rank-1 对，但 QVAM Rank-1 错，可是 QVAM Top-K 命中数更多的案例
        """
        os.makedirs(output, exist_ok=True)
        print(f"Searching for 'Hard Positive' cases (SeCap R1 Correct vs QVAM R1 Wrong)...")
        #截断ap
        def compute_ap_k(cmc_full, max_k):
            # 总的 Ground Truth 数量即 cmc_full 里面 1 的个数
            total_gt = np.sum(cmc_full) 
            if total_gt == 0: return 0.0
            
            cmc_k = cmc_full[:max_k]
            hit_count = 0.0
            sum_precision = 0.0
            for i, val in enumerate(cmc_k):
                if val == 1:
                    hit_count += 1.0
                    sum_precision += hit_count / (i + 1.0)
            return sum_precision / total_gt
        count = 0
        for q_idx in tqdm.tqdm(range(self.num_query)):
            if count >= num_vis: break
            
            # 获取两个模型的检索结果
            cmc1, sort_idx1 = self.get_matched_result(q_idx, self.dist1) # Model 1 (QVAM)
            cmc2, sort_idx2 = self.get_matched_result(q_idx, self.dist2) # Model 2 (SeCap)
            
            # === 核心筛选逻辑 ===
            # 条件1: SeCap (Model 2) Rank-1 正确
            # 条件2: QVAM (Model 1) Rank-1 错误
            if cmc2[0] == 1 and cmc1[0] == 1 and self.ap2[q_idx] * 1.5 >self.ap1[q_idx] > 1.2 * self.ap2[q_idx]:
                
                # 条件3: QVAM 在 Top-K (比如Top-10) 中找回了更多正确的图
                # 这证明 QVAM 虽然错过了第一张，但整体召回能力更强
                # ap_k_qvam = compute_ap_k(cmc1, max_rank)
                # ap_k_secap = compute_ap_k(cmc2, max_rank)
                # if ap_k_qvam > ap_k_secap:
                #     self.save_compare_fig(q_idx, sort_idx1, sort_idx2, cmc1, cmc2, max_rank, output)
                #     count += 1
                if self.ap1[q_idx] > self.ap2[q_idx]: 
                    self.save_compare_fig(q_idx, sort_idx1, sort_idx2, cmc1, cmc2, max_rank, output, self.ap1[q_idx], self.ap2[q_idx])
                    count += 1
                # self.save_compare_fig(q_idx, sort_idx1, sort_idx2, cmc1, cmc2, max_rank, output)
                # count += 1
        print(f"Done! Saved {count} comparison figures to {output}")

    def save_compare_fig(self, q_idx, sort_idx1, sort_idx2, cmc1, cmc2, max_rank, output, ap1, ap2):
            num_cols = max_rank + 1
            
            # --- 核心修改：基于图片真实比例 (128x256 = 1:2) 计算画布大小 ---
            base_img_w = 1.5  # 每张图的物理宽度（英寸）
            base_img_h = base_img_w * (256 / 128)  # 严格保持 1:2，即 3.0 英寸
            
            # 行高额外增加一点点用于显示标题
            row_h = base_img_h + 0.5 
            
            fig_w = num_cols * base_img_w
            fig_h = 2 * row_h
            
            fig, axes = plt.subplots(2, num_cols, figsize=(fig_w, fig_h))
            
            # 这里的 wspace=0.02 会让图片几乎背靠背贴在一起
            plt.subplots_adjust(wspace=0.02, hspace=0.15, left=0.01, right=0.99, bottom=0.02, top=0.9)

            # 获取 Query 图片
            query_info = self.dataset[q_idx]
            try:
                query_img = np.asarray(Image.open(query_info[0]).resize((128, 256)), dtype=np.uint8)
            except Exception as e:
                print(f"Error loading query image: {e}")
                return

            query_pid = self.q_pids[q_idx]

            def draw_single_row(row_idx, sort_idx, cmc, model_name, current_ap):
                ax_query = axes[row_idx, 0]
                ax_query.imshow(query_img)
                
                title_text = f"{model_name}\nAP: {current_ap:.1%}"
                # pad 调整标题和图片的距离
                ax_query.set_title(title_text, fontsize=14, color='blue', fontweight='bold', pad=8)
                
                # 优化边框：从 -0.5 开始画能完美包裹边缘
                rect_q = plt.Rectangle((-0.5, -0.5), 128, 256, linewidth=6, edgecolor='blue', fill=False)
                ax_query.add_patch(rect_q)
                
                ax_query.axis("off")

                for i in range(max_rank):
                    g_idx = sort_idx[i]
                    real_g_idx = self.num_query + g_idx
                    gallery_info = self.dataset[real_g_idx]
                    img_path = gallery_info[0]
                    
                    ax = axes[row_idx, i + 1]
                    
                    try:
                        img = np.asarray(Image.open(img_path).resize((128, 256)), dtype=np.uint8)
                        ax.imshow(img)
                        
                        is_correct = (cmc[i] == 1)
                        color = 'green' if is_correct else 'red'
                        
                        # 同样优化了这里的边框包裹
                        rect = plt.Rectangle((-0.5, -0.5), 128, 256, linewidth=6, edgecolor=color, fill=False)
                        ax.add_patch(rect)
                        
                        if row_idx == 0:
                            ax.set_title(f"Rank {i+1}", fontsize=13, pad=8)
                            
                    except Exception as e:
                        ax.axis("off")
                        continue
                    
                    ax.axis("off")

            draw_single_row(0, sort_idx1, cmc1, "QVAM (Ours)", ap1)
            draw_single_row(1, sort_idx2, cmc2, "SeCap", ap2)

            filename = f"Compare_Q{q_idx}_ID{query_pid}.jpg"
            save_path = os.path.join(output, filename)
            # 注意这里不用 tight_layout，直接存
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.05, dpi=150)
            plt.close()