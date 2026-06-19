import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import re

# 解决中文字体警告（Ubuntu系统适配）
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _ensure_2d(array):
    """
    将任意形状的特征数组转换为 [N, D] 形式，方便统一处理。
    """
    arr = np.asarray(array)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _is_mask_like(chunk, tol=1e-3):
    """
    判断一个向量是否满足 mask 的值域特性（0-1 之间）。
    """
    return np.all((chunk >= -tol) & (chunk <= 1 + tol))


def _infer_layout(vector, feat_dim=None, source_path=None):
    """
    推断向量是如何拼接 view_related/view_invariant/…/mask 的。
    兼容两种情况：
        1) component-first: shape = [parts, feat_dim] （旧逻辑，沿 dim=1 拼接）
        2) channel-first:  shape = [feat_dim, parts] （新逻辑，沿 dim=2 拼接）
    """
    total_dim = vector.shape[-1]
    part_candidates = [4, 3]
    layouts = []

    if feat_dim is not None:
        for parts in part_candidates:
            if total_dim == parts * feat_dim:
                layouts.append({
                    "parts": parts,
                    "feat_dim": feat_dim,
                    "priority": 2  # 配置匹配优先级较高
                })
                break

    for parts in part_candidates:
        if total_dim % parts == 0:
            inferred_dim = total_dim // parts
            layouts.append({
                "parts": parts,
                "feat_dim": inferred_dim,
                "priority": 1
            })

    if not layouts:
        raise ValueError(
            f"无法根据向量维度 {total_dim} 推断特征拆分方式，请检查输入或显式指定 feat_dim。"
        )

    def _reshape(vector, parts, dim, orientation):
        if orientation == "component_first":
            return vector.reshape(parts, dim)
        else:  # channel_first
            return vector.reshape(dim, parts).transpose(1, 0)

    best = None
    for layout in layouts:
        for orientation in ("component_first", "channel_first"):
            try:
                matrix = _reshape(vector, layout["parts"], layout["feat_dim"], orientation)
            except ValueError:
                continue

            mask_row = matrix[-1]
            mask_ok = _is_mask_like(mask_row)
            score = (
                layout["priority"],
                1 if mask_ok else 0,
                1 if orientation == "component_first" else 0
            )

            if best is None or score > best["score"]:
                best = {
                    "parts": layout["parts"],
                    "feat_dim": layout["feat_dim"],
                    "matrix": matrix,
                    "score": score,
                }

    if best is None:
        raise ValueError("无法找到合适的特征布局，请检查输入数据。")

    if (feat_dim is not None and best["feat_dim"] != feat_dim and source_path):
        print(
            f"提示：文件 {source_path} 的向量维度为 {total_dim}，"
            f"自动推断 feat_dim={best['feat_dim']}（原配置为 {feat_dim}）。"
        )

    return best


def _split_feature_components(vector, feat_dim, source_path=None):
    """
    根据 ViewTransformer 的输出格式，将拼接后的特征拆分为对应部分。
    支持以下格式：
        - [view_related, view_invariant, mask]         -> 3 * feat_dim
        - [view_related, view_invariant, view_final, mask] -> 4 * feat_dim
    """
    layout = _infer_layout(vector, feat_dim=feat_dim, source_path=source_path)
    matrix = layout["matrix"]  # [parts, feat_dim]

    view_related = matrix[0]
    view_invariant = matrix[1]
    view_invariant_final = matrix[2] if layout["parts"] == 4 else None
    mask = matrix[-1]

    return {
        "view_related": view_related,
        "view_invariant": view_invariant,
        "view_invariant_final": view_invariant_final,
        "mask": mask
    }


def _select_invariant_feat(sample, prefer_final=True):
    """
    返回用于一致性计算的视角无关特征。
    当存在 refined 特征时优先使用。
    """
    if prefer_final and sample.get("view_invariant_final") is not None:
        return sample["view_invariant_final"]
    return sample["view_invariant"]

def generate_mask_interpretability_report(
    feat_paths,  # 多个.npy文件路径（建议包含：同一ID不同视角、不同ID同一视角样本）
    sample_info,  # 样本信息列表，格式：[{"id": "ID1", "view": "Ground", "name": "样本1"}, ...]
    feat_dim=2048,
    save_path="mask_interpretability_report.md"
):
    """
    生成完整的mask可解释性量化报告
    Args:
        feat_paths: 样本.npy文件路径列表（需与sample_info一一对应）
        sample_info: 样本元信息（ID、视角、名称）
        feat_dim: 特征维度（与配置一致）
        save_path: 报告保存路径
    """
    # 1. 加载所有样本数据（添加异常处理）
    all_data = []
    for path, info in zip(feat_paths, sample_info):
        try:
            if not os.path.exists(path):
                print(f"警告：文件 {path} 不存在，已跳过")
                continue
            feat_cat = _ensure_2d(np.load(path))
            if feat_cat.shape[0] > 1:
                print(f"提示：文件 {path} 包含 {feat_cat.shape[0]} 个样本，仅使用第一个。")
            components = _split_feature_components(feat_cat[0], feat_dim, source_path=path)
            all_data.append({
                "info": info,
                "view_related": components["view_related"],
                "view_invariant": components["view_invariant"],
                "view_invariant_final": components["view_invariant_final"],
                "mask": components["mask"]
            })
        except Exception as e:
            print(f"警告：加载文件 {path} 失败，错误：{str(e)}，已跳过")
            continue
    
    if not all_data:
        print("错误：无有效样本数据，无法生成报告")
        return
    
    # 2. 量化指标计算（处理空值避免警告）
    def calc_metrics(data_list):
        masks = np.array([d["mask"] for d in data_list])

        # 指标1：mask分布合理性（30分）
        mask_mean = np.mean(masks)
        mask_std = np.std(masks)
        mean_score = max(0, 15 - abs(mask_mean - 0.5) * 30)
        std_score = 15 if mask_std > 0.1 else mask_std * 150
        mask_dist_score = mean_score + std_score

        # 指标2：特征一致性（40分）- 同一ID不同视角
        same_id_diff_view_pairs = []
        same_id_diff_view_pairs_final = []
        ids = list(set([d["info"]["id"] for d in data_list]))

        def _collect_similarity(sample_a, sample_b, top_idx, use_final):
            feat_a = _select_invariant_feat(sample_a, prefer_final=use_final)
            feat_b = _select_invariant_feat(sample_b, prefer_final=use_final)
            if feat_a is None or feat_b is None:
                return None
            sub_a = feat_a[top_idx]
            sub_b = feat_b[top_idx]
            if np.all(sub_a == 0) or np.all(sub_b == 0):
                return None
            return np.corrcoef(sub_a, sub_b)[0, 1]

        for idx in ids:
            id_samples = [d for d in data_list if d["info"]["id"] == idx]
            if len(id_samples) >= 2 and len(set([d["info"]["view"] for d in id_samples])) >= 2:
                top50_idx = np.argsort(id_samples[0]["mask"])[-50:][::-1]
                for i in range(len(id_samples)):
                    for j in range(i + 1, len(id_samples)):
                        if id_samples[i]["info"]["view"] == id_samples[j]["info"]["view"]:
                            continue
                        sim_raw = _collect_similarity(id_samples[i], id_samples[j], top50_idx, use_final=False)
                        if sim_raw is not None:
                            same_id_diff_view_pairs.append(sim_raw)
                        sim_final = _collect_similarity(id_samples[i], id_samples[j], top50_idx, use_final=True)
                        if sim_final is not None:
                            same_id_diff_view_pairs_final.append(sim_final)

        def _score_from_sims(pairs, weight=40):
            if not pairs:
                return 0, 0
            avg_sim = np.mean(pairs)
            return min(weight, avg_sim * weight / 0.5), avg_sim

        consistency_score_raw, same_id_diff_view_avg_sim = _score_from_sims(same_id_diff_view_pairs, weight=30)
        consistency_score_final, same_id_diff_view_avg_sim_final = _score_from_sims(same_id_diff_view_pairs_final, weight=10)
        total_consistency_score = consistency_score_raw + consistency_score_final

        # 指标3：解耦集中性（30分）- Top/Bottom维度区分度
        mask_avg_per_dim = np.mean(masks, axis=0)
        top10_avg = np.mean(np.sort(mask_avg_per_dim)[-10:])
        bottom10_avg = np.mean(np.sort(mask_avg_per_dim)[:10])
        decouple_score = min(30, max(0, (top10_avg - bottom10_avg) * 30 / 0.6))

        # 总分（处理nan）
        total_score = np.nan_to_num(mask_dist_score + total_consistency_score + decouple_score, nan=0.0)
        grade = "优秀" if total_score >= 85 else "良好" if total_score >= 70 else "一般" if total_score >= 50 else "较差"

        return {
            "mask_mean": mask_mean,
            "mask_std": mask_std,
            "same_id_diff_view_avg_sim": same_id_diff_view_avg_sim,
            "same_id_diff_view_avg_sim_final": same_id_diff_view_avg_sim_final,
            "top10_mask_avg": top10_avg,
            "bottom10_mask_avg": bottom10_avg,
            "mask_dist_score": mask_dist_score,
            "consistency_score_raw": consistency_score_raw,
            "consistency_score_final": consistency_score_final,
            "consistency_score": total_consistency_score,
            "decouple_score": decouple_score,
            "total_score": total_score,
            "grade": grade,
            "has_valid_pairs": len(same_id_diff_view_pairs) > 0,
            "has_valid_pairs_final": len(same_id_diff_view_pairs_final) > 0
        }
    
    metrics = calc_metrics(all_data)
    
    # 3. 生成样本对比表格
    def generate_sample_table(data_list):
        has_final = any(d["view_invariant_final"] is not None for d in data_list)
        if has_final:
            table = "| 样本名称 | ID | 视角 | mask均值 | Top10大mask维度 | 原始相似度 | 精炼相似度 |\n"
            table += "|----------|----|------|----------|-----------------|------------|------------|\n"
        else:
            table = "| 样本名称 | ID | 视角 | mask均值 | Top10大mask维度 | 与其他同ID不同视角样本相似度 |\n"
            table += "|----------|----|------|----------|-----------------|------------------------------|\n"

        for d in data_list:
            mask = d["mask"]
            mask_mean = np.mean(mask)
            top10_idx = np.argsort(mask)[-10:][::-1]
            # 计算同ID不同视角相似度
            same_id_samples = [x for x in data_list if x["info"]["id"] == d["info"]["id"] and x["info"]["view"] != d["info"]["view"]]
            sim_str_raw = []
            sim_str_final = []
            if same_id_samples:
                top50_idx = np.argsort(mask)[-50:][::-1]
                for s in same_id_samples:
                    sim_raw = _select_invariant_feat(d, False)
                    sim_peer_raw = _select_invariant_feat(s, False)
                    if sim_raw is not None and sim_peer_raw is not None:
                        sub_a = sim_raw[top50_idx]
                        sub_b = sim_peer_raw[top50_idx]
                        if np.all(sub_a == 0) or np.all(sub_b == 0):
                            sim_str_raw.append("无效特征")
                        else:
                            sim_str_raw.append(f"{np.corrcoef(sub_a, sub_b)[0,1]:.3f}")
                    else:
                        sim_str_raw.append("缺失")

                    if has_final:
                        sim_final = _select_invariant_feat(d, True)
                        sim_peer_final = _select_invariant_feat(s, True)
                        if sim_final is None or sim_peer_final is None:
                            sim_str_final.append("缺失")
                        else:
                            sub_a_f = sim_final[top50_idx]
                            sub_b_f = sim_peer_final[top50_idx]
                            if np.all(sub_a_f == 0) or np.all(sub_b_f == 0):
                                sim_str_final.append("无效特征")
                            else:
                                sim_str_final.append(f"{np.corrcoef(sub_a_f, sub_b_f)[0,1]:.3f}")

            sim_raw_out = "、".join(sim_str_raw) if sim_str_raw else "无"
            if has_final:
                sim_final_out = "、".join(sim_str_final) if sim_str_final else "无"
                table += f"| {d['info']['name']} | {d['info']['id']} | {d['info']['view']} | {mask_mean:.4f} | {top10_idx} | {sim_raw_out} | {sim_final_out} |\n"
            else:
                table += f"| {d['info']['name']} | {d['info']['id']} | {d['info']['view']} | {mask_mean:.4f} | {top10_idx} | {sim_raw_out} |\n"
        return table
    
    sample_table = generate_sample_table(all_data)
    
    # 4. 问题诊断与优化建议（补充样本配对提示）
    def generate_diagnosis(metrics):
        diagnosis = ""
        suggestions = ""
        
        if not metrics["has_valid_pairs"]:
            diagnosis += "- 无有效「同一ID不同视角」样本配对，无法计算身份特征一致性；\n"
            suggestions += "- 需选择至少1组「同一ID、不同视角」的样本（如 Cam1-197 和 Cam6-197）；\n"
        
        # mask分布问题
        if metrics["mask_mean"] < 0.3 or metrics["mask_mean"] > 0.7:
            diagnosis += "- mask均值偏离合理范围（0.3-0.7），解耦倾向单一方向；\n"
            suggestions += "- 调整mask初始化偏置（如mask_init_bias=0.5），避免初始值极端；\n"
            suggestions += "- 增大binarize_loss权重（如从0.01调整为0.05），强制mask二值化；\n"
        
        # 特征一致性问题
        if metrics["has_valid_pairs"] and metrics["same_id_diff_view_avg_sim"] < 0.5:
            diagnosis += "- 同一ID不同视角的身份特征一致性不足；\n"
            suggestions += "- 增大semantic_alignment_loss中的align_loss权重，强化身份特征对齐；\n"
            suggestions += "- 调整TripletLoss的margin（如从0.3改为0.5），提升身份特征聚集性；\n"

        if metrics["has_valid_pairs_final"]:
            if metrics["same_id_diff_view_avg_sim_final"] < metrics["same_id_diff_view_avg_sim"]:
                diagnosis += "- 精炼后的视角无关特征并未优于原始特征，可能存在过拟合；\n"
                suggestions += "- 检查 ViewTransformer 中 feature_enhance_mlp 的学习率或dropout，防止破坏已有结构；\n"
            elif metrics["same_id_diff_view_avg_sim_final"] < 0.5:
                diagnosis += "- 精炼后的视角无关特征一致性不足；\n"
                suggestions += "- 适当提高 feature_enhance_mlp 的深度或 Semantic Lambda，强化精炼约束；\n"
        
        # 解耦集中性问题
        if (metrics["top10_mask_avg"] - metrics["bottom10_mask_avg"]) < 0.4:
            diagnosis += "- Top/Bottom维度mask区分度低，解耦效果不明显；\n"
            suggestions += "- 增大view_transformer的MLP_ratio（如从4改为6），增强mask学习能力；\n"
            suggestions += "- 减少semantic_lambda权重（如从1.0改为0.8），避免过度约束；\n"
        
        if not diagnosis:
            diagnosis += "- 无明显问题，mask可解释性符合预期；\n"
            suggestions += "- 保持现有配置，可尝试微调损失权重进一步提升性能；\n"
        
        return diagnosis, suggestions
    
    diagnosis, suggestions = generate_diagnosis(metrics)
    final_avg_display = "无有效样本" if not metrics["has_valid_pairs_final"] else f"{metrics['same_id_diff_view_avg_sim_final']:.4f}"
    
    # 5. 生成markdown报告
    report = f"""# Mask可解释性量化报告
生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
## 一、总体评价
- 量化总分：{metrics['total_score']:.1f}分（满分100分）
- 评级：{metrics['grade']}
- 核心结论：{'mask具备良好可解释性，能有效区分视角无关身份特征和视角相关特征' if metrics['grade'] in ['优秀', '良好'] else 'mask可解释性不足，核心原因：' + ('无有效样本配对' if not metrics['has_valid_pairs'] else '解耦效果或特征一致性差')}

## 二、关键指标详情
| 指标名称 | 数值 | 权重占比 | 得分 |
|----------|------|----------|------|
| mask分布合理性 | 均值：{metrics['mask_mean']:.4f}，标准差：{metrics['mask_std']:.4f} | 30% | {metrics['mask_dist_score']:.1f} |
| 身份特征一致性（原始） | 平均相似度：{metrics['same_id_diff_view_avg_sim']:.4f} | 30% | {metrics['consistency_score_raw']:.1f} |
| 身份特征一致性（精炼） | 平均相似度：{final_avg_display} | 10% | {metrics['consistency_score_final']:.1f} |
| 解耦集中性（Top10-Bottom10均值差） | {metrics['top10_mask_avg'] - metrics['bottom10_mask_avg']:.4f} | 30% | {metrics['decouple_score']:.1f} |

## 三、样本对比详情
{sample_table}

## 四、问题诊断
{diagnosis}

## 五、优化建议
{suggestions}

## 六、使用说明
1. 核心要求：必须包含至少1组「同一ID、不同视角」的样本，否则无法验证mask可解释性；
2. 若评级为「优秀/良好」：可直接用于推理，mask能有效辅助跨视角重识别；
3. 若评级为「一般/较差」：按优化建议调整配置后，重新训练并验证；
4. 重点关注「同一ID不同视角相似度」，需≥0.5才说明身份特征稳定。
"""
    
    # 保存报告
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    # 生成可视化图
    plt.figure(figsize=(15, 5))
    # 1. mask分布直方图
    all_masks = np.array([d["mask"] for d in all_data])
    plt.subplot(131)
    plt.hist(all_masks.flatten(), bins=50, alpha=0.7, color='steelblue')
    plt.xlabel("Mask Value")
    plt.ylabel("Count")
    plt.title(f"Mask Value Distribution (Mean: {metrics['mask_mean']:.4f})")
    # 2. 各维度mask均值
    mask_avg_per_dim = np.mean(all_masks, axis=0)
    plt.subplot(132)
    plt.plot(np.sort(mask_avg_per_dim)[::-1], color='darkred', linewidth=1)
    plt.xlabel("Feature Dimension (Sorted)")
    plt.ylabel("Average Mask Value")
    plt.title("Average Mask Value per Dimension (Descending)")
    # 3. 同ID不同视角相似度分布
    plt.subplot(133)
    if metrics["has_valid_pairs"]:
        same_id_diff_view_pairs = []
        ids = list(set([d["info"]["id"] for d in all_data]))
        for idx in ids:
            id_samples = [d for d in all_data if d["info"]["id"] == idx]
            if len(id_samples) >= 2 and len(set([d["info"]["view"] for d in id_samples])) >= 2:
                top50_idx = np.argsort(id_samples[0]["mask"])[-50:][::-1]
                for i in range(len(id_samples)):
                    for j in range(i+1, len(id_samples)):
                        if id_samples[i]["info"]["view"] != id_samples[j]["info"]["view"]:
                            feat_i = id_samples[i]["view_invariant"][top50_idx]
                            feat_j = id_samples[j]["view_invariant"][top50_idx]
                            if not (np.all(feat_i == 0) or np.all(feat_j == 0)):
                                sim = np.corrcoef(feat_i, feat_j)[0,1]
                                same_id_diff_view_pairs.append(sim)
        plt.hist(same_id_diff_view_pairs, bins=20, alpha=0.7, color='forestgreen')
        plt.xlabel("Similarity")
        plt.ylabel("Count")
        plt.title(f"Same ID Different View Similarity (Mean: {np.mean(same_id_diff_view_pairs):.3f})")
    else:
        plt.text(0.5, 0.5, "No Valid Same-ID Different-View Pairs", ha='center', va='center', transform=plt.gca().transAxes)
    plt.tight_layout()
    plt.savefig("mask_report_visualization.png", dpi=300, bbox_inches='tight')
    
    print(f"\n报告已生成：{save_path}")
    print(f"可视化图已生成：mask_report_visualization.png")
    print(f"总体评级：{metrics['grade']}（{metrics['total_score']:.1f}分）")
    if not metrics["has_valid_pairs"]:
        print("⚠️  关键警告：无有效「同一ID不同视角」样本配对，请选择同ID但相机编号≤5和>5的样本（如Cam1-197和Cam6-197）")

# ----------------------
# 主函数（按你的规则自动生成sample_info）
# ----------------------
if __name__ == "__main__":
    # 1. 配置样本路径（请替换为你实际存在的文件！）
    # 核心要求：至少包含1组「同一ID不同视角」的样本（如 Cam1-197（camid=1≤5→Aerial）和 Cam6-197（camid=6>5→Ground））
    feat_paths = [
        "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam1_day_1_197.npy",  # ID=197，camid=1→Aerial
        "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam8_day_1_50970.npy",  # ID=197，camid=6→Ground（不同视角）
        "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam1_day_2_633.npy",  # ID=633，camid=1→Aerial（对比样本）
    ]

    # 2. 按你的规则自动生成 sample_info
    sample_info = []
    for path in feat_paths:
        # 提取文件名（不含路径和后缀）
        filename = os.path.basename(path).replace(".npy", "")
        parts = filename.split("_")
        
        # 验证文件名格式（必须是 CamX_day_num_ID.npy，拆分后长度为4）
        if len(parts) != 4:
            print(f"警告：文件名 {filename} 格式错误（需为 CamX_day_num_ID.npy),已跳过")
            continue
        
        # 按你的规则提取信息
        try:
            cam = parts[0]  # 相机名称（如 Cam1、Cam6）
            camid = int(parts[0][3:])  # 相机编号（Cam1→1，Cam6→6）
            time = parts[1]  # 时间（day/night）
            id = parts[2]  # 样本ID（如 197、633）
            viewid = 'Aerial' if camid <= 5 else 'Ground'  # 视角判断规则
        except Exception as e:
            print(f"警告：解析文件名 {filename} 失败，错误：{str(e)}，已跳过")
            continue
        
        # 生成样本名称（如 Cam1-day-ID197）
        name = f"{cam}-{time}-ID{id}"
        
        # 添加到 sample_info
        sample_info.append({
            "id": id,
            "view": viewid,  # 视角使用viewid
            "name": name
        })

    # 3. 过滤无效样本（仅保留有对应sample_info的路径）
    valid_feat_paths = []
    valid_sample_info = []
    for path, info in zip(feat_paths, sample_info):
        if os.path.exists(path) and info["view"] in ['Aerial', 'Ground']:
            valid_feat_paths.append(path)
            valid_sample_info.append(info)
    
    if not valid_sample_info:
        print("错误：无有效样本，请检查文件路径和文件名格式")
    else:
        # 4. 生成报告
        generate_mask_interpretability_report(
            feat_paths=valid_feat_paths,
            sample_info=valid_sample_info,
            feat_dim=768,  # 需与你的骨干网络输出维度一致
            save_path="mask_interpretability_report.md"
        )

        # 打印自动生成的 sample_info 供检查
        print("\n自动生成的 sample_info:")
        for info in valid_sample_info:
            print(f"- {info}")