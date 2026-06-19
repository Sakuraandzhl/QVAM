import numpy as np

# 配置
feat_dim = 768          # 每个特征的维度
num_parts = 5          # [view_related, view_invariant, view_final, mask]
top_k = 20              # 打印前多少维（这里是 0-19 共 20 维）
npy_paths = [
    "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam1_day_1_197.npy",
    "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam8_day_1_50970.npy",
    "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam1_day_2_633.npy",
    "/mnt/sda/sakura/Projects/Semantic-Alignment/demo/output/Cam8_day_2_51424.npy",
]

part_names = ["global_feats", "view_related_feats", "view_invariant_feats",
              "view_invariant_finally_feats", "mask"]

for idx, path in enumerate(npy_paths, 1):
    data = np.load(path)
    # 假设形状是 [1, 768*4] 或 [768*4]
    vec = data.reshape(-1)          # 展平到 [768*4]
    assert vec.shape[0] == feat_dim * num_parts, \
        f"{path} 维度不对: {vec.shape[0]} != {feat_dim * num_parts}"

    print(f"\n===== 文件 {idx}: {path} =====")
    for p, name in enumerate(part_names):
        start = p * feat_dim
        end = (p + 1) * feat_dim
        part_vec = vec[start:end]
        # 这里打印 0-19 共 20 个维度
        head = part_vec[:top_k]
        values_str = " ".join(f"{v:.6f}" for v in head)
        print(f"{name} 0-{top_k-1} 维度的值：")
        print(values_str)
        print()  # 空行分隔