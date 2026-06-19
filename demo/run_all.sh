
# 设置出错即停
set -e

echo ">>> Step 1: Extracting QVAM Features..."
cd /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer
# 运行 QVAM 提取脚本 (记得在脚本里把 feat 保存为 qvam_feats.npy)
CUDA_VISIBLE_DEVICES=2 python3 demo/rank_list_compare.py \
    --config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
    --dataset-name AG_ReID_v2_A2W \
    --output rank_list_compare/A2W \
    MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth

echo ">>> Step 2: Extracting SeCap Features..."
cd /mnt/sda/sakura/Projects/SeCap-AGPReID-main
# 运行 SeCap 提取脚本 (记得保存为 secap_feats.npy)
# 注意：如果 SeCap 需要不同的 conda 环境，可以在这里切换
# source activate secap_env

CUDA_VISIBLE_DEVICES=2 python3 demo/rank_list_compare.py \
    --config-file configs/AGReIDv2/secap.yml \
    --dataset-name AG_ReID_v2_A2W \
    --output /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer/rank_list_compare/A2W \
    MODEL.WEIGHTS logs/AG_ReID_v2/SeCap/model_best.pth

echo ">>> Step 3: Visualizing Comparison..."
cd /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer
# 运行可视化脚本 (读取两个 npy 文件)
python3 ./demo/rank_list_compare_visualization.py \
    --npdir rank_list_compare/A2W \
    --num_query 2209 \
    --max_rank 10 \
    --num_vis 20
echo ">>> All Done!"


# # 设置出错即停
# set -e

# echo ">>> Step 1: Extracting QVAM Features..."
# cd /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer
# # 运行 QVAM 提取脚本 (记得在脚本里把 feat 保存为 qvam_feats.npy)
# CUDA_VISIBLE_DEVICES=2 python3 demo/rank_list_compare.py \
#     --config-file logs/AG_ReID_v2/123891/0.0001_1_4_best/config.yml \
#     --dataset-name AG_ReID_v2_G2A \
#     --output rank_list_compare/G2A \
#     MODEL.WEIGHTS logs/AG_ReID_v2/123891/0.0001_1_4_best/model_best.pth

# echo ">>> Step 2: Extracting SeCap Features..."
# cd /mnt/sda/sakura/Projects/SeCap-AGPReID-main
# # 运行 SeCap 提取脚本 (记得保存为 secap_feats.npy)
# # 注意：如果 SeCap 需要不同的 conda 环境，可以在这里切换
# # source activate secap_env

# CUDA_VISIBLE_DEVICES=2 python3 demo/rank_list_compare.py \
#     --config-file configs/AGReIDv2/secap.yml \
#     --dataset-name AG_ReID_v2_G2A \
#     --output /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer/rank_list_compare/G2A \
#     MODEL.WEIGHTS logs/AG_ReID_v2/SeCap/model_best.pth

# echo ">>> Step 3: Visualizing Comparison..."
# cd /mnt/sda/sakura/Projects/Prompts-driven-Adaptive-View-Disentangling-Transformer
# # 运行可视化脚本 (读取两个 npy 文件)
# python3 ./demo/rank_list_compare_visualization.py \
#     --npdir rank_list_compare/G2A \
#     --num_query 1811 \
#     --max_rank 10

# echo ">>> All Done!"