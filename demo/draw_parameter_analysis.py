import matplotlib.pyplot as plt
import numpy as np

# ================= 数据准备 =================
# 以你的 "binarize" 参数为例
x_labels = ['0', '0.01', '0.1', '1', '2']
x = np.arange(len(x_labels))

# 数据 (根据你的表格填入 A<->G 协议下的数据，或者 All 下的数据)
# 这里假设是 A<->G 的数据
rank1 = [68.12
, 68.75
, 70.63
, 72.5
, 66.87
]
map_score = [60.34
, 62.3
, 63.58
, 64.41
, 63.28
]
minp = [46.52
, 51.02
, 51.99
, 52.78
, 53.44
]

# 计算平均值作为折线图数据 (或者你自己定义的 Avg)
avg_score = [(r + m + i) / 3 for r, m, i in zip(rank1, map_score, minp)]

# ================= 绘图设置 =================
# 设置字体，保证符合论文要求 (Times New Roman 或 Arial)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 12

fig, ax1 = plt.subplots(figsize=(8, 5)) # 宽8，高5英寸

# 柱状图设置
bar_width = 0.25
opacity = 0.9

# 绘制柱状图 (左轴)
# 调整位置：x - width, x, x + width
rects1 = ax1.bar(x - bar_width, rank1, bar_width, label='Rank-1', color='#FF6B6B', alpha=opacity, zorder=3)
rects2 = ax1.bar(x, map_score, bar_width, label='mAP', color='#FFD93D', alpha=opacity, zorder=3)
rects3 = ax1.bar(x + bar_width, minp, bar_width, label='mINP', color='#4D96FF', alpha=opacity, zorder=3)

# 设置左轴标签和范围
ax1.set_xlabel(r'Weight $\gamma$ of $L_{\mathrm{CVPA}}$', fontsize=14, fontweight='bold')
ax1.set_ylabel('Performance (%)', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(x_labels, fontsize=12)
ax1.set_ylim(40, 80) # 根据你的数据范围调整，留出一点头部空间
ax1.tick_params(axis='y', labelsize=12)

# 添加网格线 (只在 Y 轴)
ax1.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)

# ================= 双轴折线图 =================
ax2 = ax1.twinx()  # 实例化第二个轴，共享 x 轴

# 绘制折线图 (右轴)
line = ax2.plot(x, avg_score, color='black', marker='o', linewidth=2, markersize=8, label='Avg', zorder=4)

# 设置右轴标签和范围
ax2.set_ylabel('Avg (%)', fontsize=14, fontweight='bold', rotation=270, labelpad=20)
ax2.set_ylim(50, 70) # 根据 Avg 的范围调整，尽量让折线在图中间波动，不要和柱子重叠太多
ax2.tick_params(axis='y', labelsize=12)

# ================= 图例合并 =================
# 因为有两个轴，需要手动合并图例
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# bbox_to_anchor 用于把图例放在图的上方外面
ax1.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=4, frameon=False, fontsize=12)

# ================= 保存 =================
plt.tight_layout()
plt.savefig('parameter_analysis_CVPA.pdf', format='pdf', dpi=300) # 保存为矢量图
plt.show()