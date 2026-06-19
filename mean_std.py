import numpy as np

ac = [82.19, 81.93, 82.39, 82.29]
aw = [85.47, 85.03, 84.92, 85.15]
ca = [80.66, 80.49, 80.38, 80.67]
wa = [82.12, 82.20, 81.59, 82.43]

for name, data in zip(['A->C', 'A->W', 'C->A', 'W->A'], [ac, aw, ca, wa]):
    mean = np.mean(data)
    std = np.std(data, ddof=1)
    print(f"{name}: Mean = {mean:.2f}, Std = {std:.2f}")