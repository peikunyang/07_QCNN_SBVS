import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "../Result/E_train"
data = pd.read_csv(file_path, sep=r'\s+', header=None, names=["Experimental", "Calculated"])

experimental = data["Experimental"].values
calculated = data["Calculated"].values

rmsd = np.sqrt(np.mean((experimental - calculated) ** 2))

from scipy.stats import pearsonr
correlation, _ = pearsonr(experimental, calculated)

with open("3_con_55_train_1.txt", "w") as f:
    f.write(f"RMSD: {rmsd:.6f}\n")
    f.write(f"Pearson Correlation: {correlation:.6f}\n")

x_min = np.floor(experimental.min())
x_max = np.ceil(experimental.max())
y_min = x_min
y_max = x_max

plt.figure(figsize=(10, 10), dpi=100)
plt.scatter(experimental, calculated, alpha=0.7, color='gray')

x_ticks = np.arange(x_min, x_max + 1, 5)
plt.xticks(x_ticks, fontsize=36, fontname='DejaVu Serif')
plt.yticks(x_ticks, fontsize=36, fontname='DejaVu Serif')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.xlabel("Experimental (kcal/mol)", fontsize=40, fontname='DejaVu Serif')
plt.ylabel("Calculated (kcal/mol)", fontsize=40, fontname='DejaVu Serif')

plt.grid(True, which='major', linestyle='--', linewidth=1)

plt.text(
    x=x_min + (x_max - x_min) * 0.05,
    y=y_max - (y_max - y_min) * 0.05,
    s=f"Qfilters: 55\nlr = $10^{{-2}}$\nCorr = {correlation:.4f}",
    fontsize=30,
    fontname='DejaVu Serif',
    ha='left',
    va='top'
)

plt.savefig("3_con_55_train_1.png", dpi=100, bbox_inches='tight')
#plt.show()

