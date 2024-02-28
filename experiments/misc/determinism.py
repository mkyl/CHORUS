from src.table_class import table_class_experiment
from src.table_class_results import calculate_results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

for i in range(5):
	for t in np.linspace(0.0, 1.0, num=5):
		file_path = f"artifacts/determinism/temperature_{t}_trial_{i+1}.csv"
		if not os.path.exists(file_path):
			table_class_experiment(file_path, temperature=t)

results = []
for i in range(5):
	for t in np.linspace(0.0, 1.0, num=5):
		file_path = f"artifacts/determinism/temperature_{t}_trial_{i+1}.csv"
		f_score, r_score, p_score = calculate_results(file_path)
		results.append((t, i, "F1 Score", f_score))
		results.append((t, i, "Recall", r_score))
		results.append((t, i, "Precision", p_score))

results = pd.DataFrame(results, columns=["Temperature", "Trial", "Score",
	"Value"])

plt.figure(figsize=(4, 2.5))
sns.lineplot(x="Temperature", y="Value",
             hue="Score", style="Score", marker="o",
             data=results)
plt.savefig("results/temperature.pdf", bbox_inches="tight")
