from src.table_class import table_class_experiment
from src.table_class_results import calculate_results

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

ablations = ["anchoring", "metadata", "demonstration", "prefixes"]

if not os.path.exists("artifacts/ablations/"):
        os.makedirs("artifacts/ablations/")

for x in ablations:
	file_path = f"artifacts/ablations/{x}.csv"
	if not os.path.exists(file_path):
		table_class_experiment(file_path, temperature=0, ablation=x)

results = []
for x in ablations:
	file_path = f"artifacts/ablations/{x}.csv"
	f_score, r_score, p_score = calculate_results(file_path, ablation=x)
	results.append((x, "F1 Score", f_score))
	results.append((x, "Recall", r_score))
	results.append((x, "Precision", p_score))

results = pd.DataFrame(results, columns=["Ablation", "Score",
	"Value"])

print(results)

plt.figure(figsize=(4, 2.5))
sns.barplot(y="Ablation", x="Value",
             hue="Score",
             data=results)
plt.savefig("results/ablations.pdf", bbox_inches="tight")
