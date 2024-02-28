import pandas as pd

probs = pd.read_parquet("artifacts/sherlock-viznet-probs.parquet")
probs["pred"] = probs.idxmax(axis="columns")

LABELS = "/Users/moe/research/data-lake/datasets/sherlock-viznet/test_labels.parquet"
labels = pd.read_parquet(LABELS)

results = pd.merge(probs, labels, right_index=True, left_index=True)
results = results.fillna("Unknown")

from sklearn.metrics import classification_report
print(classification_report(results.type, results.pred, digits=3))
