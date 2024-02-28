import csv
import json
from pathlib import Path
from random import shuffle, seed
from ast import literal_eval

from tqdm import tqdm
import pandas as pd
import numpy as np

seed(0)

COLUMN_DEFS = Path("/Users/moe/research/data-lake/datasets/T2Dv2/extended_instance_goldstandard/property/")
csv_files = [x for x in COLUMN_DEFS.glob("*.csv") if x.stat().st_size > 0]

ground_truth = []

for csv_file in csv_files:
    with open(csv_file, "r", encoding="utf-8") as file:
        instance = pd.read_csv(file, index_col=None, names=["type", "name", "unknown", "col"])

    for row in instance.itertuples():
        ground_truth.append([csv_file.name.replace(".csv", ".json"), row.col, row.type]) 

results = pd.DataFrame(ground_truth, columns=["Table", "Col", "Label"])
# we only want DBPedia classes
results = results[results['Label'] != 'http://www.w3.org/2000/01/rdf-schema#label']

# next, load TaBERT embeddings

def lazy_eval(x):
    try:
        return literal_eval(x)
    except ValueError:
        return None

embeds = pd.read_csv("artifacts/tabert-embeddings.csv", header=[0], index_col=None)
embeds["embedding"] = embeds["embedding"].apply(lazy_eval)

data = pd.merge(results, embeds, left_on=["Table", "Col"], right_on=["table", "column"], how="left")

none_count = data['embedding'].isna().sum()
print(f"Number of None values in 'embedding' column: {none_count}")
data = data.dropna()

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(random_state = 0)
kfold = KFold(n_splits=5, shuffle=True, random_state=0)

from sklearn.metrics import f1_score, recall_score, precision_score, balanced_accuracy_score, make_scorer
f1_scorer = make_scorer(f1_score, zero_division=1, average="weighted")
recall_scorer = make_scorer(recall_score, zero_division=1, average="weighted")
precision_scorer = make_scorer(precision_score, zero_division=1, average="weighted")
ba_scorer = make_scorer(balanced_accuracy_score)


scores = cross_val_score(model, data.embedding.tolist(), data.Label, cv=kfold, scoring=f1_scorer)
print(f"Average F1 weighted score: {scores.mean():0.3f}")

scores = cross_val_score(model, data.embedding.tolist(), data.Label, cv=kfold, scoring=precision_scorer)
print(f"Average precision weighted score: {scores.mean():0.3f}")

scores = cross_val_score(model, data.embedding.tolist(), data.Label, cv=kfold, scoring=recall_scorer)
print(f"Average recall weighted score: {scores.mean():0.3f}")

scores = cross_val_score(model, data.embedding.tolist(), data.Label, cv=kfold, scoring=ba_scorer)
print(f"balanced accuracy: {scores.mean():0.3f}")
