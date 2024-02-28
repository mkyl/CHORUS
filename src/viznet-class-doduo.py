import csv
import json
from pathlib import Path
from random import shuffle, seed
from ast import literal_eval

from tqdm import tqdm
import pandas as pd
import numpy as np

seed(0)

DODUO_EMBEDS = Path(
    "/Users/moe/research/data-lake/artifacts/doduo-viznet-embeddings.parquet"
)
LABELS = Path(
    "/Users/moe/research/data-lake/datasets/sherlock-viznet/test_labels.parquet"
)
# next, load TaBERT embeddings


embeds = pd.read_parquet(DODUO_EMBEDS)
labels = pd.read_parquet(LABELS)

data = pd.merge(embeds, labels, right_index=True, left_index=True, how="left")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

model = MLPClassifier(random_state=0)
kfold = KFold(n_splits=2, shuffle=True, random_state=0)

from sklearn.metrics import (
    f1_score,
    recall_score,
    precision_score,
    balanced_accuracy_score,
    make_scorer,
)

f1_scorer = make_scorer(f1_score, average="weighted")
recall_scorer = make_scorer(recall_score, average="weighted")
precision_scorer = make_scorer(precision_score, average="weighted")

scores = cross_val_score(
    model, data.embedding.tolist(), data.type, cv=kfold, scoring=f1_scorer
)
print(f"Average F1 weighted score: {scores.mean():0.3f}")

scores = cross_val_score(
    model, data.embedding.tolist(), data.type, cv=kfold, scoring=precision_scorer
)
print(f"Average precision weighted score: {scores.mean():0.3f}")

scores = cross_val_score(
    model, data.embedding.tolist(), data.type, cv=kfold, scoring=recall_scorer
)
print(f"Average recall weighted score: {scores.mean():0.3f}")
