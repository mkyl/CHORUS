import argparse
import pandas as pd
from doduo import Doduo

# Load Doduo model
args = argparse.Namespace
args.model = "viznet" # or args.model = "viznet"
doduo = Doduo(args)

import csv
import json
from pathlib import Path
from random import shuffle
import ast

from tqdm import tqdm

VIZNET = "/Users/moe/research/data-lake/artifacts/viznet-subset.parquet"
OUTPUT = "/Users/moe/research/data-lake/artifacts/doduo-viznet-embeddings.parquet"

results = []

X = pd.read_parquet(VIZNET)

for row in tqdm(X.itertuples(), total=X.shape[0]):
    vals = ast.literal_eval(row.values)
    try:
        df = pd.Series(vals).to_frame()
        annot_df1 = doduo.annotate_columns(df)
    except RuntimeError:
        print("embedding error")
        continue
    vector = sum(annot_df1.colemb).tolist()
    results.append([row.Index, vector])


df = pd.DataFrame(results, columns=["id", "embedding"])
df.set_index('id', inplace=True)
df.to_parquet(OUTPUT)
