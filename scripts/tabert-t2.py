from table_bert import TableBertModel
import pandas as pd

model = TableBertModel.from_pretrained(
    'tabert_large_k3/model.bin',
)

from table_bert import Table, Column
from table_bert.input_formatter import TableTooLongError

def embed(values):
    table = Table(
            id='',
            header = [Column('', 'text'), ],
            data = values
    ).tokenize(model.tokenizer)

    context = 'Assign semantic types to columns'

    # model takes batched, tokenized inputs
    _, column_encoding, _ = model.encode(
        contexts=[model.tokenizer.tokenize(context)],
        tables=[table]
    )

    return column_encoding.detach().numpy()

import csv
import json
from pathlib import Path
from random import shuffle
import ast

from tqdm import tqdm

VIZNET = "/Users/moe/research/data-lake/artifacts/viznet-subset.parquet"
OUTPUT = "/Users/moe/research/data-lake/artifacts/tabert-viznet-embeddings.parquet"

results = []

X = pd.read_parquet(VIZNET)

err_count = 0

for row in tqdm(X.itertuples(), total=X.shape[0]):
    vals = ast.literal_eval(row.values)
    try:
        vector = embed(vals).squeeze().squeeze().tolist()
        results.append([row.Index, vector])
    except:
        err_count += 1
        if err_count % 50 == 0:
            tqdm.write(f"rows failed to embed: {err_count}")

print(f"rows failed to embed: {err_count}")
df = pd.DataFrame(results, columns=["id", "embedding"])
df.set_index('id', inplace=True)
df.to_parquet(OUTPUT)
