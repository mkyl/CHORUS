from table_bert import TableBertModel
import pandas as pd

model = TableBertModel.from_pretrained(
    'tabert_large_k3/model.bin',
)

from table_bert import Table, Column
from table_bert.input_formatter import TableTooLongError

def embed(df):
    table = Table(
            id='',
            header = [Column(x, 'text') for x in df.columns],
            data = df.head(3).values.tolist()
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

from tqdm import tqdm

TABLE_PREFIX = Path("/Users/moe/research/Datasets/T2Dv2/extended_instance_goldstandard/tables/")
OUTPUT = "tabert-embeddings.csv"

json_files = [x for x in TABLE_PREFIX.glob("*.json") if x.stat().st_size > 0]

results = []

for t2d_table in tqdm(json_files):
    try:
        with open(t2d_table, "r", encoding="utf-8") as json_file:
            instance = json.load(json_file)
    except UnicodeDecodeError:
        with open(t2d_table, "r", encoding="windows-1252") as json_file:
            instance = json.load(json_file)

    table = instance["relation"]
    title = instance["pageTitle"]

    if "tableOrientation" in instance and instance["tableOrientation"] == "HORIZONTAL":
        # pivot the table
        table = list(zip(*table))

    if len(table) > 4:
        table = table[:4]

    df = pd.DataFrame(table[1:], columns=table[0])
    try:
        vector = embed(df).squeeze()
    except TableTooLongError:
        vector = [None for x in df.columns]

    for i, col in enumerate(vector):
        try:
            results.append([t2d_table.name, i, json.dumps(col.tolist())])
        except AttributeError:
            results.append([t2d_table.name, i, json.dumps(col)])

pd.DataFrame(results, columns=["table", "column", "embedding"]).to_csv(OUTPUT, index=False)
