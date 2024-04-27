import argparse
import pandas as pd
from doduo import Doduo

OUTPUT = "embed-t2d-doduo-viz.csv"

# Load Doduo model
args = argparse.Namespace
args.model = "viznet" # or args.model = "viznet"
doduo = Doduo(args)

from tqdm import tqdm
import csv
import json

GOLD_STANDARD = (
    "/Users/moe/research/Datasets/T2Dv2/extended_instance_goldstandard/classes_GS.csv"
)
TABLE_PREFIX = (
    "/Users/moe/research/Datasets/T2Dv2/extended_instance_goldstandard/tables/"
)

data = []
with open(GOLD_STANDARD, "r") as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
        data.append(row)

files = None

for row in tqdm(data):
    name = row[0].replace(".tar.gz", ".json")
    dbpedia_class = row[1]

    filename = TABLE_PREFIX + name
    try:
        with open(filename, "r", encoding="utf-8") as json_file:
            instance = json.load(json_file)
    except UnicodeDecodeError:
        with open(filename, "r", encoding="windows-1252") as json_file:
            instance = json.load(json_file)

    data = instance["relation"]
    if "tableOrientation" in instance and instance["tableOrientation"] == "HORIZONTAL":
        # pivot the table
        data = list(zip(*data))

    
    df = pd.DataFrame(data[1:], columns=data[0])

    try:
        annot_df1 = doduo.annotate_columns(df)
    except RuntimeError:
        print("embedding error")
        continue
    embedding = sum(annot_df1.colemb)

    with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
        writer = csv.writer(output)
        writer.writerow([row[0], dbpedia_class, embedding.tolist()])
