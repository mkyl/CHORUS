from ratelimiter import RateLimiter
from tqdm import tqdm
import pandas as pd
import csv

# Define your XML document as a string
classes = pd.read_csv("artifacts/dbpedia_class_names.csv", header=[0], squeeze=True)

import openai


@RateLimiter(max_calls=59, period=60)
def get_embedding(str):
    return openai.Embedding.create(input=str, model="text-embedding-ada-002")["data"][
        0
    ]["embedding"]


with open("dbpedia-class-embeds.csv", "w", newline="") as file:
    writer = csv.writer(file)

    # Write each string as a row in the CSV file
    for string in tqdm(classes):
        writer.writerow([string, get_embedding(string)])

print(f"wrote {len(classes)} dbpedia classes and their embeddings")
