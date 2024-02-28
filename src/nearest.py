import numpy as np
import pandas as pd
import ast

import openai


def cosine_similarity(a, b):
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)


def convert_array(string):
    return ast.literal_eval(string)


embeds = pd.read_csv(
    "artifacts/dbpedia-class-embeds.csv",
    header=None,
    names=["class_name", "embedding"],
    converters={"embedding": convert_array},
)


def nearest_class(df, str, n=3):
    pred_embedding = openai.Embedding.create(input=str, model="text-embedding-ada-002")[
        "data"
    ][0]["embedding"]
    df["similarities"] = df.embedding.apply(
        lambda x: cosine_similarity(x, pred_embedding)
    )
    res = df.sort_values("similarities", ascending=False).head(n)
    return res


# x = nearest_class(embeds, "http://dbpedia.org/ontology/Journal", n=5)
# print(x)
