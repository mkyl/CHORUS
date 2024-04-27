import pandas as pd

from pathlib import Path
from random import shuffle

from tqdm import tqdm

VIZNET = "viznet-subset.parquet"
OUTPUT = "sherlock-viznet-probs.parquet"

results = []

X = pd.read_parquet(VIZNET)

import os
os.environ["PYTHONHASHSEED"] = "0"

import numpy as np
import pandas as pd
import pyarrow as pa

from sherlock import helpers
from sherlock.deploy.model import SherlockModel
from sherlock.functional import extract_features_to_csv
from sherlock.features.paragraph_vectors import initialise_pretrained_model, initialise_nltk
from sherlock.features.preprocessing import (
    extract_features,
    convert_string_lists_to_lists,
    prepare_feature_extraction,
    load_parquet_values,
)
from sherlock.features.word_embeddings import initialise_word_embeddings

prepare_feature_extraction()
initialise_word_embeddings()
initialise_pretrained_model(400)
initialise_nltk()

def sherlock_predict(data):
    extract_features(
        "../temporary.csv",
        data
    )
    feature_vectors = pd.read_csv("../temporary.csv", dtype=np.float32)
    model = SherlockModel();
    model.initialize_model_from_json(with_weights=True, model_id="sherlock");
    predicted_labels = model.predict_proba(feature_vectors, "sherlock")
    return predicted_labels

print("best class should be 0:", pd.Series(sherlock_predict([["Chabot Street 19", "1200 fifth Avenue", "Binnenkant 22, 1011BH"]]).squeeze().squeeze().tolist()).argmax())

import ast

def deserialize(x):
    arr = ast.literal_eval(x)
    arr = map(str, arr)
    return list(arr)

m = map(deserialize, X["values"].values)
results = sherlock_predict(m)

df = pd.DataFrame(results, columns=['address', 'affiliate', 'affiliation', 'age', 'album', 'area',
       'artist', 'birth Date', 'birth Place', 'brand', 'capacity',
       'category', 'city', 'class', 'classification', 'club', 'code',
       'collection', 'command', 'company', 'component', 'continent',
       'country', 'county', 'creator', 'credit', 'currency', 'day',
       'depth', 'description', 'director', 'duration', 'education',
       'elevation', 'family', 'file Size', 'format', 'gender', 'genre',
       'grades', 'industry', 'isbn', 'jockey', 'language', 'location',
       'manufacturer', 'name', 'nationality', 'notes', 'operator',
       'order', 'organisation', 'origin', 'owner', 'person', 'plays',
       'position', 'product', 'publisher', 'range', 'rank', 'ranking',
       'region', 'religion', 'requirement', 'result', 'sales', 'service',
       'sex', 'species', 'state', 'status', 'symbol', 'team', 'team Name',
       'type', 'weight', 'year'])
df["id"] = X.index
df.set_index('id', inplace=True)
target_classes = ["publisher", "religion", "year", "industry", "city", "team", "address", "album", "country", "artist", "state", "isbn", "genre", "language"]
df = df[target_classes]
print(df)
df.to_parquet(OUTPUT)
