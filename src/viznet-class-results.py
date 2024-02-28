import numpy as np
import pandas as pd

X = pd.read_parquet("artifacts/FM-viznet-pred.parquet")

def find_substring(list_of_strings, main_string):
    for s in list_of_strings:
        if s in main_string.lower():
            return s
    return "Unknown"

target_classes = ["publisher", "religion", "year", "industry", "city", "team", "address", "album", "country", "artist", "state", "isbn", "genre", "language", "industry"]

X.pred = X.pred.apply(lambda x: find_substring(target_classes, x))

from sklearn.metrics import classification_report
print(classification_report(X.label, X.pred, digits=3))
