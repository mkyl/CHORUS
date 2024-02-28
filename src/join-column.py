import os
import json
import re
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import InvalidRequestError, APIError
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from main import join_column_prompt, response

N = 5
random.seed(0)


def load_files(directory):
    left_path = os.path.join(directory, "left.csv")
    right_path = os.path.join(directory, "right.csv")
    param_path = os.path.join(directory, "param.json")

    left_df = pd.read_csv(left_path, index_col=0) if os.path.exists(left_path) else None
    right_df = (
        pd.read_csv(right_path, index_col=0) if os.path.exists(right_path) else None
    )
    left_df = left_df.reset_index(drop=False)
    right_df = right_df.reset_index(drop=False)

    with open(param_path, "r") as param_file:
        params = json.load(param_file)

    return left_df, right_df, params


# Define a custom formatter function
def custom_formatter(value):
    if isinstance(value, str):
        return f"{value[:25] if len(value) > 25 else value}"
    elif isinstance(value, float):
        return f"{value:.3g}"
    else:
        return value


def predict_cols(table1, table2):
    table1_csv = table1.applymap(custom_formatter).head(n=N).to_string(max_colwidth=30)
    table2_csv = table2.applymap(custom_formatter).head(n=N).to_string(max_colwidth=30)
    prompt = join_column_prompt("CSV", table1_csv, table2_csv)
    x = response(prompt, 25)
    # return extract_column(x)
    return extract_columns(x)


def extract_fields(input_string):
    # Split the input string using ',' as the delimiter
    parts = input_string.split(", ")

    # Extract the field names from the parts
    field_names = []
    for part in parts:
        field_name = part.split(": ")[1]
        field_names.append(field_name)

    return tuple(field_names)


def extract_column(s):
    result = re.search("`([^`]*)`", s)
    return result.group(1) if result else s


def extract_columns(s):
    if "right_on='" in s:
        start = s.find("'") + 1
        end = s.find("'", start)
        col1 = s[start:end]
        start = s.find("right_on='") + len("right_on='")
        end = s.find("'", start)
        col2 = s[start:end]
    elif 'right_on="' in s:
        start = s.find('"') + 1
        end = s.find('"', start)
        col1 = s[start:end]
        start = s.find('right_on="') + len('right_on="')
        end = s.find('"', start)
        col2 = s[start:end]
    else:
        col1 = col2 = None
    return col1, col2


def suggest_levenshtein(df1, df2):
    suggestions = None
    best = 0.0
    for col1 in df1.columns[1:]:
        best_match, best_score = process.extractOne(col1, df2.columns[1:])
        if best_score > best:
            suggestions = (col1, best_match)
            best = best_score
    return suggestions


def suggest_jaccard(df1, df2):
    max_overlap = 0
    max_overlap_column = None

    for col1 in df1.columns[1:]:
        for col2 in df2.columns[1:]:
            overlap = len(set(df1[col1].unique()) & set(df2[col2].unique()))
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_column = (col1, col2)

    return max_overlap_column


def main():
    base_directory = "datasets/github_notebooks/"
    subdirectories = [
        os.path.join(base_directory, d)
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]
    random.shuffle(subdirectories)
    for subdir in tqdm(subdirectories[:155]):
        try:
            left_df, right_df, params = load_files(subdir)
        except pd.errors.ParserError:
            print("parsing CSV error")
            continue

        if (
            False
            # params["how"] != "inner"
            # or params["left on"] == "index"
            # or params["right on"] != params["left on"]
            # or params["right on"] == "index"
        ):
            continue

        try:
            pred = predict_cols(left_df, right_df)
            try:
                lev = suggest_levenshtein(left_df, right_df)
            except (ValueError, TypeError):
                lev = None
            try:
                jaccard = suggest_jaccard(left_df, right_df)
            except TypeError:
                jaccard = None
        except InvalidRequestError:
            print("error in request")
            continue
        except APIError:
            # try again
            pred = predict_cols(left_df, right_df)
        results.append(
            [(params["left on"], params["right on"]), pred, lev, jaccard, subdir]
        )


results = []
if __name__ == "__main__":
    main()
results = pd.DataFrame(
    results, columns=["true", "pred", "levenshtein", "jaccard", "notebook"]
)
