import os
import json
import re
import random

import pandas as pd
import numpy as np
from tqdm import tqdm
from fuzzywuzzy import process

import os
import openai
from openai import OpenAI, BadRequestError
from ratelimiter import RateLimiter

import pandas as pd
import numpy as np
from tqdm import tqdm

import ast
from random import seed

from main import join_column_prompt, response

OUTPUT = "artifacts/t3-join-results-full.parquet"

N = 5
random.seed(0)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

def load_files(directory):
    try:
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
    except:
        raise pd.errors.ParserError


# Define a custom formatter function
def custom_formatter(value):
    if isinstance(value, str):
        return f"{value[:25] if len(value) > 25 else value}"
    elif isinstance(value, float):
        return f"{value:.3g}"
    else:
        return value


def predict_cols_prompt(table1, table2):
    table1_csv = table1.applymap(custom_formatter).head(n=N).to_string(max_colwidth=30, max_cols=25)
    table2_csv = table2.applymap(custom_formatter).head(n=N).to_string(max_colwidth=30, max_cols=25)
    prompt = join_column_prompt("CSV", table1_csv, table2_csv)
    return prompt


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
    if "right_on= '" in s:
        start = s.find("'") + 1
        end = s.find("'", start)
        col1 = s[start:end]
        start = s.find("right_on= '") + len("right_on= '")
        end = s.find("'", start)
        col2 = s[start:end]
    elif 'right_on= "' in s:
        start = s.find('"') + 1
        end = s.find('"', start)
        col1 = s[start:end]
        start = s.find('right_on= "') + len('right_on= "')
        end = s.find('"', start)
        col2 = s[start:end]
    else:
        col1 = col2 = None
    return col1, col2


@RateLimiter(max_calls=2, period=1)
def response(messages, n_tokens):
    completion = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=messages,
        max_tokens=n_tokens,
        temperature=0.1,
    )

    results = [""] * len(messages)
    
    for x in completion.choices:
        results[x.index] = x.text
    
    return results

def response_batch(prompts):
    replies = [] * len(prompts)
    batch_size = 20
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i + batch_size]
        try:
            answers = response(batch, 25)
        except BadRequestError:
            # make prompts shorter to fit in context
            tqdm.write("Error in batch, retrying!")
            batch = [x[:512] for x in batch]
            answers = response(batch, 25)
        replies[i:i + batch_size] = answers
    return replies


def suggest_levenshtein(df1, df2):
    suggestions = None
    best = 0.0
    # doesn't scale beyond 25x25
    for col1 in df1.columns[1:25]:
        best_match, best_score = process.extractOne(col1, df2.columns[1:25])
        if best_score > best:
            suggestions = (col1, best_match)
            best = best_score
    return suggestions


def suggest_jaccard(df1, df2):
    max_overlap = 0
    max_overlap_column = None

    # doesn't scale beyond 25x25
    for col1 in df1.columns[1:25]:
        for col2 in df2.columns[1:25]:
            overlap = len(set(df1[col1].unique()) & set(df2[col2].unique()))
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_column = (col1, col2)

    return max_overlap_column


def main():
    results = []
    base_directory = "datasets/merge_data_csv/"
    subdirectories = [
        os.path.join(base_directory, d)
        for d in os.listdir(base_directory)
        if os.path.isdir(os.path.join(base_directory, d))
    ]
    random.shuffle(subdirectories)
    for subdir in tqdm(subdirectories):
        try:
            left_df, right_df, params = load_files(subdir)
        except pd.errors.ParserError:
            print("parsing CSV error")
            continue

        pred = predict_cols_prompt(left_df, right_df)
        try:
            lev = suggest_levenshtein(left_df, right_df)
        except (ValueError, TypeError):
            lev = None
        try:
            jaccard = suggest_jaccard(left_df, right_df)
        except TypeError:
            jaccard = None

        results.append(
            [(params["left on"], params["right on"]), pred, lev, jaccard, subdir]
        )
    
    results = pd.DataFrame(
        results, columns=["true", "prompt", "levenshtein", "jaccard", "notebook"]
    )

    results["answer"] = response_batch(results.prompt.to_list())
    results["pred"] = results["answer"].apply(extract_columns)

    results.drop("prompt", axis="columns", inplace=True)
    results.drop("answer", axis="columns", inplace=True)

    results.to_parquet(OUTPUT)

    return results


if __name__ == "__main__":
    results = main()

