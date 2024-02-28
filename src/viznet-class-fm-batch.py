import pandas as pd

SUBSET = "artifacts/viznet-subset.parquet"
subset = pd.read_parquet(SUBSET)
# shuffle the data
subset = subset.sample(frac=1.0, random_state=0)

import os
import openai
from openai import OpenAI
from ratelimiter import RateLimiter

import pandas as pd
import numpy as np
from tqdm import tqdm

import ast
from random import seed
from time import sleep

np.random.seed(0)
seed(0)

openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

OUTPUT = "artifacts/FM-viznet-pred.parquet"

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
        results[x.index] = x.text.lower()
    
    return results

def make_msg(values):
    return f"""Consider the following list:
{values}

Which class best represents the list type? Reply with only one word from:
"publisher", "religion", "year", "industry", "city", "team", "address", "album", "country", "artist", "state", "isbn", "genre", "language"
"""

indices = []
types = []
messages = []

for row in tqdm(subset.itertuples(), total=subset.shape[0]):
    values = ast.literal_eval(row.values)
    # values = list(set(values))
    sample = np.random.choice(values, size=5, replace=False) if len(values) > 4 else values
    chatgpt_msg = make_msg(sample)
    indices.append(row.Index)
    types.append(row.type)
    messages.append(chatgpt_msg)

replies = [] * len(messages)

batch_size = 20
for i in tqdm(range(0, len(messages), batch_size)):
    batch = messages[i:i + batch_size]
    answers = response(batch, 25)
    replies[i:i + batch_size] = answers

results = []
for i, _ in enumerate(replies):
    results.append([indices[i], types[i], replies[i]])

X = pd.DataFrame(results, columns=["id", "label", "pred"])
X.set_index("id", inplace=True)
X.to_parquet(OUTPUT)
