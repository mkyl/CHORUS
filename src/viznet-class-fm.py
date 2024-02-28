import pandas as pd

SUBSET = "artifacts/viznet-subset.parquet"
subset = pd.read_parquet(SUBSET)
# shuffle the data
subset = subset.sample(frac=1.0, random_state=0)

import os
import openai
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

OUTPUT = "artifacts/FM-viznet-pred.parquet"

@RateLimiter(max_calls=1000, period=60)
def response(messages, n_tokens):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=n_tokens,
        temperature=0.1,
    )
    
    return completion.choices[0].message

def make_msg(values):
    return [
            {
                "role": "system",
                "content": "Be a helpful, accurate assistant for data discovery and exploration. Respond with one word.",
            },
            {
                "role": "user",
                "content": f"""Consider the following list:
{values}

Which class best represents the list type? Reply with only one word from:
"publisher", "religion", "year", "industry", "city", "team", "address", "album", "country", "artist", "state", "isbn", "genre", "language"
""",
            }
    ]

results = []

for row in tqdm(subset.itertuples(), total=subset.shape[0]):
    values = ast.literal_eval(row.values)
    # values = list(set(values))
    sample = np.random.choice(values, size=4, replace=False) if len(values) > 4 else values
    chatgpt_msg = make_msg(sample)
    answer = response(chatgpt_msg, 25).content.lower()

    results.append([row.Index, row.type, answer])

X = pd.DataFrame(results, columns=["id", "label", "pred"])
X.set_index("id", inplace=True)
X.to_parquet(OUTPUT)
