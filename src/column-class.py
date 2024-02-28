import csv
import json
from pathlib import Path
from random import shuffle, seed

from tqdm import tqdm

import os
import openai
from openai.error import InvalidRequestError
from ratelimiter import RateLimiter

import pandas as pd

seed(0)

openai.api_key = os.getenv("OPENAI_API_KEY")

COL_PREFIX = Path("datasets/T2Dv2/extended_instance_goldstandard/property/")
TABLE_PREFIX = Path("datasets/T2Dv2/extended_instance_goldstandard/tables/")
OUTPUT = "artifacts/FM-col-pred.csv"

@RateLimiter(max_calls=20, period=60)
def response(messages, n_tokens):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=n_tokens,
    )

    return completion.choices[0].message


table_names = [x for x in COL_PREFIX.glob("*.csv") if x.stat().st_size > 0]
shuffle(table_names)

try:
    df = pd.read_csv(OUTPUT, names=["Table", "Col", "Pred"])
    last_table = df.iloc[-1].Table
    i = list(map(lambda x: x.name, table_names)).index(last_table.replace(".json", ".csv"))
    print(f"continuing from table #{i}")
    table_names = table_names[i:] 
except:
    pass

results = []

for full_path in tqdm(table_names):
    full_path = TABLE_PREFIX / full_path.name.replace(".csv", ".json")
    try:
        with open(full_path, "r", encoding="utf-8") as json_file:
            instance = json.load(json_file)
    except UnicodeDecodeError:
        with open(full_path, "r", encoding="windows-1252") as json_file:
            instance = json.load(json_file)

    table = instance["relation"]
    title = instance["pageTitle"]

    if "tableOrientation" in instance and instance["tableOrientation"] == "HORIZONTAL":
        # pivot the table
        table = list(zip(*table))

    if len(table) > 4:
        table = table[:4]

    df = pd.DataFrame(table[1:], columns=table[0], index=None)
    if len(df.columns) > 20:
        # avoid corner cases
        continue 

    table = map(lambda x: ", ".join(x), table)
    CSV_like = ",\n".join(table)

    all_preds = []

    messages=[
        {
            "role": "system",
            "content": "Be a helpful, accurate assistant for data discovery and exploration.",
        },
        {
            "role": "user",
            "content": f"""Consider this table:
```
{CSV_like}
```
Consider only these DBPedia.org properties: releaseDate, director, populationTotal, capital, country, mayor, populationMetro, location, city, elevation, industry, publisher, currencyCode, iataLocationIdentifier, areaTotal, currency, language, usingCountry, capitalCoordinates, numberOfVisitors, frequency, revenue, locatedInArea, formerName, PopulatedPlace, developer, owner, iso, grossDomesticProduct, governmentType.

Which of those DBPedia.org properties most closely represents the first column? You can reply with `Unknown` if not possible. Otherwise start your reply with `http://dbpedia.org/ontology/`.
""",
        }
    ]

    for i, col in enumerate(df.columns):
        try:
            chatgpt_msg = response(messages, 15)
            prediction = chatgpt_msg.content
        except InvalidRequestError:
            prediction = "null"
        all_preds.append(prediction)
        messages.append(dict(chatgpt_msg))
        messages.append(
            { "role": "user",
            "content": f"Which of those DBPedia.org properties most closely represents column {i+2}? Start your reply with `http://dbpedia.org/ontology/`"
        })

    #print(df)
    #print(messages)

    for i, p in enumerate(all_preds):
        results.append([full_path.name, i, p])

    with open(OUTPUT, "a", newline="", encoding="UTF-8") as output:
        writer = csv.writer(output)
        for i, p in enumerate(all_preds):
            writer.writerow([full_path.name, i, p])
