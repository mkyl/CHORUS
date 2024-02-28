import os
from pathlib import Path
import json
import openai
from ratelimiter import RateLimiter

openai.api_key = os.getenv("OPENAI_API_KEY")

DATA_LAKE = "ingest"
OUTPUT = "metadata.json"


def summary_prompt(extension, data):
    return f"For the following {extension} sample, describe in three generic sentences the dataset it comes from.\n```\n{data}\n```"


def fields_prompt(extension, data):
    return f"In the following {extension} sample, for exactly one row, explain each of its fields in the format (<example value> : <field name>). Produce tuples seperated by newlines.\n```\n{data}\n```"


def title_prompt(extension, data):
    return f"For the following {extension} sample, suggest the title of the dataset it is from. Reply with just the title.\n```\n{data}\n```"


def dataset_class_prompt(extension, data):
    return (
        f"For the following {extension} sample, select one DBpedia.org ontology that represents the dataset from the following list: "
        + "AcademicJournal, AdministrativeRegion, Airline, Airport, Animal, BaseballPlayer, Bird,"
        + "Book, Building, City, Company, Country, Cricketer, Currency, Election, FictionalCharacter, Film,"
        + "GolfPlayer, Hospital, Hotel, Lake, Mammal, Monarch, Mountain, Museum, Newspaper, Novel,"
        + "Person, Plant, PoliticalParty, RadioStation,"
        + "Saint, Scientist, Swimmer, TelevisionShow, University, VideoGame, Work,"
        + "Wrestler.\n"
        + "For example, for a dataset about hospitals, return "
        + f"'https://dbpedia.org/ontology/Hospital'. Begin your answer with 'https://dbpedia.org/ontology'.\n```\n{data}\n```\n'https://dbpedia.org/"
    )

def dataset_class_prompt_no_example(extension, data):
    return (
        f"For the following {extension} sample, select one DBpedia.org ontology that represents the dataset from the following list: "
        + "AcademicJournal, AdministrativeRegion, Airline, Airport, Animal, BaseballPlayer, Bird,"
        + "Book, Building, City, Company, Country, Cricketer, Currency, Election, FictionalCharacter, Film,"
        + "GolfPlayer, Hospital, Hotel, Lake, Mammal, Monarch, Mountain, Museum, Newspaper, Novel,"
        + "Person, Plant, PoliticalParty, RadioStation,"
        + "Saint, Scientist, Swimmer, TelevisionShow, University, VideoGame, Work,"
        + "Wrestler.\n"
        + f"Begin your answer with 'https://dbpedia.org/ontology'.\n```\n{data}\n```"
    )

def dataset_class_prompt_no_prefix(extension, data):
    return (
        f"For the following {extension} sample, select one DBpedia.org ontology that represents the dataset from the following list: "
        + "AcademicJournal, AdministrativeRegion, Airline, Airport, Animal, BaseballPlayer, Bird,"
        + "Book, Building, City, Company, Country, Cricketer, Currency, Election, FictionalCharacter, Film,"
        + "GolfPlayer, Hospital, Hotel, Lake, Mammal, Monarch, Mountain, Museum, Newspaper, Novel,"
        + "Person, Plant, PoliticalParty, RadioStation,"
        + "Saint, Scientist, Swimmer, TelevisionShow, University, VideoGame, Work,"
        + "Wrestler.\n"
        + f"```\n{data}\n```"
    )


def dataset_class_prompt_free(extension, data):
    return f"For the following {extension} sample, select one DBpedia.org ontology that represents the dataset. For example, for a dataset about hospitals, return 'https://dbpedia.org/ontology/Hospital'. Begin your answer with 'https://dbpedia.org/ontology'.\n```\n{data}\n```"


def column_type_prompt(extension, data):
    return f"""
        For the following {extension} sample, suggest a DBPedia.org Property for each column from the `http://dbpedia.org/ontology/` namespace. For example, given the following data:
    ```
    Name, Famous Book, Rk, Year
    Fyodor Dostoevsky, Crime and Punishment, 22.5, 1866
    Mark Twain, Adventures of Huckleberry Finn, 53, 1884
    Albert Camus, The Stranger, -23, 1942
    ```
    Return `dbo:author, dbo:title, Unknown, dbo:releaseDate`

    ```
    {data}
    ```"""


def column_type_prompt_2(data):
    return f"""
Consider this example. Input:
```
Name, Famous Book, Rk, Year
Fyodor Dostoevsky, Crime and Punishment, 22.5, 1866
Mark Twain, Adventures of Huckleberry Finn, 53, 1884
Albert Camus, The Stranger, -23, 1942
```
Output: `dbo:author, dbo:title, Unknown, dbo:releaseDate`.

For the following CSV sample, suggest a DBPedia.org Property for each column from the `dbo:` namespace.
```
{data}
```"""


def column_type_prompt_3(data):
    return f"""
For the following CSV sample, suggest a DBPedia.org Property for each column from the `dbo:` namespace. Use a format like  `dbo:author, dbo:title, Unknown, dbo:releaseDate`.
```
{data}
```"""


# df1 =
# ```
#             author  bestsellers_date          title   rank_on_list
# 0    Dean R Koontz        2008-05-24      ODD HOURS              1
# 1  Stephenie Meyer        2008-05-24       THE HOST              2
# 2     Emily Giffin        2008-05-24   LOVE THE ONE              3
# ```
# df2 =
# ```
#       title_on_list   weeks_on_list
# 0  10TH ANNIVERSARY               8
# 1          11/22/63               9
# 2         11TH HOUR               8
# ```
# Return `pd.merge(df1, df2, left_on="title", right_on="title_on_list")`.


def join_column_prompt(extension, table1, table2):
    return f"""Given two Pandas Dataframes, suggest what `pd.merge` parameters to use to join the dataframes. 

df1 = 
```
{table1}
```

df2 = 
```
{table2}
```
Complete the correct Pandas merge command. `pd.merge(df1, df2, left_on="""

import time
import json
import requests
API_TOKEN = "your api token"
headers = {"Authorization": f"Bearer {API_TOKEN}","Content-Type": "application/json"}
API_URL = "your URL ending in .endpoints.huggingface.cloud"

##see https://huggingface.co/docs/api-inference/detailed_parameters
## for adding parameters under text_generation
model_parameters = {"max_new_tokens": 35,"temperature":0.2}

def query_llama2(model_input):
    input_payload = {"inputs":model_input,"parameters":model_parameters}
    data = json.dumps(input_payload)
    response = requests.request("POST", API_URL, headers=headers, data=data)
    result = json.loads(response.content.decode("utf-8"))
    return result[0]["generated_text"]

@RateLimiter(max_calls=20, period=60)
def response(prompt, n_tokens, temperature=1.0, model="gpt-3.5-turbo"):
    if model == "llama2":
        return "https://dbpedia.org/" + query_llama2(prompt)

    completion = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "Be a helpful, accurate assistant for data discovery and exploration.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        max_tokens=n_tokens,
        temperature=temperature,
    )

    return completion.choices[0].message.content


def build_metadata(files):
    datasets = []
    for path, extension in files:
        file_path = Path(path)
        snippet = file_path.read_text()

        # cols = [
        #     "placeholder example : placeholder description"
        # ]  
        cols = response(fields_prompt(extension, snippet), 320, temperature=0.1, model="gpt-4").split("\n")
        fields = []
        for c in cols:
            if ":" in c:
                fields.append(
                    {
                        "description": c.split(":")[0],
                        "value": c.split(":")[1],
                    }
                )

        row = {
            #"title": str(path),
            "title": response(title_prompt(extension, snippet), 25, temperature=0.1, model="gpt-4"),
            "summary": response(summary_prompt(extension, snippet), 128, temperature=0.1, model="gpt-4"),
            "class": response(dataset_class_prompt(extension, snippet), 18, temperature=0.1, model="gpt-4"),
            "fields": fields,
        }
        datasets.append(row)

    return datasets


def main():
    data_lake = Path(DATA_LAKE)
    files = [(x, x.suffix.upper()) for x in data_lake.iterdir() if x.name[0] != "."]
    datasets = build_metadata(files)

    with open(OUTPUT, "w", encoding="UTF-8") as json_file:
        json.dump(datasets, json_file, ensure_ascii=False)


if __name__ == "__main__":
    main()
