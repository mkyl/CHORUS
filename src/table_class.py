import csv
import json

from tqdm import tqdm

import src.main as main

OUTPUT = "artifacts/class-predictions.csv"


def table_class_experiment(result_file, temperature=1.0, model="gpt-3.5-turbo", ablation=None):
    GOLD_STANDARD = "datasets/T2Dv2/extended_instance_goldstandard/classes_GS.csv"
    TABLE_PREFIX = "datasets/T2Dv2/extended_instance_goldstandard/tables/"

    with open(GOLD_STANDARD, "r") as csvfile:
        csvreader = csv.reader(csvfile)
        data = []
        for row in csvreader:
            data.append(row)

    results = []

    for row in tqdm(data):
        name = row[0].replace(".tar.gz", ".json")
        dbpedia_class = row[1]
        dbpedia_url = row[2]

        filename = TABLE_PREFIX + name
        try:
            with open(filename, "r", encoding="utf-8") as json_file:
                instance = json.load(json_file)
        except UnicodeDecodeError:
            with open(filename, "r", encoding="windows-1252") as json_file:
                instance = json.load(json_file)

        table = instance["relation"]
        title = instance["pageTitle"]

        if "tableOrientation" in instance and instance["tableOrientation"] == "HORIZONTAL":
            # pivot the table
            table = list(zip(*table))

        if len(table) > 5:
            table = table[:5]

        if ablation == "metadata" or ablation == "prefixes" or ablation == "anchoring":
            table = map(lambda x: ", ".join(x), table[1:])
        else:
            table = map(lambda x: ", ".join(x), table)

        CSV_like = ",\n".join(table)

        if ablation == "demonstration" or ablation == "metadata":
                prompt = main.dataset_class_prompt_no_example("CSV", CSV_like)
        elif ablation == "prefixes":
            prompt = main.dataset_class_prompt_no_prefix("CSV", CSV_like)
        else:
            prompt = main.dataset_class_prompt("CSV", CSV_like)

        try:
            #prediction = main.response(prompt, 18, temperature=temperature, model=model)
            prediction = main.response(prompt, 35, temperature=temperature, model=model)
        except:
            prediction = "null"
        prediction = prediction.replace(" ", "").replace("\n", "").replace("\t", "")
        results.append([row[0], prediction])

        with open(result_file, "a", newline="", encoding="UTF-8") as output:
            writer = csv.writer(output)
            writer.writerow([row[0], prediction])

    with open(result_file, "w", newline="", encoding="UTF-8") as output:
        writer = csv.writer(output)
        for row in results:
            writer.writerow(row)


if __name__ == "__main__":
    table_class_experiment(OUTPUT)
