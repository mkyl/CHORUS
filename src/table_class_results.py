import pandas as pd
from src.nearest import embeds, nearest_class
import src.nearest as nearest
import numpy as np
import math

GOLD_STANDARD = "datasets/T2Dv2/extended_instance_goldstandard/classes_GS.csv"
CLASS_PREDS = "artifacts/class-predictions.csv"

def truncate_string(s):
    for i in range(len(s)):
        if not s[i].isalpha():
            return s[:i]
    return s


supervised_classes = [
    "AcademicJournal",
    "AdministrativeRegion",
    "Airline",
    "Airport",
    "Animal",
    "BaseballPlayer",
    "Bird",
    "Book",
    "Building",
    "City",
    "Company",
    "Country",
    "Currency",
    "Election",
    "FictionalCharacter",
    "Film",
    "GolfPlayer",
    "Hospital",
    "Hotel",
    "Lake",
    "Mammal",
    "Monarch",
    "Mountain",
    "Museum",
    "Newspaper",
    "Novel",
    "Person",
    "Plant",
    "PoliticalParty",
    "RadioStation",
    "Saint",
    "Scientist",
    "TelevisionShow",
    "University",
    "VideoGame",
    "Work",
    "Wrestler",
    "Cricketer",
    "Mountain",
    "Swimmer",
]
supervised_classes = ["http://dbpedia.org/ontology/" + x for x in supervised_classes]

relevant_embeds = embeds[embeds.class_name.isin(supervised_classes)]
# print(relevant_embeds)

anchor_count = 0

def process_class(url):
    global anchor_count
    try:
        last_slash_index = url.rfind("/")
        result = url[last_slash_index + 1 :]
        result = truncate_string(result)

        if "http://dbpedia.org/ontology/" + result not in supervised_classes:
            anchor_count += 1
            closest = (
                nearest_class(
                    relevant_embeds,
                    "http://dbpedia.org/ontology/" + result,
                    n=1,
                )
                .iloc[0]
                .class_name
            )
            last_slash_index = closest.rfind("/")
            result = closest[last_slash_index + 1 :]
            print(f"matched {url} to {result}")
        return "http://dbpedia.org/ontology/" + result
    except AttributeError as e:
        anchor_count += 1
        print(f"Error!! {e}")
        return np.nan


def extract_class(url):
    if url == "nan":
        return np.nan
    last_slash_index = url.rfind("/")
    result = url[last_slash_index + 1 :]
    return result

def calculate_results(prediction_csv, ablation=None):
    global anchor_count
    gt = pd.read_csv(
        GOLD_STANDARD, header=None, names=["dataset", "comment", "true_class"], dtype=str
    )
    ours = pd.read_csv(prediction_csv, header=None, names=["dataset", "pred_class"], dtype=str)

    results = pd.merge(gt, ours, left_on="dataset", right_on="dataset", how="inner")

    if ablation != "anchoring":
        results.pred_class = results.pred_class.apply(process_class)

    results.pred_class = results.pred_class.apply(lambda x: extract_class(str(x).lower()))
    results.true_class = results.true_class.apply(lambda x: extract_class(str(x).lower()))

    values, counts = np.unique(results.true_class, return_counts=True)
    index = np.isin(results.true_class, values[counts > 1])
    #results = results[index]

    results_2 = results.dropna()
    results.pred_class = results.pred_class.apply(lambda x: str(x).lower())

    print(f"results classes: {results.true_class.nunique()}")
    print(f"anchored {anchor_count} times")

    match = results.pred_class == results.true_class

    from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score

    r_score = recall_score(results.true_class, results.pred_class, average="weighted")
    p_score = precision_score(results_2.true_class, results_2.pred_class, average="weighted")
    a_score = accuracy_score(results.true_class, results.pred_class)
    f_score = math.sqrt(r_score * p_score)
    return (f_score, r_score, p_score)


if __name__ == "__main__":
    f_score, r_score, p_score = calculate_results(CLASS_PREDS)
    print(
        f"F1: {f_score:0.3f}, recall: {r_score:0.3f}, precision: {p_score:0.3f}"
    )

