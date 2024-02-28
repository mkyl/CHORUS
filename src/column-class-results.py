import re
from pathlib import Path

import pandas as pd

TRUTH = Path("datasets/T2Dv2/extended_instance_goldstandard/property")
PREDICTIONS = "artifacts/FM-col-pred.csv"


def extract_alphanumeric(input_string):
    rightmost_part = str(input_string).split("http://dbpedia.org/ontology/")[-1]
    match = re.match("^[a-zA-Z]+", rightmost_part)
    result = match.group(0) if match else "Unknown"
    return result

predictions = pd.read_csv(PREDICTIONS, header=None, names=["Dataset", "Column", "Prediction"])
predictions["Prediction"] = predictions["Prediction"].apply(extract_alphanumeric)

# print(f'declined predicitions: {sum(predictions["Prediction"] != "Unknown")}')
#predictions = predictions[predictions["Prediction"] != "Unknown"]
predictions = predictions[predictions["Prediction"].isin("releaseDate, populationTotal, location, elevation, capital, industry, director, country, computingPlatform, genre, populationMetro, developer, mayor, publisher, city, currencyCode, iataLocationIdentifier, governmentType, usingCountry, locatedInArea, year, grossDomesticProduct, language, currency, areaTotal, synonym, formerName, area, iso31661Code, assets, sales, mountainRange, foundingYear, symbol, circle, class, revenue, subdivision, owner, frequency, team, family, author, genus, iataAirlineCode, conservationStatus, firstAscentYear, collectionSize, programmeFormat, capitalCoordinates, floorCount, type, continent, region, child, alias, category, numberOfVisitors".split(", "))]

ground_truth = []

for filename in predictions.Dataset.unique():
    dataset_labels = pd.read_csv(
        TRUTH / filename.replace(".json", ".csv"),
        header=None,
        names=["url", "header", "unknown", "index"],
    )
    for row in dataset_labels.itertuples():
        if row.url == 'http://www.w3.org/2000/01/rdf-schema#label':
            continue
        property = extract_alphanumeric(row.url)
        ground_truth.append([filename, row.index, property])


ground_truth = pd.DataFrame(ground_truth, columns=["Dataset", "Column", "Label"])

merged_df = pd.merge(ground_truth, predictions, on=["Dataset", "Column"], how="inner")

from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, balanced_accuracy_score
print("\nClassification Report:\n", classification_report(merged_df['Label'], merged_df['Prediction'], zero_division=1))

weighted_f1 = f1_score(merged_df['Label'], merged_df['Prediction'], average='weighted', zero_division=1)
weighted_precision = precision_score(merged_df['Label'], merged_df['Prediction'], average='weighted',  zero_division=1)
weighted_recall = recall_score(merged_df['Label'], merged_df['Prediction'], average='weighted',  zero_division=1)
bas = balanced_accuracy_score(merged_df['Label'], merged_df['Prediction'])


# Print the results
print(f"Weighted F1: {weighted_f1:0.3f}")
print(f"Weighted Precision: {weighted_precision:0.3f}")
print(f"Weighted Recall: {weighted_recall:0.3f}")
print("bas", bas)
