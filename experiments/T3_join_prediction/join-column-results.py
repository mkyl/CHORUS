import pandas as pd
import numpy as np

# read the csv file into a dataframe
df = pd.read_csv(
    "results/join-column-prediction-results.csv",
    header=[0],
    dtype={
        "pred": "object",
        "levenshtein": "object",
        "jaccard": "object",
        "trifacta": "object",
    },
)


def correct(notebook, str2, tru):
    notebook = "datasets/github_notebooks/" + notebook
    if not str2 or not pd.notna(str2) or "None" in str2:
        return None
    elif tru == str2:
        return True
    else:
        str2 = eval(str2)
        tru = eval(tru)

        left = pd.read_csv(notebook + "/left.csv", header=[0], index_col=0, dtype=str)
        right = pd.read_csv(notebook + "/right.csv", header=[0], index_col=0, dtype=str)

        if left.index.name == None:
            left.index.name = "index"

        if right.index.name == None:
            right.index.name = "index"

        left.index = left.index.map(str)
        right.index = right.index.map(str)

        if tru[0] == "index":
            tru = (left.index.name, tru[1])
        if tru[1] == "index":
            tru = (tru[0], right.index.name)
        first_join = pd.merge(left, right, left_on=tru[0], right_on=tru[1])

        if str2[0] == "index" or str2[0] == "Index":
            str2 = (left.index.name, str2[1])
        if str2[1] == "index" or str2[1] == "Index":
            str2 = (str2[0], right.index.name)
        try:
            second_join = pd.merge(left, right, left_on=str2[0], right_on=str2[1])
        except KeyError:
            print(f"keyerror: {str2}")
            return False

        x = first_join.sort_index().sort_index(axis=1)
        y = second_join.sort_index().sort_index(axis=1)
        return x.shape == y.shape


def precision2(results):
    return (
        (results == True).sum() / ((results == True) | (results == False)).sum() * 100
    )


def recall2(results):
    return (results == True).sum() / results.shape[0] * 100


col_count = []
row_count = []

analysis = []
for row in df.itertuples():
    notebook = row.notebook
    ours_true = correct(notebook, row.pred, row.true)
    trifacta_true = correct(notebook, row.trifacta, row.true)
    jaccard_true = correct(notebook, row.jaccard, row.true)
    lev_true = correct(notebook, row.levenshtein, row.true)
    analysis.append([ours_true, trifacta_true, jaccard_true, lev_true])

    try:
        left = pd.read_csv(
            "datasets/github_notebooks/" + notebook + "/left.csv",
            header=[0],
            index_col=0,
            dtype=str,
        )
        right = pd.read_csv(
            "datasets/github_notebooks/" + notebook + "/right.csv",
            header=[0],
            index_col=0,
            dtype=str,
        )
        col_count.append(left.shape[1])
        col_count.append(right.shape[1])
        row_count.append(left.shape[0])
        row_count.append(left.shape[0])
    except:
        pass

analysis = pd.DataFrame(
    analysis,
    columns=[
        "ours",
        "trifacta",
        "jaccard",
        "levenshtein",
    ],
)

for y in ["ours", "levenshtein", "jaccard", "trifacta"]:
    print(
        f"{y}, precision: {precision2(analysis[y]):0.1f}, recall: {recall2(analysis[y]):0.1f}"
    )

print("col sum ", sum(col_count))
print("row avg ", np.mean(row_count))
