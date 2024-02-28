import pandas as pd
import numpy as np

from tqdm import tqdm

df = pd.read_parquet("artifacts/t3-join-results-full.parquet")


def correct(notebook, str2, tru):
    # notebook = "datasets/merge_data_csv/" + notebook
    if str2 is None or not str2.all() or "None" in str2:
        return None
    elif str(tru) == str(str2):
        return True
    else:
        #str2 = eval(str2)
        # tru = eval(tru)

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
        try:
            first_join = pd.merge(left, right, left_on=tru[0], right_on=tru[1])
        except:
            return None

        if str2[0] == "index" or str2[0] == "Index":
            str2 = (left.index.name, str2[1])
        if str2[1] == "index" or str2[1] == "Index":
            str2 = (str2[0], right.index.name)
        try:
            second_join = pd.merge(left, right, left_on=str2[0], right_on=str2[1])
        except (KeyError, ValueError):
            print(f"keyerror: {str2}, cols1: {left.columns}")
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
for row in tqdm(df.itertuples()):
    notebook = row.notebook
    ours_true = correct(notebook, row.pred, row.true)
    jaccard_true = correct(notebook, row.jaccard, row.true)
    lev_true = correct(notebook, row.levenshtein, row.true)
    analysis.append([ours_true, jaccard_true, lev_true])

    try:
        left = pd.read_csv(
            notebook + "/left.csv",
            header=[0],
            index_col=0,
            dtype=str,
        )
        right = pd.read_csv(
            notebook + "/right.csv",
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
        "jaccard",
        "levenshtein",
    ],
)
for y in ["ours", "levenshtein", "jaccard"]:
    print(
        f"{y}, precision: {precision2(analysis[y]):0.1f}, recall: {recall2(analysis[y]):0.1f}"
    )

print("col sum ", sum(col_count))
print("row avg ", np.mean(row_count))
