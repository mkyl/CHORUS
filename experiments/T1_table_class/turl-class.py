import ast

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score

# Load data from csv file
data = pd.read_csv(
    "embed-t2d-turl.csv", header=None, names=["file", "class", "embedding"]
)

data["embedding"] = data["embedding"].apply(ast.literal_eval)

# Split data into features (embeddings) and labels (classes)
X = np.array([np.array(embedding) for embedding in data["embedding"]])
y = data["class"].apply(str.lower).values

values, counts = np.unique(y, return_counts=True)
index = np.isin(y, values[counts > 1])

X = X[index]
y = y[index]

# Initialize MLP classifier
mlp = DecisionTreeClassifier(random_state=0)

# Define k-fold cross-validation
kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0)

# Define variables to store accuracy scores
scores = []
scores_recall = []
scores_precision = []
scores_acc = []

# Iterate over each fold
for train_index, test_index in kf.split(X, y):
    # Split data into training and testing sets for current fold
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train MLP on training set
    mlp.fit(X_train, y_train)

    # Make predictions on testing set
    y_pred = mlp.predict(X_test)

    # Compute accuracy score for current fold
    score = f1_score(y_test, y_pred, labels=np.unique(y), average="weighted")
    score_recall = recall_score(y_test, y_pred, labels=np.unique(y), average="weighted")
    score_precision = precision_score(
        y_test, y_pred, labels=np.unique(y), average="weighted"
    )
    score_acc = accuracy_score(y_test, y_pred)

    # Append accuracy score to list of scores
    scores.append(score)
    scores_recall.append(score_recall)
    scores_precision.append(score_precision)
    scores_acc.append(score_acc)

# Compute average accuracy score across all folds
avg_score = np.mean(scores)
stddev = np.std(scores)

print(
    f"Average F1 score across all folds: {avg_score:.3f}, recall {np.mean(scores_recall):.3f}, precision {np.mean(scores_precision):.3f}, accuracy: {np.mean(scores_acc):.3f}"
)
