from src.table_class import table_class_experiment
from src.table_class_results import calculate_results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

model = "llama2"

file_path = f"artifacts/llama2.csv"
if not os.path.exists(file_path):
	table_class_experiment(file_path, temperature=0.1, model=model)

f_score, r_score, p_score = calculate_results(file_path)
print(f"F1 Score: {f_score}, Precision: {p_score}, Recall: {r_score}")
