from src.table_class import table_class_experiment
from src.table_class_results import calculate_results

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import os

import openai
openai.api_key = "EMPTY" # Not support yet
openai.api_base = "http://localhost:8000/v1"
model = "vicuna-7b-v1.3"

file_path = f"artifacts/vicuna-small.csv"
if not os.path.exists(file_path):
	table_class_experiment(file_path, temperature=0.1, model=model)

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = "https://api.openai.com/v1"

f_score, r_score, p_score = calculate_results(file_path)
print(f"F1 Score: {f_score}, Precision: {p_score}, Recall: {r_score}")
