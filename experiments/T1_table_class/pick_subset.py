import pandas as pd

OUTPUT = "artifacts/viznet-subset.parquet"
CLASSES = ["publisher", "religion", "year", "industry", "city", "team", "address", "album", "country", "artist", "state", "isbn", "genre", "language"]

base_dir = "datasets/sherlock-viznet/"

# Load the val_labels.parquet file into a DataFrame
labels_df = pd.read_parquet(base_dir + 'test_labels.parquet')

# Load the val_values.parquet file into a DataFrame
values_df = pd.read_parquet(base_dir + 'test_values.parquet')

combined_df = pd.concat([labels_df, values_df], axis=1)
filtered_df = combined_df[combined_df['type'].isin(CLASSES)]
#random_subset = filtered_df.sample(n=2000)
filtered_df.to_parquet(OUTPUT)
