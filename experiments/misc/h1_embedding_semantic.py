import os

import numpy as np
import pandas as pd
import openai

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("MacOSX")

openai.api_key = os.getenv("OPENAI_API_KEY")
file_path = "artifacts/h1_embeddings_semantic.parquet"

if not os.path.exists(file_path):
    # Lists to hold generated emails, names and addresses
    politicians = ["George Washington", "Thomas Jefferson", "Abraham Lincoln", "Theodore Roosevelt", "Franklin D. Roosevelt", "John F. Kennedy", "Richard Nixon", "Ronald Reagan", "Bill Clinton", "George W. Bush", "Barack Obama", "Donald Trump", "Joe Biden"]
    actors = ["Tom Hanks", "Meryl Streep", "Robert De Niro", "Julia Roberts", "Denzel Washington", "Sandra Bullock", "Leonardo DiCaprio", "Jennifer Lawrence", "Morgan Freeman", "Scarlett Johansson"]
    scientists = ["Albert Einstein", "Richard Feynman", "Carl Sagan", "Barbara McClintock", "James Watson", "Edward O. Wilson", "Neil deGrasse Tyson", "Jane Goodall", "George Washington Carver", "Sally Ride"]
    musicians = ["Elvis Presley", "Bob Dylan", "Louis Armstrong", "Michael Jackson", "Madonna", "Bruce Springsteen", "Beyoncé", "Prince", "Taylor Swift", "Kanye West"]


    # Combine all data into one list
    data = (
        [("Politician", x) for x in politicians]
        + [("Actor", x) for x in actors]
        + [("Scientist", x) for x in scientists]
        + [("Artist", x) for x in musicians]
    )

    # Convert data into a pandas DataFrame
    df = pd.DataFrame(data, columns=["type", "value"])

    def get_embedding(text, model="text-embedding-ada-002"):
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=model)["data"][0][
            "embedding"
        ]

    df["embedding"] = df.value.apply(get_embedding)
    df.to_parquet(file_path)
else:
    df = pd.read_parquet(file_path)

# Perform t-SNE on your high-dimensional embeddings
tsne = TSNE(n_components=2, random_state=0)
embeddings_tsne = tsne.fit_transform(np.array(df["embedding"].tolist()))

# Add the t-SNE dimensions to your DataFrame
df["tsne-2d-one"] = embeddings_tsne[:, 0]
df["tsne-2d-two"] = embeddings_tsne[:, 1]

# Now we can create a scatterplot of the 2 t-SNE dimensions
plt.figure(figsize=(4, 2.5))
scatterplot = sns.scatterplot(
    x="tsne-2d-one",
    y="tsne-2d-two",
    hue="type",  # Color by type
    style="type",
    palette=sns.color_palette(
        "deep", len(df["type"].unique())
    ),  # Choose a color palette
    data=df,
    legend="full",
    alpha=0.9,
)
scatterplot.legend_.set_title("")
sns.move_legend(scatterplot, "upper left")
# plt.legend(loc="best", mode="expand", ncol=1)
plt.xlabel("First t-SNE component")
plt.ylabel("Second t-SNE component")

plt.savefig("results/h1-tsne-semantic.pdf", bbox_inches="tight")
