import os

import numpy as np
import pandas as pd
from faker import Faker
from faker.providers.ssn import en_GB, zh_CN, en_US, cs_CZ, it_IT
import openai

from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("MacOSX")

openai.api_key = os.getenv("OPENAI_API_KEY")
Faker.seed(0)
file_path = "artifacts/h1_embeddings.parquet"

if not os.path.exists(file_path):
    # Create Faker object
    fake_uk = en_GB.Provider(Faker('en_GB'))
    fake_usa = en_US.Provider(Faker('en_US'))
    fake_china = zh_CN.Provider(Faker('zh_CN'))
    fake_czech = cs_CZ.Provider(Faker('cs_CZ'))
    fake_italy = it_IT.Provider(Faker('it_IT'))

    # Lists to hold generated emails, names and addresses
    USA = []
    China = []
    UK = []
    Czech = []
    Italy = []

    # Generate 10 emails, names, and addresses
    for _ in range(10):
        USA.append(fake_usa.ssn())
        UK.append(fake_uk.ssn())
        China.append(fake_china.ssn())
        Czech.append(fake_czech.birth_number())
        Italy.append(fake_italy.ssn())

    # Combine all data into one list
    data = (
        [("USA", x) for x in USA]
        + [("China", x) for x in China]
        + [("UK", x) for x in UK]
        + [("Czech", x) for x in Czech]
        + [("Italy", x) for x in Italy]
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
sns.move_legend(scatterplot, "upper right")
# plt.legend(loc="best", mode="expand", ncol=1)
plt.xlabel("First t-SNE component")
plt.ylabel("Second t-SNE component")

plt.savefig("results/h1-tsne.pdf", bbox_inches="tight")
