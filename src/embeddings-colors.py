from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")


def get_embeddings(word):
    tokens = tokenizer(word, return_tensors="pt")
    with torch.no_grad():
        output = model(**tokens)

    return output.last_hidden_state[:, 0, :].squeeze().numpy()


words = ["school", "professor", "mouse", "computer", "cat", "dog"]

embeddings = {word: get_embeddings(word) for word in words}

from sklearn.decomposition import PCA

pca = PCA(n_components=2)

embeddings_2d = pca.fit_transform(list(embeddings.values()))

coordinates = {word: coord for word, coord in zip(words, embeddings_2d)}

import matplotlib.pyplot as plt
import seaborn as sns

colors = sns.color_palette("husl", len(words))

plt.figure(figsize=(10, 6))

for i, word in enumerate(words):
    plt.scatter(coordinates[word][0], coordinates[word][1], color=colors[i], label=word)
    plt.text(
        coordinates[word][0],
        coordinates[word][1],
        word,
        fontsize=12,
        ha="right",
        va="bottom",
    )
plt.title("Embeddings with Coltransformersors Visualization")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.savefig('visualizations/embeddings_visualization.png')
plt.close()
