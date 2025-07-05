import streamlit as st
import torch
import torch.nn.functional as F
from src.models.Model import CBOW
import pandas as pd

# load vocab
word2id = pd.read_pickle("data/interim/word2id")
word2id = dict(zip(word2id["Word"], word2id["ID"]))

id2word = pd.read_pickle("data/interim/id2word")
id2word = dict(zip(id2word["ID"], id2word["Word"]))
vocab = len(word2id)

# load model
model = CBOW(vocab)
model = torch.load("word2vec_model.pth", weights_only=False)

print(torch.version.git_version)


# --- Function to get the vector for a word
def get_vector(word):
    idx = word2id.get(word.lower(), 0)
    if idx is None or idx >= len(model.Embedding.weight):
        return None
    return model.Embedding.weight[idx].detach()


# --- Function to find similar words
def find_similar_words(word, top_n=5):
    vector = get_vector(word)
    if vector is None:
        return []
    all_vectors = model.Embedding.weight.detach()
    similarities = F.cosine_similarity(vector.unsqueeze(0), all_vectors)
    top_indices = torch.topk(similarities, top_n + 1).indices.tolist()
    similar_words = [
        (id2word[i], round(float(similarities[i].item()), 4))
        for i in top_indices
        if id2word[i] != word
    ][:top_n]
    return similar_words


# --- Streamlit UI
st.title("🔍 Word Similarity Explorer")
st.write("Enter a word to find semantically similar terms using your Word2Vec model.")

# Dropdown or input
selected_word = st.selectbox("Choose a word:", sorted(list(word2id.keys())))
top_n = st.slider("Number of similar words:", min_value=1, max_value=20, value=5)

if st.button("Find Similar Words"):
    similar = find_similar_words(selected_word, top_n=top_n)
    if similar:
        st.success("Top similar words:")
        for w, score in similar:
            st.write(f"**{w}** — Similarity: `{score * 100:.2f}%`")
    else:
        st.warning("Word not found or no similar words available.")
