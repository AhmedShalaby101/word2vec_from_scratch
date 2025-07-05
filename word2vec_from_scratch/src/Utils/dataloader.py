################################DATALOADERS##################################
import torch
from functools import partial
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
import re
import os
from nltk.corpus import stopwords
from constants import (
    CBOW_N_WORDS,
    MIN_WORD_FREQUENCY,
    MAX_SEQUENCE_LENGTH,
)


class text_dataset(Dataset):
    def __init__(self, filepath, text_column="text"):
        df = pd.read_pickle(filepath)
        self.samples = df[text_column].tolist()

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


def data_iterator(data_dir, filename, text_column="text"):
    filepath = os.path.join(data_dir, filename)
    dataset = text_dataset(filepath, text_column)
    return dataset


def tokenizer(text: str):
    stop_words = set(stopwords.words("english"))

    text = re.sub(r"</?s>|\[/?INST\]", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    cleaned_text = text.lower()
    words = cleaned_text.split()

    words = [word for word in words if word not in stop_words]
    return words


def build_vocab(data_iter, tokenizer):
    sentences = map(tokenizer, data_iter)
    word_counts = Counter(word for sentence in sentences for word in sentence)
    vocab_words = [
        word for word, freq in word_counts.items() if freq >= MIN_WORD_FREQUENCY
    ]
    # from word get ID
    word2id = {word: idx for idx, word in enumerate(vocab_words, start=1)}
    word2id["<unk>"] = 0
    # from ID get word
    id2word = {idx: word for word, idx in word2id.items()}
    return word2id, id2word


def collate_cbow(batch, text_pipeline):
    ##batch is a list of text paragraph
    batch_input, batch_output = [], []
    for text in batch:
        tokens_IDs = text_pipeline(text)  # return a list of the words IDS
        if len(tokens_IDs) < CBOW_N_WORDS * 2 + 1:
            continue
        if MAX_SEQUENCE_LENGTH:
            tokens_IDs = tokens_IDs[:MAX_SEQUENCE_LENGTH]

        for idx in range(len(tokens_IDs) - CBOW_N_WORDS * 2):
            token_id_sequence = tokens_IDs[idx : (idx + CBOW_N_WORDS * 2 + 1)]
            output = token_id_sequence.pop(CBOW_N_WORDS)
            input_ = token_id_sequence
            batch_input.append(input_)
            batch_output.append(output)

    batch_input = torch.tensor(batch_input, dtype=torch.long)
    batch_output = torch.tensor(batch_output, dtype=torch.long)
    return batch_input, batch_output


def get_dataloader_word2id_id2word(
    filename, data_dir, batch_size, shuffle, word2id=None
):
    data_itr = data_iterator(data_dir, filename)
    tokenizer_ = tokenizer
    if not word2id:
        word2id, id2word = build_vocab(data_itr, tokenizer_)
    text_pipeline = lambda x: [word2id.get(word, 0) for word in tokenizer(x)]
    collate_fn = collate_cbow
    dataloader = DataLoader(
        data_itr,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=partial(collate_fn, text_pipeline=text_pipeline),
    )
    return dataloader, word2id, id2word
