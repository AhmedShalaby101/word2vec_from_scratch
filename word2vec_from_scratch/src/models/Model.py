##########################MODEL#####################################
import torch.nn as nn
from src.Utils.constants import EMBED_DIMENSION, EMBED_MAX_NORM


class CBOW(nn.Module):
    def __init__(self, vocab_size: int):
        super(CBOW, self).__init__()
        self.Embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBED_DIMENSION,
            max_norm=EMBED_MAX_NORM,
        )
        self.Outputs = nn.Linear(in_features=EMBED_DIMENSION, out_features=vocab_size)

    def forward(self, inputs):
        weights = self.Embedding(inputs)
        weights = weights.mean(axis=1)
        outputs = self.Outputs(weights)
        return outputs
