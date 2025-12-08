"""Neudia modules."""

import lightning
import torch
from torch import nn

from . import data, defaults, special


class Encoder(lightning.LightningModule):
    """Encoder layer.

    This consists of a bidirectional LSTM and associated embedding and
    dropout layers.

    Args:
        dropout: dropout probability.
        embedding_size: dimensionality of embedding.
        hidden_size: encoder hidden layer size.
        layers: number of encoder layers.
    """

    dropout: float
    dropout_layer: nn.Dropout
    embeddings: nn.Embedding
    lstm: nn.LSTM

    def __init__(
        self,
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        hidden_size: int = defaults.HIDDEN_SIZE,
        layers: int = defaults.LAYERS,
    ):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(embedding_size, embedding_size)
        nn.init.constant_(self.embeddings.weight[special.PAD_IDX], 0.0)
        self.lstm = nn.LSTM(
            embedding_size,
            hidden_size,
            layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )

    def forward(self, source: data.PaddedTensor) -> torch.Tensor:
        embedded = self.dropout_layer(self.embeddings(source.tensor))
        print("embedded sequence:", embedded.shape)
        packed, _ = nn.utils.rnn.pack_padded_sequence(
            embedded,
            source.lengths(),
            batch_first=True,
            enforce_sorted=True,
        )
        encoded = self.lstm(packed)
        encoded = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, padding_value=special.PAD_IDX
        )
        print("padded encoded sequence:", encoded.shape)
        # FIXME
        return encoded[MASK_ME]


class Tagger(lightning.LightningModule):
    """Tagger layer.

    This consists of linear layers representing taggers for
    individual characters.

    Args:
        vocab_size: number of tags.
        hidden_size: encoder hidden layer size.
    """

    def __init__(
        self,
        hidden_size: int = defaults.HIDDEN_SIZE,
        *,
        # Dummy value; it will be set by the dataset object.
        vocab_size: int = 2,
    ):
        super().__init__()
        self.tagger = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        logits = self.tagger(encoded)
        print("tagger logits:", logits.shape)
        # Masks out values that are unreachable.
        # FIXME
        return torch.where(
            MASK_ME, logits, torch.full_like(logits, defaults.NEG_EPSILON)
        )
