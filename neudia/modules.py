"""Neudia modules.

In the documentation below, N is the batch size, C is the number of tags, and
L is the maximum length (in tags) of a sentence in the batch.
"""

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
        vocab_size: int = 2,  # Dummy value filled in via link.
    ):
        super().__init__()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        # Randomly initializes embeddings.
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
        lengths = source.lengths()
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.lstm(packed)
        padded, _ = nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, padding_value=special.PAD_IDX
        )
        return padded


class Tagger(lightning.LightningModule):
    """Tagger layer.

    This consists of linear layers representing taggers for
    individual characters.

    Args:
        hidden_size: encoder hidden layer size.
        vocab_size: number of tags.
    """

    def __init__(
        self,
        hidden_size: int = defaults.HIDDEN_SIZE,
        vocab_size: int = 2,  # Dummy value filled in via link.
    ):
        super().__init__()
        # 2x because of bidirectionality.
        self.tagger = nn.Linear(2 * hidden_size, vocab_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Computes logits for the taggers.

        This takes the contextual encodings of the source symbols which
        require tagging and then computes the logits. This yields logits
        of shape N x L x C. Loss and accuracy functions expect N x C x L,
        so we transpose to produce this shape.
        """
        return self.tagger(encoded).transpose(1, 2)
