"""Neudia tagger modules."""

import lightning
import torch
from torch import nn

from . import defaults


class Tagger(lightning.LightningModule):
    """Tagger layer.

    This consists of linear layers representing taggers for
    individual characters.

    Args:
        hidden_size: encoder hidden layer size; this should be
            2x the specified value for the encoder if using
            a bidirectional RNN.
        vocab_size: number of tags.
    """

    def __init__(
        self,
        hidden_size: int = defaults.HIDDEN_SIZE,
        vocab_size: int = 2,  # Dummy value filled in via link.
    ):
        super().__init__()
        self.tagger = nn.Linear(hidden_size, vocab_size)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        # We'll transpose this in the model.
        return self.tagger(encoded)
