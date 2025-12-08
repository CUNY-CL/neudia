"""The Neudia model."""

import lightning
from lightning.pytorch import cli
import torch
from torch import nn, optim
from torchmetrics import classification

from . import data, defaults, modules, special


class Neudia(lightning.LightningModule):
    """Neudia model.

    This model performs character tagging using a shared LSTM encoder and
    a shared linear classifier.

    Args:
        dropout: dropout probability.
        embedding_size: dimensionality of embedding.
        hidden_size: encoder hidden layer size.
        label_smoothing: label smoothing coefficient.
        layers: number of encoder layers.
    """

    encoder: modules.Encoder
    classifier: modules.Classifier
    loss_func: nn.CrossEntropy
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    # Used for validation in `fit` and testing in `test`.
    accuracy: classification.MulticlassAccuracy | None

    def __init__(
        self,
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        hidden_size: int = defaults.HIDDEN_SIZE,
        label_smoothing: float = defaults.LABEL_SMOOTHING,
        layers: int = defaults.LAYERS,
        *,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        vocab_size: int = 2,  # Dummy value filled in via link.
    ):
        super().__init__()
        self.encoder = modules.Encoder(
            dropout, embedding_size, hidden_size, layers
        )
        self.classifier = modules.Tagger(hidden_size, vocab_size)
        self.loss_func = nn.CrossEntropyLoss(
            ignore_index=special.PAD_IDX, label_smoothing=label_smoothing
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = classification.MulticlassAccuracy(
            vocab_size, average="micro", ignore_index=special.PAD_IDX
        )
        self.save_hyperparamaters()

    def forward(self, batch: data.Batch) -> torch.Tensor:
        encoded = self.encoder(batch.source)
        return self.tagger(encoded)

    # See the following for how these are called by the different subcommands.
    # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

    def predict_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        return self(batch)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        logits = self(batch)
        loss = self.loss_func(logits, batch.tags.tensor)
        self.log(
            "train_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        self.accuracy.reset()

    def validation_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        loss = self.loss_func(logits, batch.tags.tensor)
        self.log(
            "val_loss",
            loss,
            batch_size=len(batch),
            logger=True,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
        )

    def on_test_epoch_start(self) -> None:
        self.accuracy.reset()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        self.accuracy.updates(logits, batch.tags.tensor)
