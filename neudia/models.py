"""The Neudia model.

In the documentation below, N is the batch size, C is the number of tags, and
L is the maximum length (in tags) of a sentence in the batch.
"""

import logging

import lightning
from lightning.pytorch import cli
import torch
from torch import nn, optim
from torchmetrics import classification
from yoyodyne.models import embeddings, modules

from . import data, defaults, taggers, special


class Neudia(lightning.LightningModule):
    """Neudia model.

    This model performs character tagging using a shared LSTM encoder and
    a shared linear classifier.

    Args:
        encoder: Yoyodyne encoder instance.
        embedding_size: dimensionality of the embedding layer.
        label_smoothing: label smoothing coefficient.
    """

    # TODO(kbg): ideally we'd have a type like `BaseEncoder` that's a
    # bit more specific, but this has to be fixed upstream with Yoyodyne.

    embeddings: nn.Embedding
    encoder: modules.BaseModule
    tagger: taggers.Tagger
    loss_func: nn.CrossEntropyLoss
    optimizer: optim.Optimizer
    scheduler: optim.lr_scheduler.LRScheduler
    encoder_keep: torch.Tensor
    tags_mask: torch.Tensor
    # Used for validation in `fit` and testing in `test`.
    accuracy: classification.MulticlassAccuracy | None

    def __init__(
        self,
        encoder: modules.BaseModule,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        label_smoothing: float = defaults.LABEL_SMOOTHING,
        *,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        # Dummy values filled in via link.
        source_vocab_size: int = 2,
        tags_vocab_size: int = 2,
        source2tags: dict[int, list[int]] = None,
    ):
        super().__init__()
        self.encoder = encoder
        # TODO: should we consider using Xavier embeddings if the encoder is a
        # transformer?
        self.embeddings = embeddings.normal_embedding(
            source_vocab_size, embedding_size
        )
        self.tagger = taggers.Tagger(self.encoder.output_size, tags_vocab_size)
        self.loss_func = nn.CrossEntropyLoss(
            ignore_index=special.PAD_IDX, label_smoothing=label_smoothing
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = classification.MulticlassAccuracy(
            tags_vocab_size, average="micro", ignore_index=special.PAD_IDX
        )
        # Precomputing constraints.
        tags_mask = torch.zeros(
            source_vocab_size, tags_vocab_size, dtype=torch.bool
        )
        encoder_keep = torch.zeros(source_vocab_size, dtype=torch.bool)
        for idx, allowed_tags in source2tags.items():
            tags_mask[idx, allowed_tags] = True
            encoder_keep[idx] = True
        self.register_buffer("tags_mask", tags_mask)
        self.register_buffer("encoder_keep", encoder_keep)
        self.save_hyperparameters()

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]

    def forward(self, batch: data.Batch) -> torch.Tensor:
        encoded = self.encoder(batch.source, self.embeddings)
        # Removes the second dimension of the encoder output when the
        # corresponding source symbol, the one being encoded, doesn't have any
        # associated tags. We also build a packed source tensor for masking
        # the tagger outputs later.
        keep_mask = self.encoder_keep[batch.source.tensor]
        encoded_filtered = []
        source_filtered = []
        for encoded_row, source_row, mask_row in zip(
            encoded, batch.source.tensor, keep_mask
        ):
            encoded_filtered.append(encoded_row[mask_row])
            source_filtered.append(source_row[mask_row])
        encoded = nn.utils.rnn.pad_sequence(
            encoded_filtered,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        logits = self.tagger(encoded)
        # Masks out the tags that aren't compatible with the source sequence.
        source = nn.utils.rnn.pad_sequence(
            source_filtered,
            batch_first=True,
            padding_value=special.PAD_IDX,
        )
        valid_tags = self.tags_mask[source]
        logits.masked_fill_(~valid_tags, defaults.NEG_EPSILON)
        # The logits are of shape N x L x C, but loss and accuracy functions
        # expect N x C x L, so we transpose to produce this shape.
        return logits.transpose(1, 2)

    # See the following for how these are called by the different subcommands.
    # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

    def on_fit_start(self):
        # Rather than crashing, we simply warn about lack of deterministic
        # algorithms.
        if torch.are_deterministic_algorithms_enabled():
            logging.info("(Only) warning about non-deterministic algorithms")
            torch.use_deterministic_algorithms(True, warn_only=True)

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
        self.accuracy.update(logits, batch.tags.tensor)

    def on_validation_epoch_end(self) -> None:
        self.log(
            "val_accuracy",
            self.accuracy.compute(),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_start(self) -> None:
        self.accuracy.reset()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        self.accuracy.update(logits, batch.tags.tensor)

    def on_test_epoch_end(self) -> None:
        self.log(
            "test_accuracy",
            self.accuracy.compute(),
            logger=True,
            on_epoch=True,
            prog_bar=True,
        )
