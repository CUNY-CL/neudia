"""Custom callbacks."""

from typing import TextIO

import lightning
from lightning.pytorch import callbacks, trainer
import torch
from yoyodyne import util

from . import data, models


class PredictionWriter(callbacks.BasePredictionWriter):
    """Writes predictions.

    Args:
        path: Path for the predictions file.
    """

    path: str
    sink: TextIO | None

    # path is given a default argument to silence a warning if no prediction
    # callback is configured.
    # TODO: remove default if this is addressed:
    #
    #   https://github.com/Lightning-AI/pytorch-lightning/issues/20851
    def __init__(
        self,
        path: str = "",
    ):
        super().__init__("batch")
        self.path = path
        self.sink = None

    # Required API.

    def on_predict_start(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        # Placing this here prevents the creation of an empty file in the case
        # where a prediction callback was specified but this is not running
        # in predict mode.
        util.mkpath(self.path)
        self.sink = open(self.path, "w")

    def write_on_batch_end(
        self,
        trainer: trainer.Trainer,
        model: models.Neudia,
        logits: torch.Tensor,
        batch_indices: list[int] | None,
        batch: data.Batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mapper = data.Mapper(trainer.datamodule.index)
        for source, tags in zip(
            batch.source.tensor, torch.argmax(logits, dim=1)
        ):
            print("".join(mapper.decode_tagged(source, tags)), file=self.sink)
        self.sink.flush()

    def on_predict_end(
        self, trainer: trainer.Trainer, pl_module: lightning.LightningModule
    ) -> None:
        self.sink.close()
