"""Command-line interface."""

import logging

from lightning.pytorch import callbacks as pytorch_callbacks, cli
from yoyodyne import trainers

from . import callbacks, data, models


class NeudiaCLI(cli.LightningCLI):
    """The Neudia CLI interface.

    Use with `--help` to see the full list of options.
    """

    def add_arguments_to_parser(
        self, parser: cli.LightningArgumentParser
    ) -> None:
        parser.add_lightning_class_args(
            pytorch_callbacks.ModelCheckpoint,
            "checkpoint",
            required=False,
        )
        parser.add_lightning_class_args(
            callbacks.PredictionWriter,
            "prediction",
            required=False,
        )
        parser.link_arguments(
            "data.source_vocab_size",
            "model.source_vocab_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.tags_vocab_size",
            "model.tags_vocab_size",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.source2tags",
            "model.source2tags",
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data.model_dir",
            "trainer.logger.init_args.save_dir",
            apply_on="instantiate",
        )


def main() -> None:
    logging.basicConfig(
        format="%(levelname)s: %(asctime)s - %(message)s",
        datefmt="%d-%b-%y %H:%M:%S",
        level="INFO",
    )
    NeudiaCLI(
        models.Neudia,
        data.DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        # Prevents prediction logits from accumulating in memory; see the
        # documentation in `trainers.py` for more context.
        trainer_class=trainers.Trainer,
    )


def python_interface(args: cli.ArgsType = None):
    """Interface to use models through Python."""
    NeudiaCLI(
        models.Neudia,
        data.DataModule,
        parser_kwargs={"parser_mode": "omegaconf"},
        trainer_class=trainers.Trainer,
        args=args,
    )
