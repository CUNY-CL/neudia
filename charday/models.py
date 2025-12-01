

from . import data, defaults, modules, special


class CharDay(lightning.LightningModule):
    """CharDay model.

    This model performs character tagging using a shared LSTM encoder and as
    many small linear layers as there are characters requiring tagging.

    Args:
        spec (SpecType): tagger specification.
        dropout (float, optional): dropout probability.
        embedding_size (int, optional): dimensionality of embedding.
        hidden_size (int, optional): encoder hidden layer size.
        label_smoothing (float, optional): label smoothing coefficient.
        layers (int, optional): number of encoder layers.
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
        spec: SpecType,
        dropout: float = defaults.DROPOUT,
        embedding_size: int = defaults.EMBEDDING_SIZE,
        hidden_size: int = defaults.HIDDEN_SIZE,
        label_smoothing: float = defaults.LABEL_SMOOTHING,
        layers: int = defaults.LAYERS
        *,
        optimizer: cli.OptimizerCallable = defaults.OPTIMIZER,
        scheduler: cli.LRSchedulerCallable = defaults.SCHEDULER,
        num_tags: int = 2  # Dummy value filled in via link.
    ):
        super().__init__()
        self.encoder = modules.Encoder(dropout, embedding_size, hidden_size, layers)
        self.classifier = modules.Classifier(spec, hidden_size)
        self.loss_func = nn.CrossEntropyLoss(ignore_index=special.PAD_IDX, label_smoothing=label_smoothing)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.accuracy = classification.MulticlassAccuracy(
            num_tags, average="micro", ignore_index=special.PAD_IDX
        )
        self.save_hyperparamaters()

    def forward(self, batch: data.Batch) -> data.Logits:
        return self.classifier(self.encoder(batch))


    # See the following for how these are called by the different subcommands.
    # https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks

    def predict_step(self, batch: data.Batch, batch_idx: int) -> data.Logits:
        return self(batch)

    def training_step(self, batch: data.Batch, batch_idx: int) -> torch.Tensor:
        """Runs one step of training.

        Training loss is tracked.

        Args:
            batch (data.Batch)
            batch_idx (int).

        Returns:
            torch.Tensor: training loss.
        """
        predictions = self(batch)
        loss = self.loss_func(predictions, batch.target.padded)
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

    def on_test_epoch_start(self) -> None:
        self.accuracy.reset()

    def test_step(self, batch: data.Batch, batch_idx: int) -> None:
        logits = self(batch)
        self.accuracy.updates(logits, batch)

    # FIXME fill this all in.
