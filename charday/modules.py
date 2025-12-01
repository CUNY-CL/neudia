"""Uses a tagger spec to create all necessary tagger layers.

An empty key in the spec means that the diacritic:

- potentially applies to every segment, and
- is concatenative (i.e., non-spacing, a true diacritic)
"""

import lightning
from torch import nn
import yaml

# FIXME make this tunable.
INPUT_SIZE = 256

# FIXME put this somewhere else.
SpecType = dict[str, dict[str, str]]


# FIXME put this somewhere else.
def _load(path: str) -> SpecType:
    with open(path, "r") as source:
        return yaml.safe_load(source)


class Encoder(lightning.LightningModule):
    """Encoder layer.

    This consists of a bidirectional LSTM and associated embedding and
    dropout layers.

    Args:
        dropout (float, optional): dropout probability.
        embedding_size (int, optional): dimensionality of embedding.
        hidden_size (int, optional): encoder hidden layer size.
        layers (int, optional): number of encoder layers.
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
        embedded = self.dropout_layer(self.embeddings(symbols))
        packed, _ = nn.utils.rnn.pack_padded_sequence(
            sequence,
            source.lengths(),
            batch_first=True,
            enforce_sorted=True,
        )
        encoded = self.lstm(packed)
        return nn.utils.rnn.pad_packed_sequence(
            encoded, batch_first=True, padding_value=special.PAD_IDX
        )


class Classifier(lightning.LightningModule):
    """Classifier layer.

    This consists of linear layers representing taggers for
    individual characters.

    Args:
        spec: the tagger spec.
        hidden_size (int, optional): encoder hidden layer size.
    """

    # FIXME add support for a global tagger notion.

    def __init__(
        self,
        spec: SpecType,
        hidden_size: int = defaults.HIDDEN_SIZE,
        # Optimization and LR scheduling.
        **kwargs
    ):
        super().__init__()
        self.taggers = nn.ParameterDict()
        for defec, plenes in spec.items():
            self.taggers[defec] = nn.Linear(hidden_size, len(plenes))

    def get_tagger(self, defec: str) -> nn.Linear | None:
        return self.taggers.get(char)

    def forward(
        self, encodings: torch.Tensor, defec: str
    ) -> torch.Tensor | None:
        tagger = self.get_tagger(defec)
        if not tagger:
            return None
        return tagger(encodings)


spec = _load("../configs/lat_naive.yaml")
# spec = _load("../configs/heb.yaml")
classifier = Classifier(spec)
print(classifier)
