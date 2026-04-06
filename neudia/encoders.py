"""Additional encoder models.

Neudia inherits randomly-initialized RNN and transformer encoders from
Yoyodyne, wrapping them so as to own a needed embedding layer.

Neudia also supports pre-trained ByT5 encoders.
"""

from __future__ import annotations

import abc
import itertools

import torch
from torch import nn
import transformers
from yoyodyne.models import embeddings, modules

from . import data, defaults

# We use a mixin system to wrap the Yoyodyne encoders.


class EmbeddingMixin(abc.ABC):
    """Adds an owned embedding layer to a Yoyodyne encoder."""

    embeddings: nn.Embedding

    def __init__(self, source_vocab_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.embeddings = self.init_embeddings(
            source_vocab_size, self.embedding_size
        )

    def forward(self, symbols, *args, **kwargs) -> torch.Tensor:
        return super().forward(symbols, self.embeddings, *args, **kwargs)

    @staticmethod
    @abc.abstractmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding: ...


class RNNEmbeddingMixin(EmbeddingMixin):

    @staticmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        return embeddings.normal_embedding(num_embeddings, embedding_size)


class TransformerEmbeddingMixin(EmbeddingMixin):

    @staticmethod
    def init_embeddings(
        num_embeddings: int, embedding_size: int
    ) -> nn.Embedding:
        return embeddings.xavier_embedding(num_embeddings, embedding_size)


class GRUEncoder(RNNEmbeddingMixin, modules.GRUEncoder):
    pass


class LSTMEncoder(RNNEmbeddingMixin, modules.LSTMEncoder):
    pass


class TransformerEncoder(
    TransformerEmbeddingMixin, modules.TransformerEncoder
):
    pass


class RotaryTransformerEncoder(
    TransformerEmbeddingMixin, modules.RotaryTransformerEncoder
):
    pass


class ByT5Encoder(modules.BaseEncoder):
    """ByT5-backed character-level encoder.

    Runs a pre-trained ByT5 encoder over the UTF-8 byte tokens that make up
    each source character, then mean-pools the byte-level hidden states back
    to character-level representations.

    An index is required to invert the integer encoding back to strings
    temporarily.

    Args:
        model_name: HuggingFace model name.
        index: index.
        pooling_layers: number of layers to use to compute embeddings.
        *args: passed to superclass.
        **kwargs: passed to superclass.
    """

    model_name: str
    index: data.Index
    pooling_layers: int
    model: transformers.T5EncoderModel
    tokenizer: transformers.AutoTokenizer
    byte_lengths: dict[str, int]

    def __init__(
        self,
        model_name: str = "google/byt5-base",
        index: data.Index | None = None,  # Filled in when instantiating.
        pooling_layers: int = defaults.POOLING_LAYERS,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = model_name
        self.index = index
        self.pooling_layers = pooling_layers
        self.byte_lengths = {}
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.model = transformers.T5EncoderModel.from_pretrained(model_name)

    def forward(
        self,
        symbols: "data.PaddedTensor",  # noqa: F821
        *args,
        **kwargs,
    ) -> torch.Tensor:
        """Encodes a padded batch of source-symbol index sequences.

        Args:
            symbols: padded integer tensor of shape N × L.
            *args: ignored.
            **kwargs: ignored.

        Returns:
            Encoded tensor.
        """
        batch_size = len(symbols)
        source_length = symbols.tensor.size(1)
        # Decodes integer indices back to character strings.
        char_strings = self._decode_to_chars(symbols)
        # Builds per-example byte-token sequences (with alignment
        # maps) and pad them into a single batch for ByT5.
        input_ids, attention_mask, alignments = self._tokenize_batch(
            char_strings
        )
        # Runs the encoder.
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.pooling_layers > 1,
        )
        if self.pooling_layers == 1:
            # Special case for just using the last layer's hidden states.
            encoded = output.last_hidden_state
        else:
            # Mean-pools the last n layers' hidden states.
            encoded = torch.stack(
                output.hidden_states[-self.pooling_layers :]
            ).mean(dim=0)
        # Mean-pools byte positions to obtain character encodings.
        encoded = self._pool_bytes_to_chars(
            encoded, alignments, batch_size, source_length
        )
        # Zeros out padding positions.
        encoded[symbols.mask] = 0.0
        return self.dropout_layer(encoded)

    @property
    def output_size(self) -> int:
        return self.model.config.d_model

    @property
    def name(self) -> str:
        return f"ByT5 ({self.model_name})"

    def _decode_to_chars(
        self,
        symbols: "data.PaddedTensor",  # noqa: F821
    ) -> list[list[str]]:
        """Converts integer index rows back to lists of character strings.

        Args:
            symbols: padded tensor.

        Returns:
            A List of lists, each containing the non-padding character strings
            for that example.
        """
        return [
            [
                self.index.source_vocabulary.get_symbol(idx)
                for idx, is_pad in itertools.takewhile(
                    lambda t: not t[1], zip(row, mask)
                )
            ]
            for row, mask in zip(
                symbols.tensor.tolist(), symbols.mask.tolist()
            )
        ]

    def _byte_length(self, char: str) -> int:
        """Returns the number of ByT5 byte tokens for a single character.

        Results are cached in byte_lengths.

        Args:
            char: the character string.

        Returns:
            Number of byte tokens.
        """
        if char not in self.byte_lengths:
            self.byte_lengths[char] = max(
                len(self.tokenizer.encode(char, add_special_tokens=False)), 1
            )
        return self.byte_lengths[char]

    def _tokenize_batch(
        self,
        char_strings: list[list[str]],
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[list[int]]]]:
        """Tokenizes a batch of character lists into ByT5 byte tokens.

        Each example's characteres are joined into a single string and the
        tokenizer is called once per example. The alignment is then
        reconstructed by re-tokenizing each character individually, counting
        how many byte tokens each produces.

        Args:
            char_strings: list of lists of character strings.

        Returns:
            input IDs, attention mask, and alignments.
        """
        raw_ids: list[list[int]] = []
        alignments: list[list[list[int]]] = []
        for chars in char_strings:
            # Tokenizes the joined string (adds <eos>, handles multi-byte).
            joined = "".join(chars)
            enc = self.tokenizer(
                joined,
                add_special_tokens=True,
                return_tensors=None,
            )
            ids = enc["input_ids"]  # Includes trailing <eos>.
            n_byte_tokens = len(ids) - 1
            # Builds per-character alignment by counting byte tokens for each
            # character individually, then assigning contiguous ranges.
            alignment: list[list[int]] = []
            cursor = 0
            for char in chars:
                # Looks up the source vocabulary index for cache keying. The
                # cache key is the char string itself.
                n = self._byte_length(char)
                end = min(cursor + n, n_byte_tokens)
                alignment.append(list(range(cursor, end)))
                cursor = end
            alignments.append(alignment)
            raw_ids.append(ids)
        # Pads to max length.
        max_length = max(len(ids) for ids in raw_ids)
        pad_idx = self.tokenizer.pad_token_id
        padded_ids = [
            ids + [pad_idx] * (max_length - len(ids)) for ids in raw_ids
        ]
        attn_masks = [
            [1] * (len(ids) - 1) + [0] * (1 + max_length - len(ids))
            for ids in raw_ids
        ]
        input_ids = torch.tensor(
            padded_ids, dtype=torch.long, device=self.device
        )
        attention_mask = torch.tensor(
            attn_masks, dtype=torch.long, device=self.device
        )
        return input_ids, attention_mask, alignments

    def _pool_bytes_to_chars(
        self,
        encoded: torch.Tensor,
        alignments: list[list[list[int]]],
        batch_size: int,
        source_length: int,
    ) -> torch.Tensor:
        """Mean-pools byte-level hidden states to character-level.

        Args:
            encoded: encodings from ByT5.
            alignments.
            batch_size.
            source_length.

        Returns:
            Tensor of pooled encodings.
        """
        pooled = torch.zeros(
            batch_size, source_length, encoded.size(2), device=self.device
        )
        for i, char_alignment in enumerate(alignments):
            for j, byte_positions in enumerate(char_alignment):
                if not byte_positions:
                    # Fallback: shouldn't normally happen, but if alignment
                    # went wrong, this position is left as zero.
                    continue
                positions = torch.tensor(
                    byte_positions, dtype=torch.long, device=self.device
                )
                # Mean over the byte-token hidden states for this character.
                pooled[i, j] = encoded[i, positions].mean(dim=0)
        return pooled
