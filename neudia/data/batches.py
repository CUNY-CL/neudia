"""Batch objects."""

import torch
from torch import nn

from .. import special


class PaddedTensor(nn.Module):
    """A tensor with an optional padding mask.

    Args:
        tensorlist: a list of tensors.
    """

    tensor: torch.Tensor

    def __init__(self, tensorlist: list[torch.Tensor]):
        super().__init__()
        pad_len = max(len(tensor) for tensor in tensorlist)
        self.register_buffer(
            "tensor",
            torch.stack(
                [self.pad_tensor(tensor, pad_len) for tensor in tensorlist]
            ),
        )

    @staticmethod
    def pad_tensor(tensor: torch.Tensor, pad_len: int) -> torch.Tensor:
        """Pads a tensor.

        Args:
            tensor: the tensor to pad.
            pad_len: desired tensor length.

        Returns:
            A tensor with padding.
        """
        padding = pad_len - len(tensor)
        return nn.functional.pad(
            tensor, (0, padding), "constant", special.PAD_IDX
        )

    def __len__(self) -> int:
        return len(self.tensor)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}"
            f"({', '.join(str(i) for i in self.tensor.shape)})"
        )

    @property
    def mask(self) -> torch.Tensor:
        return self.tensor == special.PAD_IDX

    def lengths(self) -> torch.Tensor:
        """Computes the lengths of all strings in the tensor.

        This needs to be on CPU for packing.
        """
        return (~self.mask).sum(dim=1).cpu()


class Batch(nn.Module):
    """Source tensor, with optional tags tensor.

    Args:
        source: padded source tensor.
        tags: optional padded tag tensor.
    """

    source: PaddedTensor
    tags: PaddedTensor | None

    def __init__(self, source, tags=None):
        super().__init__()
        self.register_module("source", source)
        self.register_module("tags", tags)

    @property
    def has_tags(self) -> bool:
        return self.tags is not None

    def __len__(self) -> int:
        return len(self.source)
