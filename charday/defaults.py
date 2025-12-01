"""Default values for flags and modules."""

import numpy

from torch import optim

from . import schedulers

# All elements should be styled as CONSTANTS.

# Default text encoding.
ENCODING = "utf-8"

# Data configuration arguments.
SOURCE_COL = 1
TARGET_COL = 2

# Architectural arguments.
EMBEDDING_SIZE = 128
HIDDEN_SIZE = 512
LAYERS = 1

# Training arguments.
BATCH_SIZE = 32
BETA1 = 0.9
BETA2 = 0.999
DROPOUT = 0.2
LABEL_SMOOTHING = 0.0
OPTIMIZER = optim.Adam
SCHEDULER = schedulers.Dummy
