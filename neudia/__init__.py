"""Neudia: neural diacritization."""

import warnings

# Silences some stupid warnings.
warnings.filterwarnings(
    "ignore",
    ".*adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
