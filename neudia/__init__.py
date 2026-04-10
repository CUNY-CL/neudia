"""Neudia: neural diacritization."""

import warnings

# Silences some stupid warnings.
warnings.filterwarnings(
    "ignore",
    r"(?s).*adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings(
    "ignore", r"(?s).*is a wandb run already in progress.*"
)
warnings.filterwarnings("ignore", r"(?s).*Unable to serialize instance.*")
