"""CharDay: character tagging.

This module just silences some uninformative warnings.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    ".*adds dropout after all but last recurrent layer.*",
)
warnings.filterwarnings("ignore", ".*is a wandb run already in progress.*")
