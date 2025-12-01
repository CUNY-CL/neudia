This directory contains some helper scripts:

[`methods.py`](methods.py), while not executable itself, implements a simple
Unicode decomposition method as well as the *Uninames* method used by the other
scripts.

- [`discover.py`](discover.py) can be used to build a tagging spec using plene
  text. More precise tagger specs can be written by hand or by adapting the
  output.
- [`strip.py`](strip.py) creates a TSV file of parallel data from plene text.
