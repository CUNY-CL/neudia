# Neudia ✒️

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/CUNY-CL/neudia/tree/master.svg?style=shield&circle-token=CCIPRJ_WpmicDuct9B3ToyWZLjVcX_f7e2378d72fae79fa4f8e8a002f29d277e2bf9b4)](https://dl.circleci.com/status-badge/redirect/gh/CUNY-CL/neudia/tree/master)

Neudia is a neural network-based diacritization system.

## Philosophy

Neudia is closely inspired by the Nakdimon diacritization system for Hebrew
(Gershuni & Pinter 2022), but is intended to be much more general.

## Design

The Neudia model consists of a randomly initialized bidirectional LSTM which
feeds into a tagger layer.

Lightning is used to generate the [training, validation, inference, and
evaluation
loops](https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#hooks).
The [LightningCLI
interface](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html#lightning-cli)
is used to provide a user interface and manage configuration.

Below, we use [YAML](https://yaml.org/) to specify configuration options, and we
strongly recommend users do the same. However, most configuration options can
also be specified using POSIX-style command-line flags.

## Authors

Neudia was created by [Kyle Gorman](https://wellformedness.com) and [opther
contributors](https://github.com/CUNY-CL/neudia/graphs/contributors) like you.

## Installation

To install Neudia and its dependencies, run the following command;

    pip install .

## File formats

Other than YAML configuration files, Yoyodyne operates on basic tab-separated
values (TSV) data files.

### Data format

The default data format is a two-column TSV file in which the first column is
the source string and the second the target string.

    source   target

One can specify different 1-indexed column indices using arguments to `data:`:

    ...
    data:
      source_col: 2
      target_col: 1
      ...

## Usage

The `neudia` command-line tool uses a subcommand interface, with four different
modes. To see a full set of options available for each subcommand, use the
`--print_config` flag. For example:

    neudia fit --print_config

will show all configuration options (and their default values) for the `fit`
subcommand.

For more detailed examples, see the [`configs`](configs) directory.

### Training (`fit`)

In `fit` mode, one trains a model, either from scratch or optionally, resuming
from a pre-existing checkpoint. Naturally, most configuration options need to be
set at training time.

This mode is invoked using the `fit` subcommand, like so.

    neudia fit --config path/to/config.yaml

Alternatively, one can resume training from a pre-existing checkpoint so long as
it matches the specification of the configuration file.

    neudia fit --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

#### Seeding

Setting the `seed_everything:` argument to some fixed value ensures a
reproducible experiment (modulo hardware non-determinism).

#### Model architecture

A specification for a model determines specific properties of:

-   `dropout` probability
-   the dimensionality of the embeddings (`embedding_size`)
-   the dimensionality of the encoder `hidden_size`
-   `label_smoothing` probability
-   the number of encoder `layers`

These are all specified under `model:`.

#### Optimization

Neudia requires an optimizer and a learning rate scheduler. The system is
borrowed from Yoyodyne; [see here for more
information](https://github.com/CUNY-CL/yoyodyne/blob/master/README.md#optimization).

#### Checkpointing

A checkpoint config must be specified or no checkpoints will be generated; [see
here for more
information](https://github.com/CUNY-CL/yoyodyne/blob/master/README.md#checkpointing).

#### Callbacks

[See here for more
information](https://github.com/CUNY-CL/yoyodyne/blob/master/README.md#callbacks).

#### Logging

[See here for more
information](https://github.com/CUNY-CL/yoyodyne/blob/master/README.md#logging).

#### Other options

Batch size is specified using `data: batch_size: ...`.

By default, training uses 32-bit precision. However, the `trainer: precision:`
flag allows the user to perform training with half precision (`16`), or with
mixed-precision formats like `bf16-mixed` if supported by the accelerator. This
might reduce the size of the model and batches in memory, allowing one to use
larger batches, or it may simply provide small speed-ups.

There are a number of ways to specify how long a model should train for. For
example, the following YAML snippet specifies that training should run for 100
epochs or 6 wall-clock hours, whichever comes first:

    ...
    trainer:
      max_epochs: 100
      max_time: 00:06:00:00
      ...

### Validation (`validate`)

In `validation` mode, one runs the validation step over labeled validation data
(specified as `data: val: path/to/validation.tsv`) using a previously trained
checkpoint (`--ckpt_path path/to/checkpoint.ckpt` from the command line),
recording loss and other statistics for the validation set. In practice this is
mostly useful for debugging.

This mode is invoked using the `validate` subcommand, like so:

    neudia validate --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Evaluation (`test`)

In `test` mode, one computes accuracy over held-out test data (specified as
`data: test: path/to/test.tsv`) using a previously trained checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line); it differs from
validation mode in that it uses the `test` file rather than the `val` file.

This mode is invoked using the `test` subcommand, like so:

    neudia test --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

### Inference (`predict`)

In `predict` mode, a previously trained model checkpoint
(`--ckpt_path path/to/checkpoint.ckpt` from the command line) is used to label
an input file. One must also specify the path where the predictions will be
written:

    ...
    predict:
      path: path/to/predictions.txt
    ...

This mode is invoked using the `predict` subcommand, like so:

    neudia predict --config path/to/config.yaml --ckpt_path path/to/checkpoint.ckpt

## Examples

The [`examples`](examples) directory contains some relevant examples.

## Related projects

-   Neudia is closely based on
    [Yoyodyne](https://github.com/CUNY-CL/yoyodyne/tree/master) and reuses much
    of its core code.

## License

Neudia is distributed under an [Apache 2.0 license](LICENSE.txt).

## For developers

We welcome contributions using the fork-and-pull model.

### Testing

A small integration test diacritizes lines of the
[*Aeneid*](https://en.wikipedia.org/wiki/Aeneid). To run the test, run the
following:

    pytest -vvv tests

### Releasing

We welcome contributions using the fork-and-pull model.

### Releasing

1.  Create a new branch. E.g., if you want to call this branch "release":
    `git checkout -b release`
2.  Sync your fork's branch to the upstream master branch. E.g., if the upstream
    remote is called "upstream": `git pull upstream master`
3.  Increment the version field in [`pyproject.toml`](pyproject.toml).
4.  Stage your changes: `git add pyproject.toml`.
5.  Commit your changes: `git commit -m "your commit message here"`
6.  Push your changes. E.g., if your branch is called "release":
    `git push origin release`
7.  Submit a PR for your release and wait for it to be merged into `master`.
8.  Tag the `master` branch's last commit. The tag should begin with `v`; e.g.,
    if the new version is 3.1.4, the tag should be `v3.1.4`. This can be done:
    -   on GitHub itself: click the "Releases" or "Create a new release" link on
        the right-hand side of the GitHub page) and follow the dialogues.
    -   from the command-line using `git tag`.
9.  Build the new release: `python -m build`
10. Upload the result to PyPI: `twine upload dist/*`

## References

Gershuni, E. and Pinter, Y. 2022. [Restoring Hebrew diacritics without a
dictionary](https://aclanthology.org/2022.findings-naacl.75/). In *Findings of
the Association for Computational Linguistics: NAACL 2022*, pages 1010-1018.
