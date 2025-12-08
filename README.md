# Neudia ✒️

Neudia is a neural network-based diacritization system.

## Philosophy

Neudia is closely inspired by the Nakdimon diacritization system for Hebrew (Gershuni & Pinter 2022), but is intended to be much more general.

## Design

The Neudia model consists of a randomly initialized bidirectional LSTM which feeds into a tagger layer.

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

Neudia was created by [Kyle Gorman](https://wellformedness.com) and [opther contributors](https://github.com/CUNY-CL/neudia/graphs/contributors) likeyoyodyne yourself.

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

## Usage

The `yoyodyne` command-line tool uses a subcommand interface, with four
different modes. To see a full set of options available for each subcommand, use
the `--print_config` flag. For example:

    yoyodyne fit --print_config

will show all configuration options (and their default values) for the `fit`
subcommand.

For more detailed examples, see the [`configs`](configs) directory.

### Training (`fit`)

FIXME

### Validation (`validate`)

FIXME

### Evaluation (`test`)

FIXME

### Inference (`predict`)

FIXME

## Examples

FIXME

## License

Neudia is distributed under an [Apache 2.0 license](LICENSE.txt).

## For developers

We welcome contributions using the fork-and-pull model.

### Testing

FIXME

### Releasing

FIXME

## References

Gershuni, E. and Pinter, Y. 2022. Restoring Hebrew diacritics without a dictionary. In _Findings of the Association for Computational Linguistics: NAACL 2022_, pages 1010-1018.
