# Generating data via diacritic stripping

The [`strip.py`](strip.py) script can be used to remove Unicode diacritics,
producing two-column TSV files which can be used as training and/or validation
data.

## Usage

By default, `strip.py` uses NFC/NFD (de)composition to identify diacritics:

    ./strip.py path/to/source.txt path/to/sink.tsv

With the `--compatibility` flag, it'll instead use NFKC/NFKD (de)composition:

    ./strip.py --compatibility path/to/source.txt path/to/sink.tsv

For even more aggressive stripping, one can use the "Uninames" method introduced
by Náplava et al. (2018), which maps characters using their Unicode names, and
thus identifies many look-alike characters that Unicode does not consider to be
formally related.

    ./strip.py --method uninames path/to/source.txt path/to/sink.tsv

## References

Náplava, J., Straka, M., Straňák, P., and Hajič, J. 2018. Diacritic restoration
using neural networks. In *Proceedings of the Eleventh International Conference
on Language Resources and Evaluation*, pages 1566-1573.
