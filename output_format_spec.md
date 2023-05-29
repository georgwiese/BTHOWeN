# Output Format Spec

This codebase persists models the pickle format.
However, this format is difficult to work with from any programming language other than Python.

For this reason, the `convert_to_hdf5.py` script converts models to a file format based on [HDF5](https://www.hdfgroup.org/solutions/hdf5/).

The file contains the following attributes:
- `num_classes`: The number of output classes.
- `num_inputs`: The number of input pixels, i.e., `width * height` of the images the model was trained on.
- `bits_per_input`: The number of bits used to represent a single pixel.
- `num_filter_inputs`: The number of bits that are sent to one filter. Should be a power of 2.
- `num_filter_entries`: The number of entries of the bloom filter array.
- `num_filter_hashes`: The number of hash functions used for each bloom filter.
- `p`: The prime used in the MishMash hash function (`x^3 \mod p) \mod 2^l`, where `l = ln2(num_filter_inputs) * num_filter_hashes`).
  `p` should be representable in exactly `l + 1` bits.

The file contains three datasets:
- `binarization_thresholds`:
  A shape `(width, height, bits_per_input)` `float32` array containing the binarization thresholds for each pixel, used for the thermometer encoding.
  Thresholds should be sorted, so that the temperature encoding of a pixel can be optained by computing `pixel_intensity >= binarization_thresholds[x, y, :]`.
- `bloom_filter`: A shape `(num_classes, num_filters, num_filter_entries)` `bool` array of bloom filter array, where `num_filters = num_inputs * bits_per_input / num_filter_inputs`.
- `input_order`: A shape `(num_inputs * bits_per_input, )` `uint64` array that describes the permutation of bits that the model should implement.
  To permute a list of `bits`, do `permuted_bits = [bits[i] for i in input_order]`.