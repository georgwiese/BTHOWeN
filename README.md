# BTHOWeN-zero-g

This is a fork of [ZSusskind/BTHOWeN](https://github.com/ZSusskind/BTHOWeN) with some slight modifications to make it more friendly for zero-knowledge proofs.
See [their readme](https://github.com/ZSusskind/BTHOWeN/blob/master/README.md) for background on the models and the [readme of the main `zero_g` repository](https://github.com/zkp-gravity/zero_g/blob/main/README.md) on how to create proofs!

# Usage
## Prerequisites

Our codebase was written for Python 3.8.10; other version may very well work but are untested.

We recommend constructing a virtual environment for dependency management:
```
python3 -m venv env
source env/bin/activate
```

From here, dependency installation can be automatically handled with a single command:
```
pip install -r requirements.txt
```

## Creating BTHOWeN Models
All relevant code lives in the `software_model/` directory. Natively supported datasets are MNIST, Ecoli, Iris, Letter, Satimage, Shuttle, Vehicle, Vowel, and Wine.

`train_swept_models.py` is the primary script for programmatic model sweeping. It allows for specification of Bloom filter and encoding parameters; run with `--help` for more details.  
Example usage: `./train_swept_models.py MNIST --filter_inputs 28 --filter_entries 1024 --filter_hashes 2 --bits_per_input 2`  
`--filter_inputs`, `--filter_entries`, `--filter_hashes`, and `--bits_per_input` can all be provided with multiple values, in which case all permutations are tried.  
Run-to-run variation in accuracy is expected, particularly on small models. This is largely a result of the random input mapping.  
*Note*: Dataset names are not case-sensitive

`evaluate.py` runs inference on a pre-trained model - invocation takes the form `./evaluate.py <model_fname> <dset_name>`.

`convert_to_hdf5.py` converts a trained model to an HDF5 file. This is necessary to use the model in the [`zero_g`](https://github.com/zkp-gravity/zero_g) repository.

`export_mnist.py` writes the MNIST test dataset to disc in PNG format.