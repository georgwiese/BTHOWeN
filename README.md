# BTHOWeN-0g

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

### Training

`train_swept_models.py` is the primary script for programmatic model sweeping. It allows for specification of Bloom filter and encoding parameters; run with `--help` for more details.  `--filter_inputs`, `--filter_entries`, `--filter_hashes`, and `--bits_per_input` can all be provided with multiple values, in which case all permutations are tried.  
Run-to-run variation in accuracy is expected, particularly on small models. This is largely a result of the random input mapping.  

Example usage: 
```bash
    ./train_swept_models.py MNIST --filter_inputs 28 --filter_entries 1024 --filter_hashes 2 --bits_per_input 2
```  
*Note*: Dataset names are not case-sensitive

### Evaluation

`evaluate.py` runs inference on a pre-trained model.

Usage:
```bash
    ./evaluate.py <model_fname> <dset_name>
```

Here `<model_fname>` is a `lzma` model file, and `<dset_name>` is the dataset name. 
*Note*: Dataset names are not case-sensitive

### Serialization

`convert_to_hdf5.py` converts a trained model to an HDF5 file. This is necessary to use the model in the [`zero_g`](https://github.com/zkp-gravity/zero_g) repository. 

Usage:
```bash
    ./convert_to_hdf5.py <model_fname>
```

See [`output_format_spec.md`](output_format_spec.md) for a specification of the file format.

### Exporting MNIST

`export_mnist.py` writes the MNIST test dataset to disc in PNG format.

Usage:
```bash
    ./export_mnist.py
```
