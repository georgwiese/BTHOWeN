#!/usr/bin/env python3

import argparse
import lzma
import math
import pickle
from pathlib import Path

import h5py
import numpy as np


def convert_model(model_fname):
    print("Loading model")
    with lzma.open(model_fname, "rb") as f:
        state_dict = pickle.load(f)

    model = state_dict["model"]
    info = state_dict["info"]

    suffix = ".pickle.lzma"
    assert model_fname.endswith(suffix)
    output_path = Path(model_fname[:-len(suffix)] + ".hdf5")

    if output_path.exists():
        print("Deleting existing output file")
        output_path.unlink()

    with h5py.File(str(output_path), "w") as f:
        bloom_filters = np.array([
            [
                bloom_filter.data
                for bloom_filter in discriminator.filters
            ]
            for discriminator in model.discriminators
        ])

        assert bloom_filters.min() == 0
        assert bloom_filters.max() == 1
        bloom_filters = bloom_filters.astype(bool)

        expected_shape = (info["num_classes"], int(info["num_inputs"] * info["bits_per_input"] / info["num_filter_inputs"]), info["num_filter_entries"])
        assert bloom_filters.shape == expected_shape

        input_order = model.input_order.astype(np.uint64)

        # Reshape binarization thresholds to have shape (width, height, bits_per_input), assuming quadratic images
        binarization_thresholds = info["binarization_thresholds"]
        width = int(math.sqrt(binarization_thresholds.shape[0]))
        binarization_thresholds = binarization_thresholds.reshape((width, width, binarization_thresholds.shape[1]))

        f.create_dataset("bloom_filters", data=bloom_filters)
        f.create_dataset("input_order", data=input_order)
        f.create_dataset("binarization_thresholds", data=binarization_thresholds)
        for k, v in info.items():
            if k != "binarization_thresholds":
                f.attrs[k] = v

def read_arguments():
    parser = argparse.ArgumentParser(description="Convert a given model to HDF5")
    parser.add_argument("model_fname", help="Path to pretrained model .pickle.lzma")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_arguments()
    convert_model(args.model_fname)

