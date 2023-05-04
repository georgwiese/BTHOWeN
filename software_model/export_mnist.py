#!/usr/bin/env python3

import argparse
import os

import imageio
import numpy as np
from train_swept_models import get_datasets


def export_mnist():    
    _, test_dataset = get_datasets("MNIST")

    print("Storing images")
    os.makedirs("data/MNIST/png", exist_ok=True)
    for i, (img, label) in enumerate(test_dataset):
        name = f"{i:04d}_{label}"
        imageio.v3.imwrite(f"data/MNIST/png/{name}.png", (img.reshape((28, 28)) * 255).astype(np.uint8))
    

def read_arguments():
    parser = argparse.ArgumentParser(description="Converts the MNIST test dataset, and stores the result to data/MNIST/png")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = read_arguments()
    export_mnist()

