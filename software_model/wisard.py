#!/usr/bin/false

import math

import numpy as np
from bloom_filter import BloomFilter
from numba import jit
from sympy import isprime


# Converts a vector of booleans to an unsigned integer
#  i.e. (2**0 * xv[0]) + (2**1 * xv[1]) + ... + (2**n * xv[n])
# Inputs:
#  xv: The boolean vector to be converted
# Returns: The unsigned integer representation of xv
@jit(nopython=True, inline='always')
def input_to_value(xv):
    result = 0
    for i in range(xv.size):
        result += xv[i] << i
    return result

def choose_p(num_inputs, num_hashes):

    num_input_bits = math.log2(num_inputs)
    assert num_input_bits.is_integer(), "num_inputs should be a power of 2!"

    num_bits_needed = num_hashes * int(num_input_bits) + 1
    # Note that due to the Bertrand's postulate it is guaranteed that there exists a prime
    # between any x and 2*x for x > 3.
    # thus the condition that p > 2*num_hashes * int(num_input_bits) will be fulfilled.

    for candidate in range(1 << num_bits_needed, 1 << (num_bits_needed - 1), -1):
        if isprime(candidate):
            return candidate

# Implementes a single discriminator in the WiSARD model
# A discriminator is a collection of boolean LUTs with associated input sets
# During inference, the outputs of all LUTs are summed to produce a response
class Discriminator:
    # Constructor
    # Inputs:
    #  num_inputs:    The total number of inputs to the discriminator
    #  unit_inputs:   The number of boolean inputs to each LUT/filter in the discriminator
    #  unit_entries:  The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:   The number of hash functions for each filter.
    #  p:             Prime to use for the MishMash hash. Should be larger than unit_entries^unit_hashes
    def __init__(self, num_inputs, unit_inputs, unit_entries, unit_hashes, p):
        assert((num_inputs/unit_inputs).is_integer())
        self.num_filters = num_inputs // unit_inputs
        self.filters = [BloomFilter(unit_inputs, unit_entries, unit_hashes, p) for i in range(self.num_filters)]

    # Performs a training step (updating filter values)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
        for idx, inp in enumerate(filter_inputs):
            self.filters[idx].add_member(inp)

    # Performs an inference to generate a response (number of filters which return True)
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: The response of the discriminator to the input
    def predict(self, xv):
        filter_inputs = xv.reshape(self.num_filters, -1) # Divide the inputs between the filters
        response = 0
        for idx, inp in enumerate(filter_inputs):
            response += int(self.filters[idx].check_membership(inp))
        return response
    
    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for f in self.filters:
            f.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for f in self.filters:
            f.binarize()
    
# Top-level class for the WiSARD weightless neural network model
class WiSARD:
    # Constructor
    # Inputs:
    #  num_inputs:       The total number of inputs to the model
    #  num_classes:      The number of distinct possible outputs of the model; the number of classes in the dataset
    #  unit_inputs:      The number of boolean inputs to each LUT/filter in the model
    #  unit_entries:     The size of the underlying storage arrays for the filters. Must be a power of two.
    #  unit_hashes:      The number of hash functions for each filter.
    def __init__(self, num_inputs, num_classes, unit_inputs, unit_entries, unit_hashes):
        self.pad_zeros = (((num_inputs // unit_inputs) * unit_inputs) - num_inputs) % unit_inputs
        pad_inputs = num_inputs + self.pad_zeros
        self.input_order = np.arange(pad_inputs) # Use each input exactly once
        np.random.shuffle(self.input_order) # Randomize the ordering of the inputs
        self.p = choose_p(unit_entries, unit_hashes)
        self.discriminators = [Discriminator(self.input_order.size, unit_inputs, unit_entries, unit_hashes, self.p) for i in range(num_classes)]

    # Performs a training step (updating filter values) for all discriminators
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    def train(self, xv, label):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order] # Reorder input
        self.discriminators[label].train(xv)

    # Performs an inference with the provided input
    # Passes the input through all discriminators, and returns the one or more with the maximal response
    # Inputs:
    #  xv: A vector of boolean values representing the input sample
    # Returns: A vector containing the indices of the discriminators with maximal response
    def predict(self, xv):
        xv = np.pad(xv, (0, self.pad_zeros))[self.input_order] # Reorder input
        responses = np.array([d.predict(xv) for d in self.discriminators], dtype=int)
        max_response = responses.max()
        return np.where(responses == max_response)[0]

    # Sets the bleaching value for all filters
    # See the BloomFilter implementation for more information on what this means
    # Inputs:
    #  bleach: The new bleaching value to set
    def set_bleaching(self, bleach):
        for d in self.discriminators:
            d.set_bleaching(bleach)

    # Binarizes all filters; this process is irreversible
    # See the BloomFilter implementation for more information on what this means
    def binarize(self):
        for d in self.discriminators:
            d.binarize()
   
