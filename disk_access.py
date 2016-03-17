"""Module responsible for disk access.

This module supports loading data and metadata from a given file path. 
This data can be compressed and cached in ram or on the disk, depending on
the configuration file.
"""

import cPickle as pickle

import compressed_cache

@compressed_cache.memoize()
def load_data_from_file(folder):
    return pickle.load(open(folder, "r"))['data'].astype('float32')

@compressed_cache.simple_memoized
def load_metadata_from_file(folder):
    return pickle.load(open(folder, "r"))['metadata']