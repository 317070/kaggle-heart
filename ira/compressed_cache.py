"""Module implementing a compressed cache.

It provides a decorator that caches function results. If the function returns
a numpy array, it is stored in a compressed way.
Average compression rate for single patients (excluding metadata): 2.91
"""

import functools
import tempfile
import os
import sys

import blz
import numpy as np
import collections

import configuration
_config = configuration.config


class CompressedCache(dict):
    """Implements a cache that compresses numpy Arrays."""

    def __init__(self):
        super(CompressedCache, self).__init__()
  
    def __setitem__(self, key, value):
        # Check config file
        location = (
            _config().caching if hasattr(_config(), 'caching') else None)
        assert location in ('disk', 'memory')
         
        # Use tempfolder for saving on disk   
        save_folder = tempfile.mkdtemp() if location == 'disk' else None
        self.compress = lambda v: blz.barray(v, rootdir=save_folder)

        if type(value) == np.ndarray:
            value = self.compress(value)
        return super(CompressedCache, self).__setitem__(key, value)
    
    def __getitem__(self, key):
        value = super(CompressedCache, self).__getitem__(key)
        if value and type(value) == blz.blz_ext.barray:
          value = value[:]
        return value
    

def memoize():
    """Creates a memoize decorator that caches results in the given location.
    """
    def actual_memoize_decorator(obj):

        cache = obj.cache = CompressedCache()
     
        @functools.wraps(obj)
        def memoizer(*args, **kwargs):
            
            # Check config file 
            cache_location = (
                _config().caching if hasattr(_config(), 'caching') else None)
            if not cache_location in (None, 'disk', 'memory'):
                raise ValueError(
                    "location should be either None, 'disk' or 'memory'")
            if cache_location == 'disk':
                raise NotImplementedError('Saving on disk not yet supported')

            if cache_location is None:
                # No caching
                return obj(*args, **kwargs)
            else:
                # Do caching
                key = str(args) + str(kwargs)
                if key not in cache:
                    cache[key] = obj(*args, **kwargs)
                return cache[key]
        return memoizer
    return actual_memoize_decorator
