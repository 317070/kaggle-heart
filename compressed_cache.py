"""Module implementing a compressed cache.

It provides a decorator that caches function results. If the function returns
a numpy array, it is stored in a compressed way.
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
        uncompressed_cache = obj.uncompressed_cache = dict()
     
        @functools.wraps(obj)
        def memoizer(*args, **kwargs):
            
            # Check config file 
            cache_location = (
                _config().caching if hasattr(_config(), 'caching') else None)
            # Check validity
            possible_args = (None, 'disk', 'memory', 'uncompressed')
            if not cache_location in possible_args:
                raise ValueError(
                    "location should be in", possible_args )
            if cache_location == 'disk':
                raise NotImplementedError('Saving on disk not yet supported')

            if cache_location is None:
                # No caching
                return obj(*args, **kwargs)
            else:
                # Do caching
                cache_to_use = cache if cache_location == 'memory' else uncompressed_cache
                key = str(args) + str(kwargs)
                if key not in cache_to_use:
#                    print "[Memoize] Caching call of %s" % str(obj)
                    cache_to_use[key] = obj(*args, **kwargs)
                return cache_to_use[key]
        return memoizer
    return actual_memoize_decorator


class simple_memoized(object):
   '''Decorator. Caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned
   (not reevaluated).
   '''
   def __init__(self, func):
      self.func = func
      self.cache = {}

   def __call__(self, *args):
      if not isinstance(args, collections.Hashable):
         # uncacheable. a list, for instance.
         # better to not cache than blow up.
         return self.func(*args)
      if args in self.cache:
         return self.cache[args]
      else:
         value = self.func(*args)
         self.cache[args] = value
         return value

   def __get__(self, obj, objtype):
      '''Support instance methods.'''
      return functools.partial(self.__call__, obj)