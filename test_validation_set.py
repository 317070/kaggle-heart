"""Unittest for making sure the validation split remains the same.
"""

import cPickle as pickle
import os
import unittest

import validation_set


_TEST_SEED = 317070
_TEST_NO_SPLITS = 6
_TEST_INDICES = range(1, 501)


_DEFAULT_SPLIT_RESULT_FILE = os.path.join(
    "unittest_targets", "default_xval_split.pkl")


class TestValidationSplit(unittest.TestCase):

    def test_default_split(self):
        # Create the cross
        splits = [
            validation_set.get_cross_validation_indices(
                _TEST_INDICES, i, _TEST_NO_SPLITS, _TEST_SEED)
            for i in xrange(_TEST_NO_SPLITS)]
        with open(_DEFAULT_SPLIT_RESULT_FILE, 'r') as f:
            target_splits = pickle.load(f)
        for gen, target in zip(splits, target_splits):
          self.assertEqual(gen, target)

if __name__ == '__main__':
    unittest.main()
