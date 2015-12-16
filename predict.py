from __future__ import division
import argparse
from configuration import config, set_configuration
import numpy as np
import string
import utils


def predict_model(metadata_path, metadata=None):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)

    required.add_argument('-m', '--model',
                          help='model file with metadata',
                          required=True)

    args = parser.parse_args()
    set_configuration(args.config)

    predict_model(args.model)
