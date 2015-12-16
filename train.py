from __future__ import division
import argparse
from configuration import config, set_configuration
import numpy as np
import string
from predict import predict_model
import utils
import cPickle as pickle
import os

def train_model(metadata_path, metadata=None):
    if metadata is None:
        if os.path.isfile(metadata_path):
            metadata = pickle.load(open(open(metadata_path, 'r')))

    print "Build model"
    l_ins, l_out = config().build_model()

    """train, train, train"""

    with open(metadata_path, 'w') as f:
        pickle.dump({
            'metadata_path': metadata_path,
            'configuration_file': config().__name__,
            'git_revision_hash': utils.get_git_revision_hash(),
        }, f, pickle.HIGHEST_PROTOCOL)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    required.add_argument('-c', '--config',
                          help='configuration to run',
                          required=True)
    args = parser.parse_args()
    set_configuration(args.config)
    print config().__name__
    print utils.get_git_revision_hash()
    expid = utils.generate_expid(args.config)
    metadata_path = "metadata/%s.pkl" % expid

    train_model(metadata_path)
    predict_model(metadata_path)


