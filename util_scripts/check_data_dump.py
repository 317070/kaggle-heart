import glob
import numpy as np
import re
from configuration import config
import cPickle as pickle
import utils
from validation_set import get_cross_validation_indices
import random
import matplotlib.pyplot as plt
import cPickle as pickle

dump_files = sorted(glob.glob("/home/jonas/kip/kaggle-heart/data_loader_dump*.pkl"))

for file in dump_files[:1]:
    print "opening %s" % file
    d = pickle.load(open(file, "r"))
    print d["output"].keys()
    print d["output"]['diastole:value']
    print d["input"]['sliced:data:singleslice:difference'].shape
