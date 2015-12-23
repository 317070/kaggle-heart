import numpy as np
import re
from configuration import config

def uint_to_float(img):
    return 1 - (img / np.float32(255.0))

def preprocess(chunk_x, img, chunk_y, lbl):
    chunk_x[:, :] = uint_to_float(img)
    chunk_y[:, :] = lbl.astype(np.float32)