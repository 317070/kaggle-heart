
import gzip
import platform
import random
import time

import numpy as np
# TODO: remove this import, it is annoying for a utils file
import theano
import theano.tensor as T

from dicom.sequence import Sequence
from scipy.special import erf

from compressed_cache import simple_memoized


maxfloat = np.finfo(np.float32).max


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def timestamp():
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def hostname():
    return platform.node()


def generate_expid(arch_name):
    # expid shouldn't matter on anything else than configuration name.
    # Configurations need to be deterministic!
    return "%s" % (arch_name, )

@simple_memoized  # solves memory crash
def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)

    predictions = np.argmax(y, axis=1)
    return np.mean(predictions == t)


def softmax(x):
    m = np.max(x, axis=1, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=1, keepdims=True)


def entropy(x):
    h = -x * np.log(x)
    h[np.invert(np.isfinite(h))] = 0
    return h.sum(1)


def load_gz(path): # load a .npy.gz file
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
        return np.load(f)
    else:
        return np.load(path)

def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr

def segmentation_log_loss(outputs, labels):
    eps=1e-15
    outputs = np.clip(outputs, eps, 1 - eps)
    result = - np.mean(labels * np.log(outputs) + (1-labels) * np.log(1-outputs), axis=(0,1,2))
    return result

def segmentation_accuracy(outputs, labels):
    eps=1e-15
    outputs = (outputs > 0.5).astype('int8')
    result = np.mean(labels * outputs + (1-labels) * (1-outputs), axis=(0,1,2))
    return result

def segmentation_visualization(outputs, labels):
    print outputs[0]
    print labels[0]

def merge(a, b, path=None):
    "merges dict b into dict a"
    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            else:
                raise Exception('Conflict at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def detect_nans(loss, xs_shared, ys_shared, all_params):
    if np.isnan(loss):
        print "NaN Detected."
        #if not np.isfinite(g_n): print "Nan in gradients detected"
        for p in all_params:
            if not np.isfinite(p.get_value()).all():
                print "Nan detected in", p.name
        for k, v in xs_shared.iteritems():
            if not np.isfinite(v).all():
                print "Nan detected in loaded data: %s"%k
        for k, v in ys_shared.iteritems():
            if not np.isfinite(v).all():
                print "Nan detected in loaded data: %s"%k


def theano_mu_sigma_erf(mu_erf, sigma_erf, eps=1e-7):
    x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
    if sigma_erf.ndim==0:
        sigma_erf = T.clip(sigma_erf.dimshuffle('x','x'), eps, 1)
    elif sigma_erf.ndim==1:
        sigma_erf = T.clip(sigma_erf.dimshuffle(0,'x'), eps, 1)
    x = (x_axis - mu_erf.dimshuffle(0,'x')) / (sigma_erf * np.sqrt(2).astype('float32'))
    return (T.erf(x) + 1)/2


def numpy_mu_sigma_erf(mu_erf, sigma_erf, eps=1e-7):
    batch_size = mu_erf.shape[0]
    x_axis = np.tile(np.arange(0, 600, dtype='float32'), (batch_size, 1))
    mu_erf = np.tile(mu_erf[:,None], (1, 600))
    sigma_erf = np.tile(sigma_erf[:,None], (1, 600))
    sigma_erf += eps

    x = (x_axis - mu_erf) / (sigma_erf * np.sqrt(2))
    return (erf(x) + 1)/2


def linear_weighted(value):
    """
    create a (600, ) array which is linear weighted around the desired value
    :param value:
    :return:
    """
    n = np.arange(600, dtype='float32')
    dist = np.abs(n-value)
    normed = dist / np.mean(dist)
    return normed


def cumulative_one_hot(value):
    target = np.zeros( (600,) , dtype='float32')
    target[int(np.ceil(value)):] = 1  # don't forget to ceil!
    return target


def CRSP(distribution, value):
    return np.mean( (distribution - cumulative_one_hot(value))**2 )


def convert_to_number(value):
    value = str(value)
    try:
        if "." in value:
            return float(value)
        else:
            return int(value)
    except:
        pass
    return value


METADATA_CLEAN_TAG = 'META_CLEANED'
def _is_clean(metadatadict):
    return metadatadict.get(METADATA_CLEAN_TAG, False)


def _tag_clean(metadatadict, is_cleaned=True):
    metadatadict[METADATA_CLEAN_TAG] = is_cleaned


def clean_metadata(metadatadict):
    # Check if already cleaned
    if _is_clean(metadatadict):
        return metadatadict
    # Do cleaning
    keys = sorted(list(metadatadict.keys()))
    for key in keys:
        value = metadatadict[key]
        if key == 'PatientAge':
            metadatadict[key] = int(value[:-1])
        if key == 'PatientSex':
            metadatadict[key] = 1 if value == 'F' else -1
        else:
            if isinstance(value, Sequence):
                #convert to list
                value = [i for i in value]
            if isinstance(value, (list,)):
                metadatadict[key] = [convert_to_number(i) for i in value]
            else:
                metadatadict[key] = convert_to_number(value)
    _tag_clean(metadatadict)
    return metadatadict



def norm_geometric_average(x, weights=None, eps=1e-7):
    """Computes the geometric average over the first dimension of a matrix.
    """
    # Convert to log domain
    x_log = np.log(x + eps)
    # Compute the mean
    geom_av_log = np.average(x_log, weights=weights, axis=0)
    # Go back to normal domain and renormalise
    geom_av_log = geom_av_log - np.max(geom_av_log)
    geom_av = np.exp(geom_av_log)
    return geom_av / geom_av.sum()


def geometric_average(x, eps=1e-7):
    """Computes the geometric average over the first dimension of a matrix.
    """
    # Convert to log domain
    x_log = np.log(x+eps)
    # Compute the mean
    geom_av_log = np.mean(x_log, axis=0)
    # Go back to normal domain and renormalise
    geom_av_log = geom_av_log
    geom_av = np.exp(geom_av_log)
    return geom_av


def norm_prod(x, eps=1e-7):
    """Computes the product and renormalises over the first dimension of a matrix.
    """
    # Convert to log domain
    x_log = np.log(x + eps)
    # Compute the mean
    geom_sum_log = np.sum(x_log, axis=0)
    # Go back to normal domain and renormalise
    geom_sum_log = geom_sum_log - np.max(geom_sum_log)
    geom_sum = np.exp(geom_sum_log)
    return geom_sum / geom_sum.sum()


def prod(x):
    """Computes the product and renormalises over the first dimension of a matrix.
    """
    print 'prodding'
    # Convert to log domain
    x_log = np.log(x)
    # Compute the mean
    geom_sum_log = np.sum(x_log, axis=0)
    # Go back to normal domain and renormalise
    geom_sum_log = geom_sum_log
    geom_sum = np.exp(geom_sum_log)
    return geom_sum


def cdf_to_pdf(x):
    if x.ndim==1:
        res = np.diff(x, axis=0)
        return np.hstack([x[:1], res])
    elif x.ndim==2:
        res = np.diff(x, axis=1)
        return np.hstack([x[:, :1], res])
    else:
        return np.apply_along_axis(cdf_to_pdf, axis=-1, arr=x)


def pdf_to_cdf(x):
    return np.cumsum(x, axis=1)


def merge_dicts(dicts):
    res = {}
    for d in dicts:
        res.update(d)
    return res


def pick_random(arr, no_picks):
    """Randomly selects elements.

    If there are not enough elements, repetition is allowed.
    """
    # Expand untill there is enough repetition
    arr_to_pick_from = arr
    while len(arr_to_pick_from) < no_picks:
        arr_to_pick_from += arr
    # Pick
    random.shuffle(arr_to_pick_from)
    return arr_to_pick_from[:no_picks]


import scipy.ndimage.interpolation

def zoom_array(array, zoom_factor):
    result = np.ones(array.shape)
    zoom = [1.0]*array.ndim
    zoom[-1] = zoom_factor
    zr = scipy.ndimage.interpolation.zoom(array,
                                          zoom,
                                          order=3,
                                          mode='nearest',
                                          prefilter=True)
    result[...,:min(zr.shape[-1],array.shape[-1])] = zr[...,:min(zr.shape[-1],array.shape[-1])]
    return result
