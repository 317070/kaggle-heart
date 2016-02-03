import time
import platform
import numpy as np
import gzip
from scipy.special import erf
import theano
import theano.tensor as T
from compressed_cache import simple_memoized

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


def theano_mu_sigma_erf(mu_erf, sigma_erf):
    eps = 1e-7
    x_axis = theano.shared(np.arange(0, 600, dtype='float32')).dimshuffle('x',0)
    sigma_erf = T.clip(sigma_erf.dimshuffle('x','x'), eps, 1)
    x = (x_axis - mu_erf.dimshuffle(0,'x')) / (sigma_erf * np.sqrt(2).astype('float32'))
    return (T.erf(x) + 1)/2


def numpy_mu_sigma_erf(mu_erf, sigma_erf):
    eps = 1e-7
    batch_size = mu_erf.shape[0]
    x_axis = np.tile(np.arange(0, 600, dtype='float32'), (batch_size, 1))
    mu_erf = np.tile(mu_erf[:,None], (1, 600))
    sigma_erf = np.tile(sigma_erf[:,None], (1, 600)) + 1e-7

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


def CRSP(distribution, value):
    target = np.zeros( (600,) , dtype='float32')
    target[int(np.ceil(value)):] = 1  # don't forget to ceil!
    return np.mean( (distribution - target)**2 )