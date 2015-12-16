import time
import platform
import numpy as np
import gzip


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
    return "%s-%s-%s" % (arch_name, hostname(), timestamp())


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