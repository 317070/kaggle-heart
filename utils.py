import time
import platform
import numpy as np
import gzip
from scipy.stats import norm


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
    return "%s-%s-0" % (arch_name, hostname())


def get_git_revision_hash():
    import subprocess
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()


def load_gz(path):  # load a .npy.gz file
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


def rmse(predictions, targets):
    return np.sqrt(np.mean((predictions - targets) ** 2))


def real_to_cdf(y, sigma):
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf


def heaviside_function(y):
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = np.float32((np.linspace(0, 599, 600) - y[i]) >= 0)
    return cdf


def crps(predictions, targets, sigma):
    predictions_cdf = real_to_cdf(predictions, sigma)
    target_cdf = heaviside_function(targets)
    return np.mean((predictions_cdf - target_cdf) ** 2)
