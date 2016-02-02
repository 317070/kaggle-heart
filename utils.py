import time
import platform
import numpy as np
import gzip
from scipy.stats import norm
import subprocess
import cPickle
from collections import defaultdict


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
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip()
    except:
        return 0


def save_pkl(obj, path, protocol=cPickle.HIGHEST_PROTOCOL):
    with open(path, 'wb') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with open(path, 'rb') as f:
        obj = cPickle.load(f)
    return obj


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


def crps(prediction_cdf, target_cdf):
    return np.mean((prediction_cdf - target_cdf) ** 2)


def real_to_cdf(y, sigma=1e-10):
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf


def heaviside_function(x):
    return np.float32((np.linspace(0, 599, 600) - x) >= 0)


def get_avg_patient_predictions(batch_predictions, batch_patient_ids):
    nbatches = len(batch_predictions)
    npredictions = len(batch_predictions[0])

    patient_ids = []
    for i in xrange(nbatches):
        patient_ids += batch_patient_ids[i]

    patient2idxs = defaultdict(list)
    for i, pid in enumerate(patient_ids):
        patient2idxs[pid].append(i)

    patient2cdf = defaultdict(list)  # list[0] -systole cdf, list[1] - diastole cdf
    for i in xrange(npredictions):
        # collect predictions over batches
        p = []
        for j in xrange(nbatches):
            p.append(batch_predictions[j][i])
        p = np.vstack(p)

        # average predictions over patient's predeictions
        for patient_id, patient_idxs in patient2idxs.iteritems():
            prediction_cdfs = heaviside_function(p[patient_idxs])
            avg_prediction_cdf = np.mean(prediction_cdfs, axis=0)
            patient2cdf[patient_id].append(avg_prediction_cdf)

    return patient2cdf
