import time
import platform
import numpy as np
import csv
from scipy.stats import norm
import subprocess
import cPickle
from collections import defaultdict
import os
import pwd


def get_dir_path(dir_name, root_dir='/mnt/storage/metadata/kaggle-heart'):
    root_dir = '/mnt/storage/users/lpigou/kaggle-heart/metadata'  # TODO hack
    username = pwd.getpwuid(os.getuid())[0]
    platform_name = hostname()
    dir_path = root_dir + '/' + dir_name + '/%s-%s' % (username, platform_name)
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)
    return dir_path


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
    with open(path, 'w') as f:
        cPickle.dump(obj, f, protocol=protocol)


def load_pkl(path):
    with open(path) as f:
        obj = cPickle.load(f)
    return obj


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr


def save_submisssion(patient_predictions, submission_path):
    """
    :param patient_predictions: dict of {patient_id: [systole_cdf, diastole_cdf]}
    :param submission_path:
    """
    fi = csv.reader(open('sample_submission_validate.csv'))
    f = open(submission_path, 'w+')
    fo = csv.writer(f, lineterminator='\n')
    fo.writerow(fi.next())
    for line in fi:
        idx = line[0]
        patient_id, target = idx.split('_')
        patient_id = int(patient_id)
        out = [idx]
        if patient_id in patient_predictions.keys():
            if target == 'Diastole':
                out.extend(list(patient_predictions[patient_id][1]))
            else:
                out.extend(list(patient_predictions[patient_id][0]))
        else:
            print 'missed', idx
        fo.writerow(out)
    f.close()


def rmse(predictions, targets):
    """
    :param predictions: (batch_size, 1)
    :param targets: (batch_size, 1)
    :return: RMSE (mean over batch)
    """
    return np.sqrt(np.mean((predictions - targets) ** 2))


def crps(prediction_cdf, target_cdf):
    """
    Use it with batch_size of 1
    :param prediction_cdf: (batch_size, 600)
    :param target_cdf: (batch_size, 600)
    :return: CRPS mean over batch
    """
    return np.mean((prediction_cdf - target_cdf) ** 2)


def real_to_cdf(y, sigma=1e-10):
    cdf = np.zeros((y.shape[0], 600))
    for i in range(y.shape[0]):
        cdf[i] = norm.cdf(np.linspace(0, 599, 600), y[i], sigma)
    return cdf


def heaviside_function(x):
    return np.float32((np.linspace(0, 599, 600) - x) >= 0)


def get_avg_patient_predictions(batch_predictions, batch_patient_ids):
    # TODO where is the best place for this function?
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

        # average predictions over patient's predictions
        for patient_id, patient_idxs in patient2idxs.iteritems():
            # print patient_id, p[patient_idxs]
            prediction_cdfs = heaviside_function(p[patient_idxs])
            avg_prediction_cdf = np.mean(prediction_cdfs, axis=0)
            patient2cdf[patient_id].append(avg_prediction_cdf)

    return patient2cdf
