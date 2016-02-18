import cPickle
import csv
import os
import platform
import pwd
import subprocess
import time
import re
import numpy as np


def get_dir_path(dir_name, root_dir='/mnt/storage/metadata/kaggle-heart'):
    root_dir = '/home/ikorshun/metadata'  # TODO hack
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


def get_patient_id(path):
    return re.search(r'/(\d+)/study', path).group(1)


def get_slice_id(path):
    return re.search(r'/(sax_\d+\.pkl)$', path).group(1)


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
