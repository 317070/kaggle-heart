import cPickle
import csv
import os
import platform
import pwd
import subprocess
import time
import re
import numpy as np
import dicom2pkl
import create_validation_split
import glob

maxfloat = np.finfo(np.float32).max


def find_model_metadata(metadata_dir, config_name):
    metadata_paths = glob.glob(metadata_dir + '/%s-*' % config_name)
    if not metadata_paths:
        raise ValueError('No metadata files for config %s' % config_name)
    elif len(metadata_paths) > 1:
        raise ValueError('Multiple metadata files for config %s' % config_name)
    print metadata_paths[0]
    return metadata_paths[0]


def get_train_valid_split(train_data_path):
    from pathfinder import SUBMISSION_NR
    if SUBMISSION_NR == 1:
        filename = 'valid_split.pkl'
        if not os.path.isfile(filename):
            print 'Making validation split'
            create_validation_split.save_train_validation_ids(filename, train_data_path)
        return load_pkl(filename)
    else:
        return {'train': None, 'valid': [1]}


def check_data_paths(data_path, pkl_data_path):
    if not os.path.isdir(data_path):
        raise ValueError('wrong path to DICOM data')
    if not os.path.isdir(pkl_data_path):
        print ' converting DICOM to pkl'
        dicom2pkl.preprocess(data_path, pkl_data_path)
        print ' Saved in', pkl_data_path


def get_dir_path(dir_name, root_dir):
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


def copy(from_folder, to_folder):
    command = "cp -r %s %s/." % (from_folder, to_folder)
    print command
    os.system(command)


def get_patient_id(path):
    return re.search(r'/(\d+)/study', path).group(1)


def get_slice_id(path):
    return re.search(r'/((sax|2ch|4ch)_\d+\.pkl)$', path).group(1)


def get_patient_age(s):
    age = float(s[:-1])
    units = s[-1]
    if units == 'M':
        age /= 12.
    elif units == 'W':
        age /= 52.1429
    return age


def current_learning_rate(schedule, idx):
    s = schedule.keys()
    s.sort()
    current_lr = schedule[0]
    for i in s:
        if idx >= i:
            current_lr = schedule[i]

    return current_lr


def save_submission(patient_predictions, submission_path):
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
