import argparse
import glob
import re
import cPickle as pickle

from dicom.sequence import Sequence

from log import print_to_file
from paths import LOGS_PATH, TRAIN_DATA_PATH, TEST_DATA_PATH


def read_slice(path):
    return pickle.load(open(path))['data']


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


def clean_metadata(metadatadict):
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
    return metadatadict


def read_metadata(path):
    d = pickle.load(open(path))['metadata'][0]
    metadata = clean_metadata(d)
    return metadata


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + r'/*.pkl'),
                    key=lambda x: int(re.search(r'/*_(\d+)\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(((4ch)|(2ch)|(sax))_\d+\.pkl)$', s).group(1)
        metadata = read_metadata(s)
        patient_data.append({'metadata': metadata,
                             'slice_id': slice_id})
        print slice_id
    return patient_data, pid


def get_metadata(data_path):
    patient_paths = sorted(glob.glob(data_path + '*/study'))
    metadata_dict = {}
    for p in patient_paths:
        patient_data, pid = get_patient_data(p)
        print "patient", pid
        metadata_dict[pid] = dict()
        for pd in patient_data:
            metadata_dict[pid][pd['slice_id']] = pd['metadata']

    filename = data_path.split('/')[-2] + '_metadata.pkl'
    with open(filename, 'w') as f:
        pickle.dump(metadata_dict, f)
    print 'saved to ', filename
    return metadata_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    #required.add_argument('-c', '--config',
    #                      help='configuration to run',
    #                      required=True)
    args = parser.parse_args()

    data_paths = [TRAIN_DATA_PATH, TEST_DATA_PATH]
    log_path = LOGS_PATH + "generate_metadata.log"
    with print_to_file(log_path):
        for d in data_paths:
            get_metadata(d)
        print "log saved to '%s'" % log_path

