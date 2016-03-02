import cPickle as pickle
import os, glob

train_data_path='/data/dsb15_pkl/pkl_train'
validate_data_path='/data/dsb15_pkl/pkl_validate'

_extract_id_from_path = lambda path: int(re.search(r'/(\d+)/', path).group(1))

def _find_patient_folders(root_folder):
    """Finds and sorts all patient folders.
    """
    patient_folder_format = os.path.join(root_folder, "*", "study")
    print str(patient_folder_format)
    patient_folders = glob.glob(patient_folder_format)
    print str(patient_folders)
    patient_folders.sort(key=_extract_id_from_path)
    return patient_folders


def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data

