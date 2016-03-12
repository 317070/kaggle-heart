import utils
import numpy as np
import utils_heart
from pathfinder import SUBMISSION_PATH


def read_submission(file_path):
    pid2cdf0, pid2cdf1 = {}, {}
    csv_file = open(file_path)
    lines = csv_file.readlines()
    for l in lines[1:]:
        row = l.split(',')
        pid, target = row[0].split('_')
        cdf_string = row[1:]
        if all(x == '' for x in cdf_string[:-1]) or all(x == 'nan' for x in cdf_string[:-1]):
            cdf = None
        else:
            cdf = np.float64(cdf_string)
        pid = int(pid)
        if target == 'Systole':
            pid2cdf0[pid] = cdf
        if target == 'Diastole':
            pid2cdf1[pid] = cdf

    return pid2cdf0, pid2cdf1


def fix_submission(submission_path_meta, submission_path_ss, submission_path_mix, threshold=0.03):
    pid2cdf0_meta, pid2cdf1_meta = read_submission(submission_path_meta)
    pid2cdf0_ss, pid2cdf1_ss = read_submission(submission_path_ss)
    pid2cdf0_mix, pid2cdf1_mix = read_submission(submission_path_mix)
    pid2cdf0_fixed, pid2cdf1_fixed = {}, {}

    patient_ids = pid2cdf0_meta.keys()
    assert patient_ids == pid2cdf0_ss.keys()
    assert patient_ids == pid2cdf0_mix.keys()

    # no prediction from metamodel -> take prediction from slice model
    for pid in patient_ids:
        pid2cdf0_fixed[pid] = pid2cdf0_ss[pid] if pid2cdf0_meta[pid] is None else pid2cdf0_meta[pid]
        pid2cdf1_fixed[pid] = pid2cdf1_ss[pid] if pid2cdf1_meta[pid] is None else pid2cdf1_meta[pid]

    # metamodel disagrees with slice model -> take prediction from mixture ensemble
    for pid in patient_ids:
        crps0 = utils_heart.crps(pid2cdf0_fixed[pid], pid2cdf0_ss[pid])
        if crps0 > threshold:
            print 'sys', pid, crps0
        pid2cdf0_fixed[pid] = pid2cdf0_mix[pid] if crps0 > threshold else pid2cdf0_fixed[pid]

        crps1 = utils_heart.crps(pid2cdf1_fixed[pid], pid2cdf1_ss[pid])
        if crps1 > threshold:
            print 'dst', pid, crps1
        pid2cdf1_fixed[pid] = pid2cdf1_mix[pid] if crps1 > threshold else pid2cdf1_fixed[pid]

    fixed_predictions = {}
    for pid in patient_ids:
        fixed_predictions[pid] = [pid2cdf0_fixed[pid], pid2cdf1_fixed[pid]]

    meta_expid = submission_path_meta.split('-')[-1].replace('.csv', '')
    ss_expid = submission_path_ss.split('-')[-1].replace('.csv', '')
    mix_expid = submission_path_mix.split('-')[-1].replace('.csv', '')
    fixed_submission_path = SUBMISSION_PATH + 'ira_%s-%s-%s.csv' % (meta_expid, ss_expid, mix_expid)
    utils.save_submission(fixed_predictions, fixed_submission_path)
    print 'Submission save to', fixed_submission_path


if __name__ == '__main__':
    # TODO here I will fill my final ensemble
    submission_path_meta = SUBMISSION_PATH + 'final_submission-1457786956.36.csv'
    submission_path_ss = SUBMISSION_PATH + 'final_submission-1457790522.0.csv'
    submission_path_mix = SUBMISSION_PATH + 'final_submission-1457794389.54.csv'

    fix_submission(submission_path_meta, submission_path_ss, submission_path_mix)
