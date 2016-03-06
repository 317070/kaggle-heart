import utils
import sys
import numpy as np
import utils_heart

# from pathfinder import MODEL_PATH

# if len(sys.argv) < 3:
#     sys.exit("Usage: fix.py <s1> <s2>")
#
# submission_name1 = sys.argv[1]
# submission_name2 = sys.argv[2]
# submission_path1 = 'final_submission-all_except_ira_00115_fixed.csv'
# submission_path2 = 'final_submission-1457129444.05.csv'
submission_path_meta = 'submissions/meta_ira_6.csv'
submission_path_ss = 'submissions/ss_final_submission-1457220052.92.csv'
submission_path_mix = 'submissions/mix_final_submission-1457218324.95.csv'
fixed_submission_path = 'submissions/fixed_submission_ira6_92.95.csv'


# submissions paths
# submission_dir = utils.get_dir_path('submissions', MODEL_PATH)
# submission_path1 = submission_dir + '/' + submission_name1
# submission_path2 = submission_dir + '/' + submission_name2

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


threshold = 0.04

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

utils.save_submission(fixed_predictions, fixed_submission_path)
