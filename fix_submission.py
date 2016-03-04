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
submission_path2 = 'final_submission-1457129444.05.csv'
submission_path1 = 'final_submission-1456939840.96.csv'



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
        cdf = np.float64(row[1:])
        pid = int(pid)
        if target == 'Systole':
            pid2cdf0[pid] = cdf
        if target == 'Diastole':
            pid2cdf1[pid] = cdf

    return pid2cdf0, pid2cdf1


pid2cdf0_1, pid2cdf1_1 = read_submission(submission_path1)
pid2cdf0_2, pid2cdf1_2 = read_submission(submission_path2)

# systole
assert pid2cdf0_1.keys() == pid2cdf0_2.keys()
for pid in pid2cdf0_1.iterkeys():
    crps = utils_heart.crps(pid2cdf0_1[pid], pid2cdf0_2[pid])
    if crps > 0.02:
        print pid, crps

print '-------------------------------'
# diastole
assert pid2cdf1_1.keys() == pid2cdf1_2.keys()
for pid in pid2cdf1_1.iterkeys():
    crps1 = utils_heart.crps(pid2cdf1_1[pid], pid2cdf1_2[pid])
    crps0 = utils_heart.crps(pid2cdf0_1[pid], pid2cdf0_2[pid])
    crps = 0.5 * (crps0 + crps1)
    if crps > 0.03:
        print pid, crps