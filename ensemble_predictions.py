"""
Given a set of validation predictions, this script computes the optimal linear weights on the validation set.
It computes the weighted blend of test predictions, where some models are replaced by their bagged versions.
"""

import os
import numpy as np
import theano
import theano.tensor as T
import scipy.optimize
import data
import utils
import nn_heart

labels_path = '/data/dsb15_pkl/train.csv'

CONFIGS = ['convroll4_doublescale_fs5', 'cp8', 'convroll4_big_wd_maxout512',
           'triplescale_fs2_fs5', 'cr4_ds', 'convroll5_preinit_resume_drop@420',
           'doublescale_fs5_latemerge_2233', 'convroll_all_broaden_7x7_weightdecay_resume', 'convroll4_1024_lesswd',
           'convroll4_big_weightdecay']


n_models = len(CONFIGS)
valid_predictions_paths = []
predictions_dir = utils.get_dir_path('predictions')
for config in CONFIGS:
    p = predictions_dir + '/%s.pkl' % config
    valid_predictions_paths.append(p)

test_predictions_paths = [p.replace('valid', 'test', 1) for p in valid_predictions_paths]

# loading validation predictions
id2labels = data.read_labels(labels_path)
pids = []
p0_list, p1_list = [], []
t0_list, t1_list = [], []

# fill targets and pids lists
pid2avg_pred = utils.load_pkl(valid_predictions_paths[0])
for pid in sorted(pid2avg_pred.keys()):
    pids.append(pid)
    t0_list.append(id2labels[pid][0])
    t1_list.append(id2labels[pid][1])

# fill predictions
for path in valid_predictions_paths:
    pid2avg_pred = utils.load_pkl(path)
    p00_list, p11_list = [], []

    for pid in sorted(pid2avg_pred.keys()):
        p00_list.append(pid2avg_pred[pid][0])
        p11_list.append(pid2avg_pred[pid][1])

    p0_list.append(p00_list)
    p1_list.append(p11_list)

p0_stack = np.array(p0_list)  # num_sources x num_patients x 600
p1_stack = np.array(p1_list)  # num_sources x num_patients x 600

t0_stack = np.array(t0_list)  # num_datapoints x 1
t1_stack = np.array(t1_list)  # num_datapoints x 1

# ------------- optimizing systole weights
X0 = theano.shared(p0_stack)  # source cdf predictions
t0 = theano.shared(t0_stack)  # volume targets
W0 = T.vector('W')
s0 = T.nnet.softmax(W0).reshape((W0.shape[0], 1, 1))
weighted_avg_predictions = T.sum(X0 * s0, axis=0)  # T.tensordot(X, s, [[0], [0]])
error = nn_heart.crps(weighted_avg_predictions, t0)
grad = T.grad(error, W0)

f = theano.function([W0], error)
g = theano.function([W0], grad)

w_init = np.zeros(n_models, dtype=theano.config.floatX)
out, loss, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

weights0 = np.exp(out)
weights0 /= weights0.sum()

print 'Loss:', loss
print 'Optimal systole weights'
for i in xrange(n_models):
    print weights0[i], os.path.basename(valid_predictions_paths[i])
print

# ------------- optimizing diastole weights
X1 = theano.shared(p1_stack)  # source cdf predictions
t1 = theano.shared(t1_stack)  # volume targets
W1 = T.vector('W')
s1 = T.nnet.softmax(W1).reshape((W1.shape[0], 1, 1))
weighted_avg_predictions = T.sum(X1 * s1, axis=0)  # T.tensordot(X, s, [[0], [0]])
error = nn_heart.crps(weighted_avg_predictions, t1)
grad = T.grad(error, W1)

f = theano.function([W1], error)
g = theano.function([W1], grad)

w_init = np.zeros(n_models, dtype=theano.config.floatX)
out, loss, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

weights1 = np.exp(out)
weights1 /= weights1.sum()

print 'Loss', loss
print 'Optimal diastole weights'
for i in xrange(n_models):
    print weights1[i], os.path.basename(valid_predictions_paths[i])
print

# -----------------------------

print 'Generating test set predictions'
test_pids = []
p0_list, p1_list = [], []

# fill targets and pids lists
pid2avg_pred = utils.load_pkl(test_predictions_paths[0])
for pid in sorted(pid2avg_pred.keys()):
    test_pids.append(pid)

# fill predictions
for path in test_predictions_paths:
    pid2avg_pred = utils.load_pkl(path)
    p00_list, p11_list = [], []

    for pid in sorted(pid2avg_pred.keys()):
        p00_list.append(pid2avg_pred[pid][0])
        p11_list.append(pid2avg_pred[pid][1])

    p0_list.append(p00_list)
    p1_list.append(p11_list)

p0_stack = np.array(p0_list)  # num_sources x num_patients x 600
p1_stack = np.array(p1_list)  # num_sources x num_patients x 600

weighted_p0 = np.sum(p0_stack * weights0.reshape((weights0.shape[0], 1, 1)), axis=0)
weighted_p1 = np.sum(p1_stack * weights1.reshape((weights1.shape[0], 1, 1)), axis=0)

pid2weighted_predictions = {}
for i in xrange(len(test_pids)):
    pid2weighted_predictions[test_pids[i]] = [weighted_p0[i], weighted_p1[i]]

submission_path = utils.get_dir_path('submissions') + '/blend0.csv'
utils.save_submission(pid2weighted_predictions, submission_path)
