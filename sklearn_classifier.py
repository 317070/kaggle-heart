import utils
import data
from collections import defaultdict
from paths import MODEL_PATH, PKL_TRAIN_DATA_PATH
import glob
import utils_heart
import numpy as np

sax_predictions_path = 'meta_gauss_roi_zoom-geit-20160226-230926_mu_sigma.pkl'
ch2_predictions_path = 'ch2_zoom-paard-20160226-172015_mu_sigma.pkl'
ch4_predictions_path = 'ch4_zoom-paard-20160226-190407_mu_sigma.pkl'
labels_path = 'train.csv'

predictions_dir = utils.get_dir_path('predictions', MODEL_PATH)

# sax_predictions = utils.load_pkl(predictions_dir + '/%s' % sax_predictions_path)
# ch2_predictions = utils.load_pkl(predictions_dir + '/%s' % ch2_predictions_path)
# ch4_predictions = utils.load_pkl(predictions_dir + '/%s' % ch4_predictions_path)

sax_predictions = utils.load_pkl(sax_predictions_path)
ch2_predictions = utils.load_pkl(ch2_predictions_path)
ch4_predictions = utils.load_pkl(ch4_predictions_path)

labels = data.read_labels(labels_path)

pid2features = defaultdict(list)

train_pids = set(sax_predictions['train'].keys() + \
                 ch2_predictions['train'].keys() + ch4_predictions['train'].keys())

for pid in train_pids:
    slice_paths = glob.glob(PKL_TRAIN_DATA_PATH + '/' + str(pid) + '/study/sax_*.pkl')
    metadata = data.read_metadata(slice_paths[0])
    age = metadata['PatientAge']
    sex = metadata['PatientSex']
    pid2features[pid].extend([age, sex])

    if pid in sax_predictions['train']:
        mu0 = sax_predictions['train'][pid]['mu'][0][0]
        mu1 = sax_predictions['train'][pid]['mu'][1][0]

        sigma0 = sax_predictions['train'][pid]['sigma'][0][0]
        sigma1 = sax_predictions['train'][pid]['sigma'][1][0]
    else:
        mu0, mu1 = 0., 0.
        sigma0, sigma1 = 0., 0.

    pid2features[pid].extend([mu0, mu1, sigma0, sigma1])

    if pid in ch2_predictions['train']:
        mu0_ch2 = ch2_predictions['train'][pid]['mu'][0][0]
        mu1_ch2 = ch2_predictions['train'][pid]['mu'][1][0]

        sigma0_ch2 = ch2_predictions['train'][pid]['sigma'][0][0]
        sigma1_ch2 = ch2_predictions['train'][pid]['sigma'][1][0]
    else:
        mu0_ch2, mu1_ch2 = 0., 0.
        sigma0_ch2, sigma1_ch2 = 0., 0.

    pid2features[pid].extend([mu0_ch2, mu1_ch2, sigma0_ch2, sigma1_ch2])

    if pid in ch4_predictions['train']:
        mu0_ch4 = ch4_predictions['train'][pid]['mu'][0][0]
        mu1_ch4 = ch4_predictions['train'][pid]['mu'][1][0]

        sigma0_ch4 = ch4_predictions['train'][pid]['sigma'][0]
        sigma1_ch4 = ch4_predictions['train'][pid]['sigma'][1]
    else:
        mu0_ch4, mu1_ch4 = 0., 0.
        sigma0_ch4, sigma1_ch4 = 0., 0.

    pid2features[pid].extend([mu0_ch2, mu1_ch2, sigma0_ch2, sigma1_ch2])
    print pid, pid2features[pid], labels[pid]

X, y0, y1 = [], [], []
for pid, feats in pid2features.iteritems():
    X.append(feats)
    y0.append(labels[pid][0])
    y1.append(labels[pid][1])

X = np.vstack(X)
y0 = np.vstack(y0)
y1 = np.vstack(y1)

mu0_sax = X[:, 2]
mu1_sax = X[:, 3]
sigma0_sax = X[:, 4]
sigma1_sax = X[:, 5]
crps0, crps1 = [], []
for i in xrange(X.shape[0]):
    if sigma0_sax[i] > 0 and sigma1_sax[i] > 0:
        cdf0 = utils_heart.norm_cdf_1d(mu0_sax[i], sigma0_sax[i])
        cdf1 = utils_heart.norm_cdf_1d(mu1_sax[i], sigma1_sax[i])
        crps0.append(utils_heart.crps(cdf0, utils_heart.heaviside_function(y0[i])))
        crps1.append(utils_heart.crps(cdf1, utils_heart.heaviside_function(y1[i])))
print np.mean(crps0), np.mean(crps1)

from sklearn import linear_model
from sklearn import preprocessing

X_scaled = preprocessing.scale(X)
clf = linear_model.Ridge()
clf.fit(X_scaled, y0)
y0_pred = clf.predict(X_scaled)

sigma0_sax = X[:, 4]
crps0 = []
for i in xrange(X.shape[0]):
    if sigma0_sax[i]:
        cdf0 = utils_heart.norm_cdf_1d(y0_pred[i], sigma0_sax[i])
        crps0.append(utils_heart.crps(cdf0, utils_heart.heaviside_function(y0[i])))
print np.mean(crps0)


clf = linear_model.Ridge()
clf.fit(X_scaled, y1)
y1_pred = clf.predict(X_scaled)

sigma1_sax = X[:, 5]
crps1 = []
for i in xrange(X.shape[0]):
    if sigma1_sax[i]:
        cdf1 = utils_heart.norm_cdf_1d(y1_pred[i], sigma1_sax[i])
        crps1.append(utils_heart.crps(cdf1, utils_heart.heaviside_function(y1[i])))
print np.mean(crps1)