import utils
import utils_heart
import numpy as np
import data

d = utils.load_pkl('meta_gauss_roi_zoom_mask_leaky_after-geit-20160302-094713-test-10-arithmetic.pkl')
lab = data.read_labels('train.csv')
train_valid_ids = utils.load_pkl('valid_split.pkl')
print train_valid_ids

crpss_sys, crpss_dst = [], []
for id in d.iterkeys():
    if str(id) in train_valid_ids['valid']:
        crpss_sys.append(utils_heart.crps(d[id][0], utils_heart.heaviside_function(lab[id][0])))
        crpss_dst.append(utils_heart.crps(d[id][1], utils_heart.heaviside_function(lab[id][1])))
        print id, 0.5 * (crpss_sys[-1] + crpss_dst[-1]), crpss_sys[-1], crpss_dst[-1]

crps0, crps1 = np.mean(crpss_sys), np.mean(crpss_dst)
print 0.5*(crps0 + crps1)