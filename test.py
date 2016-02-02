import numpy as np
import utils
import cPickle as pickle
from configuration import config, set_configuration
from collections import defaultdict
import sys
import data_iterators

patch_size = (64, 64)
train_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0)
}

valid_transformation_params = {
    'patch_size': patch_size,
    'rotation_range': None,
    'translation_range': None,
    'shear_range': None
}

# if not (3 <= len(sys.argv) <= 5):
#     sys.exit("Usage: test.py <metadata_path> <test_method>")
#
# metadata_path = sys.argv[1]
# method = sys.argv[2]
#
# metadata = pickle.load(open('/mnt/storage/metadata/kaggle-heart/train/ira/' + metadata_path))
# config_name = metadata['configuration_file']
# set_configuration(config_name)
#
# # predictions
# prediction_dir = '/mnt/storage/metadata/kaggle-heart/predictions/ira'
# if not os.path.isdir(prediction_dir):
#     os.mkdir(prediction_dir)
# predictions_path = prediction_dir + "/%s--%s.npy" % (metadata['experiment_id'], method)
#
# print "Build model"
# model = config().build_model()
# all_layers = nn.layers.get_all_layers(model.l_top)
# all_params = nn.layers.get_all_params(model.l_top)
# num_params = nn.layers.count_params(model.l_top)
# print '  number of parameters: %d' % num_params
# print string.ljust('  layer output shapes:', 36),
# print string.ljust('#params:', 10),
# print 'output shape:'
# for layer in all_layers[:-1]:
#     name = string.ljust(layer.__class__.__name__, 32)
#     num_param = sum([np.prod(p.get_value().shape) for p in layer.get_params()])
#     num_param = string.ljust(num_param.__str__(), 10)
#     print '    %s %s %s' % (name, num_param, layer.output_shape)
#
# nn.layers.set_all_param_values(model.l_top, metadata['param_values'])
#
# valid_data_iterator = config().valid_data_iterator
#
# print
# print 'Data'
# print 'n test: %d' % valid_data_iterator.nsamples
#
# xs_shared = [nn.utils.shared_empty(dim=len(l.shape)) for l in model.l_ins]
# givens_in = {}
# for l_in, x in izip(model.l_ins, xs_shared):
#     givens_in[l_in.input_var] = x
#
# iter_test_det = theano.function([], [nn.layers.get_output(l, deterministic=True) for l in model.l_outs],
#                                 givens=givens_in)
#
# # validation set predictions

valid_data_iterator = data_iterators.PreloadingSlicesVolumeDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/valid',
                                                                         batch_size=32*16,
                                                                         transform_params=valid_transformation_params,
                                                                         labels_path='/data/dsb15_pkl/train.csv',
                                                                         full_batch=False, random=False, infinite=False)

batch_predictions, batch_targets, batch_ids = [], [], []
for _ in xrange(1):
    for xs_batch_valid, ys_batch_valid, ids_batch in valid_data_iterator.generate():
        batch_targets.append(ys_batch_valid)
        batch_ids.append(ids_batch)



# valid_crps = config().get_mean_crps_loss(batch_predictions, batch_targets, batch_ids)
# print 'Validation CRPS: ', valid_crps

print 'pred'
avg_patient_predictions = utils.get_avg_patient_predictions(batch_targets, batch_ids)
print 'tgt'
patient_targets = utils.get_avg_patient_predictions(batch_targets, batch_ids)

assert avg_patient_predictions.viewkeys() == patient_targets.viewkeys()
crpss_sys, crpss_dst = [], []
for id in avg_patient_predictions.iterkeys():
    crpss_sys.append(utils.crps(avg_patient_predictions[id][0], patient_targets[id][0]))
    crpss_dst.append(utils.crps(avg_patient_predictions[id][1], patient_targets[id][1]))

print 'Validation Systole CRPS: ', np.mean(crpss_sys)
print 'Validation Diastole CRPS: ', np.mean(crpss_dst)
