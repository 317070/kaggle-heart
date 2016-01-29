import data_iterators
import pickle

# iterator = PatientsDataGenerator('data/train', labels_path='data/train.csv', batch_size=2)
# iterator = TransformSliceDataGenerator(data_path='data/train', labels_path='data/train.csv', batch_size=2,
#                                        transform_params={'patch_size': (64, 64)}, full_batch=True)

# for x, y in iterator.generate():
#     for k, v in x.iteritems():
#         print k, y[k], v.keys()
#         # for vv in v.keys():
#         #     print vv
#         #     print x[k][vv].shape
#     print '======================='
# print '******************'

#
# for x, y in iterator.generate():
#     print x.shape, y

train_transformation_params = {
    'patch_size': (64, 64),
    'rotation_range': (-16, 16),
    'translation_range': (-8, 8),
    'shear_range': (0, 0)
}

from configuration import *

set_configuration('test')
# iterator = SlicesDataGenerator(data_path='data/train', labels_path='data/train.csv', batch_size=32,
#                                transform_params=train_transformation_params, full_batch=True)

train_data_iterator = data_iterators.SlicesDataGenerator(data_path='/data/dsb15_pkl/pkl_splitted/train',
                                                         batch_size=32,
                                                         transform_params=train_transformation_params,
                                                         labels_path='/data/dsb15_pkl/train.csv', full_batch=True,
                                                         random=True, infinite=True)

for x, y in train_data_iterator.generate():
    print x.shape, y
