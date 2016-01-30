from data import *
import glob
import re
import numpy as np
import itertools
import utils


class PatientsDataGenerator(object):
    """
    Generates batches of patients data
    Never use it.
    """

    def __init__(self, data_path, batch_size, labels_path=None, full_batch=False, random=True, **kwargs):
        self.patient_paths = sorted(glob.glob(data_path + '/*/study/'),
                                    key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random

    def base_generator(self):
        """
        :return: tuple of : 1st dict contains patients data
                            {<patient_id>:
                                {
                                'sax_1': np_data},
                                'sax2': np_data},
                                }
                            }
         if labels are given 2nd dict is:
         {<patient_id>: {'systole':_, 'diastole':_}}
         else None

        """
        while True:
            rand_idxs = np.arange(len(self.patient_paths))
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                patient_paths_batch = [self.patient_paths[i] for i in idxs_batch]
                data_dict = read_patients_data(patient_paths_batch)
                labels_dict = {id: self.id2labels[id] for id in data_dict.iterkeys()} if self.id2labels else None
                if self.full_batch:
                    if len(idxs_batch) == self.batch_size:
                        yield data_dict, labels_dict
                else:
                    yield data_dict, labels_dict


class TransformSliceDataGenerator(PatientsDataGenerator):
    def __init__(self, transform_params, **kwargs):
        super(TransformSliceDataGenerator, self).__init__(**kwargs)
        self.transform_params = transform_params

    def generate(self):
        def _gen():
            for data_dict, labels_dict in self.base_generator():
                batch_size = sum([len(d.keys()) for d in data_dict.values()])
                i = 0
                batch_x = np.zeros((batch_size, 30) + self.transform_params['patch_size'], dtype='float32')
                batch_y = np.zeros((batch_size, 2), dtype='float32')
                for patient_id, slice_dict in data_dict.iteritems():
                    print patient_id, slice_dict.keys()
                    for slice_id, x in slice_dict.iteritems():
                        batch_x[i] = transform(x, self.transform_params)
                        batch_y[i, 0] = labels_dict[patient_id][0]
                        batch_y[i, 1] = labels_dict[patient_id][1]
                        i += 1
                yield batch_x, batch_y

        # return buffering.buffered_gen_threaded(_gen())
        return _gen()


class SlicesDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        self.patient_paths = sorted(glob.glob(data_path + '/*/study/'),
                                    key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))
        self.slice_paths = [sorted(glob.glob(p + '/*.pkl')) for p in self.patient_paths]
        self.slice_paths = list(itertools.chain(*self.slice_paths))
        self.slice_paths = [s for s in self.slice_paths if 'sax' in s]
        self.nsamples = len(self.slice_paths)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params

    def generate(self):
        x_batch = np.zeros((self.batch_size, 30) + self.transformation_params['patch_size'], dtype='float32')
        y0_batch = np.zeros((self.batch_size, 600), dtype='float32')
        y1_batch = np.zeros((self.batch_size, 600), dtype='float32')

        def _gen():
            while True:
                rand_idxs = np.arange(len(self.slice_paths))
                if self.random:
                    self.rng.shuffle(rand_idxs)
                for pos in xrange(0, len(rand_idxs), self.batch_size):
                    idxs_batch = rand_idxs[pos:pos + self.batch_size]
                    nb = len(idxs_batch)
                    for i, j in enumerate(idxs_batch):
                        x_batch[i] = transform(read_slice(self.slice_paths[j]), self.transformation_params)
                        patient_id = int(re.search(r'/(\d+)/', self.slice_paths[j]).group(1))
                        y0_batch[i] = utils.heaviside_function(self.id2labels[patient_id][0])
                        y1_batch[i] = utils.heaviside_function(self.id2labels[patient_id][1])
                        print patient_id, self.id2labels[patient_id][0], self.id2labels[patient_id][1]
                    if self.full_batch:
                        if nb == self.batch_size:
                            yield [x_batch], [y0_batch, y1_batch]
                    else:
                        yield [x_batch[:nb]], [y0_batch[:nb], y1_batch[:nb]]

                if not self.infinite:
                    break

        return _gen()


class SlicesVolumeDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        self.patient_paths = sorted(glob.glob(data_path + '/*/study/'),
                                    key=lambda folder: int(re.search(r'/(\d+)/', folder).group(1)))
        self.slice_paths = [sorted(glob.glob(p + '/*.pkl')) for p in self.patient_paths]
        self.slice_paths = list(itertools.chain(*self.slice_paths))
        self.slice_paths = [s for s in self.slice_paths if 'sax' in s]
        self.nsamples = len(self.slice_paths)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params

    def generate(self):
        x_batch = np.zeros((self.batch_size, 30) + self.transformation_params['patch_size'], dtype='float32')
        y0_batch = np.zeros((self.batch_size, 1), dtype='float32')
        y1_batch = np.zeros((self.batch_size, 1), dtype='float32')

        def _gen():
            while True:
                rand_idxs = np.arange(len(self.slice_paths))
                if self.random:
                    self.rng.shuffle(rand_idxs)
                for pos in xrange(0, len(rand_idxs), self.batch_size):
                    idxs_batch = rand_idxs[pos:pos + self.batch_size]
                    nb = len(idxs_batch)
                    for i, j in enumerate(idxs_batch):
                        x_batch[i] = transform(read_slice(self.slice_paths[j]), self.transformation_params)
                        patient_id = int(re.search(r'/(\d+)/', self.slice_paths[j]).group(1))
                        y0_batch[i] = self.id2labels[patient_id][0]
                        y1_batch[i] = self.id2labels[patient_id][1]
                        print patient_id, self.id2labels[patient_id][0], self.id2labels[patient_id][1]
                    if self.full_batch:
                        if nb == self.batch_size:
                            yield [x_batch], [y0_batch, y1_batch]
                    else:
                        yield [x_batch[:nb]], [y0_batch[:nb], y1_batch[:nb]]

                if not self.infinite:
                    break

        return _gen()
