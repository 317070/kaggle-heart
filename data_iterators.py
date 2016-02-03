from data import *
import glob
import re
import numpy as np
import itertools
import utils
import os
from collections import defaultdict


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

        return _gen()


class PreloadingSlicesVolumeDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        self.data_path = data_path
        self.patient_paths = glob.glob(data_path + '/*/study/')
        self.slice_paths = [sorted(glob.glob(p + '/*.pkl')) for p in self.patient_paths]
        self.slice_paths = list(itertools.chain(*self.slice_paths))
        self.slice_paths = [s for s in self.slice_paths if 'sax' in s]  # TODO sax in glob
        self.slicepath2pid = {}
        for s in self.slice_paths:
            self.slicepath2pid[s] = int(re.search(r'/(\d+)/', s).group(1))
        self.nsamples = len(self.slice_paths)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params
        # data
        self.slice2npy = None
        self.x_batch, self.y0_batch, self.y1_batch = None, None, None

    def load_data(self):
        if not os.path.isfile(self.data_path + '.pkl'):
            print 'loading data from %s' % self.data_path
            self.slice2npy = {}
            for s in self.slice_paths:
                self.slice2npy[s] = read_slice(s)
            utils.save_pkl(self.slice2npy, self.data_path + '.pkl')
            print 'saved to %s.pkl' % self.data_path
        else:
            print 'loading data from %s.pkl' % self.data_path
            self.slice2npy = utils.load_pkl(self.data_path + '.pkl')

        # make empty batch
        self.x_batch = np.zeros((self.batch_size, 30) + self.transformation_params['patch_size'], dtype='float32')
        self.y0_batch = np.zeros((self.batch_size, 1), dtype='float32')
        self.y1_batch = np.zeros((self.batch_size, 1), dtype='float32')

    def generate(self):
        if not self.slice2npy:
            self.load_data()

        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                patients_ids = []
                nb = len(idxs_batch)
                for i, j in enumerate(idxs_batch):
                    self.x_batch[i] = transform(self.slice2npy[self.slice_paths[j]], self.transformation_params)
                    patient_id = self.slicepath2pid[self.slice_paths[j]]
                    patients_ids.append(patient_id)
                    if self.id2labels:
                        self.y0_batch[i] = self.id2labels[patient_id][0]
                        self.y1_batch[i] = self.id2labels[patient_id][1]
                if self.full_batch:
                    if nb == self.batch_size:
                        yield [self.x_batch], [self.y0_batch, self.y1_batch], patients_ids
                else:
                    yield [self.x_batch[:nb]], [self.y0_batch[:nb], self.y1_batch[:nb]], patients_ids
            if not self.infinite:
                break


class PreloadingPatientsGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False, random=True,
                 infinite=True, **kwargs):
        patient_paths = glob.glob(data_path + '/*/study/')

        self.pid2slice_paths = defaultdict(list)
        self.slice_paths = []
        self.max_slices = 0
        for p in patient_paths:
            pid = int(re.search(r'/(\d+)/', p).group(1))
            spaths = sorted(glob.glob(p + '/sax_*.pkl'))
            self.pid2slice_paths[pid] = spaths
            self.slice_paths += spaths
            if len(spaths) > self.max_slices:
                self.max_slices = len(spaths)

        self.patient_ids = self.pid2slice_paths.keys()

        self.data_path = data_path
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.nsamples = len(self.patient_ids)
        self.batch_size = batch_size
        self.infinite = infinite
        self.id2labels = read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params
        # data
        self.slice_path2npy = None
        self.x_batch, self.y0_batch, self.y1_batch = None, None, None

    def load_data(self):
        if not os.path.isfile(self.data_path + '.pkl'):
            print 'loading data from %s' % self.data_path
            self.slice_path2npy = {}
            for s in self.slice_paths:
                self.slice_path2npy[s] = read_slice(s)
            utils.save_pkl(self.slice_path2npy, self.data_path + '.pkl')
            print 'saved to %s.pkl' % self.data_path
        else:
            print 'loading data from %s.pkl' % self.data_path
            self.slice_path2npy = utils.load_pkl(self.data_path + '.pkl')

        # make empty batch
        self.x_batch = np.zeros((self.batch_size, self.max_slices, 30) + self.transformation_params['patch_size'],
                                dtype='float32')
        self.y0_batch = np.zeros((self.batch_size, 1), dtype='float32')
        self.y1_batch = np.zeros((self.batch_size, 1), dtype='float32')

    def generate(self):
        if not self.slice_path2npy:
            self.load_data()

        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                self.x_batch.fill(0.0)  # fill batch with zeros
                patients_ids = []
                nb = len(idxs_batch)
                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)
                    slice_paths = self.pid2slice_paths[pid]
                    for j, sp in enumerate(slice_paths):
                        self.x_batch[i, j] = transform(self.slice_path2npy[sp], self.transformation_params)

                    if self.id2labels:
                        self.y0_batch[i] = self.id2labels[pid][0]
                        self.y1_batch[i] = self.id2labels[pid][1]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [self.x_batch], [self.y0_batch, self.y1_batch], patients_ids
                else:
                    yield [self.x_batch[:nb]], [self.y0_batch[:nb], self.y1_batch[:nb]], patients_ids
            if not self.infinite:
                break
