import data
import glob
import re
import itertools
from collections import defaultdict
import numpy as np


class SliceDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        self.data_path = data_path
        self.patient_paths = glob.glob(data_path + '/*/study/')
        self.slice_paths = [sorted(glob.glob(p + '/sax_*.pkl')) for p in self.patient_paths]
        self.slice_paths = list(itertools.chain(*self.slice_paths))
        self.slicepath2pid = {}
        for s in self.slice_paths:
            self.slicepath2pid[s] = int(re.search(r'/(\d+)/', s).group(1))
        self.nsamples = len(self.slice_paths)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params

    def generate(self):
        raise NotImplementedError


class SlicePixAreaDataGenerator(SliceDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        super(SlicePixAreaDataGenerator, self).__init__(data_path, batch_size, transform_params, labels_path,
                                                        full_batch, random, infinite, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batch
                x_batch = np.zeros((nb, 30) + self.transformation_params['patch_size'], dtype='float32')
                pix_area_batch = np.zeros((nb, 1), dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, j in enumerate(idxs_batch):
                    slice_data = data.read_slice(self.slice_paths[j])
                    metadata = data.read_metadata(self.slice_paths[j])
                    x_batch[i], pix_area_batch[i] = data.transform_with_pixel_area(slice_data, metadata,
                                                                                   self.transformation_params)
                    patient_id = self.slicepath2pid[self.slice_paths[j]]
                    patients_ids.append(patient_id)
                    if self.id2labels:
                        y0_batch[i] = self.id2labels[patient_id][0]
                        y1_batch[i] = self.id2labels[patient_id][1]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch, pix_area_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch, pix_area_batch], [y0_batch, y1_batch], patients_ids
            if not self.infinite:
                break


class SliceNormRescaleDataGenerator(SliceDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False,
                 random=True, infinite=False, **kwargs):
        super(SliceNormRescaleDataGenerator, self).__init__(data_path, batch_size, transform_params, labels_path,
                                                            full_batch, random, infinite, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batch
                x_batch = np.zeros((nb, 30) + self.transformation_params['patch_size'], dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, j in enumerate(idxs_batch):
                    slice_data = data.read_slice(self.slice_paths[j])
                    metadata = data.read_metadata(self.slice_paths[j])
                    x_batch[i] = data.transform_norm_rescale(slice_data, metadata,
                                                             self.transformation_params)
                    patient_id = self.slicepath2pid[self.slice_paths[j]]
                    patients_ids.append(patient_id)
                    if self.id2labels:
                        y0_batch[i] = self.id2labels[patient_id][0]
                        y1_batch[i] = self.id2labels[patient_id][1]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch], [y0_batch, y1_batch], patients_ids
            if not self.infinite:
                break


class PatientsDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False, random=True,
                 infinite=True, **kwargs):
        patient_paths = glob.glob(data_path + '/*/study/')

        self.pid2slice_paths = defaultdict(list)
        self.slice_paths = []
        nslices = []
        for p in patient_paths:
            pid = int(re.search(r'/(\d+)/', p).group(1))
            spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
            self.pid2slice_paths[pid] = spaths
            self.slice_paths += spaths
            nslices.append(len(spaths))

        # take most common number of slices
        self.nslices = int(np.median(nslices))

        self.patient_ids = self.pid2slice_paths.keys()

        self.data_path = data_path
        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.nsamples = len(self.patient_ids)
        self.batch_size = batch_size
        self.infinite = infinite
        self.transformation_params = transform_params

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, self.nslices, 30) + self.transformation_params['patch_size'],
                                   dtype='float32')
                slice_mask_batch = np.zeros((nb, self.nslices), dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)
                    # sample random transformation per patient
                    patient_augmentation_params = data.sample_augmentation_parameters(self.transformation_params)
                    slice_paths = self.pid2slice_paths[pid]
                    slice_paths = self.rng.choice(slice_paths, size=min(self.nslices, len(slice_paths)), replace=False)
                    for j, sp in enumerate(slice_paths):
                        x_batch[i, j] = data.transform_norm_rescale(data.read_slice(sp), self.transformation_params,
                                                       patient_augmentation_params)
                        slice_mask_batch[i, j] = 1.0
                    if self.id2labels:
                        y0_batch[i] = self.id2labels[pid][0]
                        y1_batch[i] = self.id2labels[pid][1]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch, slice_mask_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch, slice_mask_batch], [y0_batch, y1_batch], patients_ids

            if not self.infinite:
                break


class PatientsDataGeneratorFixedSlices(PatientsDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, full_batch=False, random=True,
                 infinite=True, **kwargs):
        super(PatientsDataGeneratorFixedSlices, self).__init__(data_path, batch_size, transform_params, labels_path,
                                                               full_batch,
                                                               random,
                                                               infinite, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, self.nslices, 30) + self.transformation_params['patch_size'],
                                   dtype='float32')
                slice_mask_batch = np.zeros((nb, self.nslices), dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)
                    # sample random transformation per patient
                    patient_augmentation_params = data.sample_augmentation_parameters(self.transformation_params)
                    slice_paths = self.pid2slice_paths[pid][:self.nslices]
                    for j, sp in enumerate(slice_paths):
                        x_batch[i, j] = data.transform(data.read_slice(sp), self.transformation_params,
                                                       patient_augmentation_params)
                        slice_mask_batch[i, j] = 1.0
                    if self.id2labels:
                        y0_batch[i] = self.id2labels[pid][0]
                        y1_batch[i] = self.id2labels[pid][1]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch, slice_mask_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch, slice_mask_batch], [y0_batch, y1_batch], patients_ids

            if not self.infinite:
                break
