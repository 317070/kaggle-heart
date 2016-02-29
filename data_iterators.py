import data
import glob
import re
import itertools
from collections import defaultdict
import numpy as np
import os
import slice2roi
import utils


class SliceDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, slice2roi_path=None,
                 full_batch=False, random=True, infinite=False, view='sax', **kwargs):
        self.data_path = data_path
        self.patient_paths = glob.glob(data_path + '/*/study/')
        self.slice_paths = [sorted(glob.glob(p + '/%s_*.pkl' % view)) for p in self.patient_paths]
        self.slice_paths = list(itertools.chain(*self.slice_paths))
        self.slicepath2pid = {}
        for s in self.slice_paths:
            self.slicepath2pid[s] = int(utils.get_patient_id(s))
        self.nsamples = len(self.slice_paths)
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.transformation_params = transform_params
        if slice2roi_path:
            if not os.path.isfile(slice2roi_path):
                print 'Generating ROI'
                self.slice2roi = slice2roi.get_slice2roi(data_path)
            self.slice2roi = utils.load_pkl(slice2roi_path)
        else:
            self.slice2roi = None

    def generate(self):
        raise NotImplementedError


class SliceNormRescaleDataGenerator(SliceDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, slice2roi_path=None, full_batch=False,
                 random=True, infinite=False, view='sax', **kwargs):
        super(SliceNormRescaleDataGenerator, self).__init__(data_path, batch_size, transform_params,
                                                            labels_path, slice2roi_path,
                                                            full_batch, random, infinite, view, **kwargs)

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
                    slicepath = self.slice_paths[j]
                    patient_id = self.slicepath2pid[slicepath]
                    patients_ids.append(patient_id)
                    slice_roi = self.slice2roi[str(patient_id)][
                        utils.get_slice_id(slicepath)] if self.slice2roi else None

                    slice_data = data.read_slice(slicepath)
                    metadata = data.read_metadata(slicepath)
                    x_batch[i], targets_zoom = data.transform_norm_rescale(slice_data, metadata,
                                                                           self.transformation_params,
                                                                           roi=slice_roi)

                    if self.id2labels:
                        y0_batch[i] = self.id2labels[patient_id][0] * targets_zoom
                        y1_batch[i] = self.id2labels[patient_id][1] * targets_zoom

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch], [y0_batch, y1_batch], patients_ids
            if not self.infinite:
                break


class SliceFFTNormRescaleDataGenerator(SliceDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, slice2roi_path=None, full_batch=False,
                 random=True, infinite=False, view='sax', **kwargs):
        super(SliceFFTNormRescaleDataGenerator, self).__init__(data_path, batch_size, transform_params,
                                                               labels_path, slice2roi_path,
                                                               full_batch, random, infinite, view, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batch
                x_batch = np.zeros((nb, 1) + self.transformation_params['patch_size'], dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, j in enumerate(idxs_batch):
                    slicepath = self.slice_paths[j]
                    patient_id = self.slicepath2pid[slicepath]
                    patients_ids.append(patient_id)
                    slice_roi = self.slice2roi[str(patient_id)][
                        utils.get_slice_id(slicepath)] if self.slice2roi else None

                    slice_data = data.read_fft_slice(slicepath)
                    metadata = data.read_metadata(slicepath)
                    x_batch[i], targets_zoom = data.transform_fft_norm_rescale(slice_data, metadata,
                                                                               self.transformation_params,
                                                                               roi=slice_roi)

                    if self.id2labels:
                        y0_batch[i] = self.id2labels[patient_id][0] * targets_zoom
                        y1_batch[i] = self.id2labels[patient_id][1] * targets_zoom

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_batch], [y0_batch, y1_batch], patients_ids
            if not self.infinite:
                break


class PatientsDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, slice2roi_path=None, full_batch=False,
                 random=True,
                 infinite=True, min_slices=0, **kwargs):

        patient_paths = glob.glob(data_path + '/*/study/')
        self.pid2slice_paths = defaultdict(list)
        nslices = []
        for p in patient_paths:
            pid = int(utils.get_patient_id(p))
            spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/sax_(\d+)\.pkl$', x).group(1)))
            # consider patients only with min_slices
            if len(spaths) > min_slices:
                self.pid2slice_paths[pid] = spaths
                nslices.append(len(spaths))

        # take max number of slices
        self.nslices = int(np.max(nslices))

        self.patient_ids = self.pid2slice_paths.keys()
        self.nsamples = len(self.patient_ids)

        self.data_path = data_path
        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.batch_size = batch_size
        self.infinite = infinite
        self.transformation_params = transform_params
        if slice2roi_path:
            if not os.path.isfile(slice2roi_path):
                print 'Generating ROI'
                self.slice2roi = slice2roi.get_slice2roi(data_path)
            self.slice2roi = utils.load_pkl(slice2roi_path)
        else:
            self.slice2roi = None

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
                sex_age_batch = np.zeros((nb, 2), dtype='float32')
                slice_location_batch = np.zeros((nb, self.nslices, 1), dtype='float32')
                slice_mask_batch = np.zeros((nb, self.nslices), dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)
                    slice_paths = self.pid2slice_paths[pid]

                    # fill metadata dict for linefinder code and sort slices
                    slicepath2metadata = {}
                    slice_locations = []
                    for sp in slice_paths:
                        slice_mtd = data.read_metadata(sp)
                        slicepath2metadata[sp] = slice_mtd
                        slice_locations.append(slice_mtd['SliceLocation'])

                    slice_paths = [s for _, s in sorted(zip(slice_locations, slice_paths),
                                                        key=lambda x: x[0])]

                    # linefinder
                    normalized_slice_pos = self.transformation_params['normalized_slice_pos'] \
                        if 'normalized_slice_pos' in self.transformation_params else False

                    slicepath2location = data.slice_location_finder(slicepath2metadata,
                                                                    normalized=normalized_slice_pos)

                    # sample augmentation params per patient
                    random_params = data.sample_augmentation_parameters(self.transformation_params)
                    for j, sp in enumerate(slice_paths):
                        slice_roi = self.slice2roi[str(pid)][
                            utils.get_slice_id(sp)] if self.slice2roi else None

                        slice_data = data.read_slice(sp)
                        x_batch[i, j], targets_zoom = data.transform_norm_rescale(slice_data, slicepath2metadata[sp],
                                                                                  self.transformation_params,
                                                                                  roi=slice_roi,
                                                                                  random_augmentation_params=random_params)
                        slice_location_batch[i, j] = slicepath2location[sp]
                        slice_mask_batch[i, j] = 1.

                    sex_age_batch[i, 0] = slicepath2metadata[slice_paths[0]]['PatientSex']
                    sex_age_batch[i, 1] = slicepath2metadata[slice_paths[0]]['PatientAge']

                    if self.id2labels:
                        y0_batch[i] = self.id2labels[pid][0] * targets_zoom
                        y1_batch[i] = self.id2labels[pid][1] * targets_zoom

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_batch, slice_mask_batch, slice_location_batch, sex_age_batch], [y0_batch,
                                                                                                 y1_batch], patients_ids
                else:
                    yield [x_batch, slice_mask_batch, slice_location_batch, sex_age_batch], [y0_batch,
                                                                                             y1_batch], patients_ids

            if not self.infinite:
                break


class PatientsAllViewsDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, labels_path=None, slice2roi_path=None, full_batch=False,
                 random=True,
                 infinite=True, min_sax_slices=5, **kwargs):
        patient_paths = glob.glob(data_path + '/*/study/')

        self.pid2_sax_slice_paths = defaultdict(list)
        self.pid2ch2slicepath = {}
        self.pid2ch4slicepath = {}
        nslices = []
        for p in patient_paths:
            pid = int(utils.get_patient_id(p))
            spaths = sorted(glob.glob(p + '/*.pkl'), key=lambda x: int(re.search(r'/(\d+)\.pkl$', x).group(1)))
            self.pid2_sax_slice_paths[pid] = spaths
            nslices.append(len(spaths))

            self.pid2ch2slicepath = glob.glob(p + '/2ch*.pkl')
            self.pid2ch4slicepath = glob.glob(p + '/4ch*.pkl')

        # take most common number of slices
        self.n_sax_slices = int(np.max(nslices))

        self.patient_ids = self.pid2_sax_slice_paths.keys()

        self.data_path = data_path
        self.min_sax_slices = min_sax_slices
        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.nsamples = len(self.patient_ids)
        self.batch_size = batch_size
        self.infinite = infinite
        self.transformation_params = transform_params
        if slice2roi_path:
            if not os.path.isfile(slice2roi_path):
                print 'Generating ROI'
                self.slice2roi = slice2roi.get_slice2roi(data_path)
            self.slice2roi = utils.load_pkl(slice2roi_path)
        else:
            self.slice2roi = None

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_sax_batch = np.zeros((nb, self.n_sax_slices, 30) + self.transformation_params['patch_size'],
                                       dtype='float32')
                sax_slice_location_batch = np.zeros((nb, self.n_sax_slices, 1), dtype='float32')
                sax_slice_mask_batch = np.zeros((nb, self.n_sax_slices), dtype='float32')

                x_ch2_batch = np.zeros((nb, 1, 30) + self.transformation_params['patch_size'],
                                       dtype='float32')
                x_ch4_batch = np.zeros((nb, 1, 30) + self.transformation_params['patch_size'],
                                       dtype='float32')

                view_mask_batch = np.zeros((nb, 3), dtype='float32')
                sex_age_batch = np.zeros((nb, 2), dtype='float32')

                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)
                    sax_slice_paths = self.pid2_sax_slice_paths[pid]
                    ch2_slicepath = self.pid2ch2slicepath
                    ch4_slicepath = self.pid2ch4slicepath

                    # fill views mask
                    view_mask_batch[i, 0] = 1 if len(sax_slice_paths) > self.min_sax_slices else 0
                    view_mask_batch[i, 1] = 1 if ch2_slicepath else 0
                    view_mask_batch[i, 2] = 1 if ch4_slicepath else 0

                    # fill metadata dict for linefinder code and sort slices
                    slicepath2metadata = {}
                    slice_locations = []
                    for sp in sax_slice_paths:
                        slice_mtd = data.read_metadata(sp)
                        slicepath2metadata[sp] = slice_mtd
                        slice_locations.append(slice_mtd['SliceLocation'])

                    slice_paths = [s for _, s in sorted(zip(slice_locations, sax_slice_paths),
                                                        key=lambda x: x[0])]

                    # linefinder
                    normalized_slice_pos = self.transformation_params['normalized_slice_pos'] \
                        if 'normalized_slice_pos' in self.transformation_params else False

                    slicepath2location = data.slice_location_finder(slicepath2metadata,
                                                                    normalized=normalized_slice_pos)

                    # sample augmentation params per patient
                    random_params = data.sample_augmentation_parameters(self.transformation_params)
                    for j, sp in enumerate(sax_slice_paths):
                        slice_roi = self.slice2roi[str(pid)][
                            utils.get_slice_id(sp)] if self.slice2roi else None

                        slice_data = data.read_slice(sp)
                        x_sax_batch[i, j], targets_zoom = data.transform_norm_rescale(slice_data,
                                                                                      slicepath2metadata[sp],
                                                                                      self.transformation_params,
                                                                                      roi=slice_roi,
                                                                                      random_augmentation_params=random_params)
                        sax_slice_location_batch[i, j] = slicepath2location[sp]
                        sax_slice_mask_batch[i, j] = 1.

                    # ch2
                    if ch2_slicepath:
                        ch2slice_data = data.read_slice(ch2_slicepath)
                        ch2slice_mtd = data.read_metadata(ch2_slicepath)
                        ch2slice_roi = self.slice2roi[str(pid)][
                            utils.get_slice_id(ch2_slicepath)] if self.slice2roi else None
                        x_ch2_batch[i], targets_zoom = data.transform_norm_rescale(ch2slice_data,
                                                                                      ch2slice_mtd,
                                                                                      self.transformation_params,
                                                                                      roi=ch2slice_roi,
                                                                                      random_augmentation_params=random_params)


                    sex_age_batch[i, 0] = slicepath2metadata[slice_paths[0]]['PatientSex']
                    sex_age_batch[i, 1] = slicepath2metadata[slice_paths[0]]['PatientAge']

                    if self.id2labels:
                        y0_batch[i] = self.id2labels[pid][0] * targets_zoom
                        y1_batch[i] = self.id2labels[pid][1] * targets_zoom

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_sax_batch, sax_slice_mask_batch, sax_slice_location_batch, x_ch2_batch, x_ch4_batch,
                               sex_age_batch], [y0_batch,
                                                y1_batch], patients_ids
                else:
                    yield [x_sax_batch, sax_slice_mask_batch, sax_slice_location_batch, x_ch2_batch, x_ch4_batch,
                           sex_age_batch], [y0_batch,
                                            y1_batch], patients_ids

            if not self.infinite:
                break
