import data
import glob
import re
import itertools
from collections import defaultdict
import numpy as np
import utils


class SliceNormRescaleDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids=None, labels_path=None,
                 slice2roi_path=None, full_batch=False, random=True, infinite=False, view='sax',
                 data_prep_fun=data.transform_norm_rescale, **kwargs):

        if patient_ids:
            self.patient_paths = []
            for pid in patient_ids:
                self.patient_paths.append(data_path + '/%s/study/' % pid)
        else:
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
        self.data_prep_fun = data_prep_fun
        self.slice2roi = utils.load_pkl(slice2roi_path) if slice2roi_path else None

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
                    x_batch[i], targets_zoom = self.data_prep_fun(slice_data, metadata, self.transformation_params,
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
    def __init__(self, data_path, batch_size, transform_params, patient_ids=None, labels_path=None,
                 slice2roi_path=None, full_batch=False, random=True, infinite=True, min_slices=0,
                 data_prep_fun=data.transform_norm_rescale,
                 **kwargs):

        if patient_ids:
            patient_paths = []
            for pid in patient_ids:
                patient_paths.append(data_path + '/%s/study/' % pid)
        else:
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
        self.data_prep_fun = data_prep_fun
        self.slice2roi = utils.load_pkl(slice2roi_path) if slice2roi_path else None

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
                    for sp in slice_paths:
                        slicepath2metadata[sp] = data.read_metadata(sp)
                    slicepath2location = data.slice_location_finder(slicepath2metadata)
                    slice_paths = sorted(slicepath2location.keys(), key=slicepath2location.get)

                    # sample augmentation params per patient
                    random_params = data.sample_augmentation_parameters(self.transformation_params)

                    for j, sp in enumerate(slice_paths):
                        slice_roi = self.slice2roi[str(pid)][
                            utils.get_slice_id(sp)] if self.slice2roi else None

                        slice_data = data.read_slice(sp)

                        x_batch[i, j], targets_zoom = self.data_prep_fun(slice_data, slicepath2metadata[sp],
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


class Ch2Ch4DataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids=None, labels_path=None,
                 slice2roi_path=None, full_batch=False, random=True, infinite=True, min_slices=5, **kwargs):

        if patient_ids:
            patient_paths = []
            for pid in patient_ids:
                patient_paths.append(data_path + '/%s/study/' % pid)
        else:
            patient_paths = glob.glob(data_path + '/*/study/')

        self.pid2sax_slice_paths = defaultdict(list)
        self.pid2ch2_path, self.pid2ch4_path = {}, {}
        for p in patient_paths:
            pid = int(utils.get_patient_id(p))
            spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/sax_(\d+)\.pkl$', x).group(1)))
            if len(spaths) > min_slices:
                self.pid2sax_slice_paths[pid] = spaths

                ch2_path = glob.glob(p + '/2ch_*.pkl')
                self.pid2ch2_path[pid] = ch2_path[0] if ch2_path else None
                ch4_path = glob.glob(p + '/4ch_*.pkl')
                self.pid2ch4_path[pid] = ch4_path[0] if ch4_path else None

        self.patient_ids = self.pid2sax_slice_paths.keys()
        self.nsamples = len(self.patient_ids)

        self.id2labels = data.read_labels(labels_path) if labels_path else None
        self.batch_size = batch_size
        self.rng = np.random.RandomState(42)
        self.full_batch = full_batch
        self.random = random
        self.batch_size = batch_size
        self.infinite = infinite
        self.transformation_params = transform_params
        self.slice2roi = utils.load_pkl(slice2roi_path) if slice2roi_path else None

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_ch2_batch = np.zeros((nb, 30) + self.transformation_params['patch_size'],
                                       dtype='float32')
                x_ch4_batch = np.zeros((nb, 30) + self.transformation_params['patch_size'],
                                       dtype='float32')
                y0_batch = np.zeros((nb, 1), dtype='float32')
                y1_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid = self.patient_ids[idx]
                    patients_ids.append(pid)

                    # do everything with sax
                    sax_slice_paths = self.pid2sax_slice_paths[pid]
                    sax_slicepath2metadata = {}
                    sax_slicepath2roi = {}
                    for s in sax_slice_paths:
                        sax_metadata = data.read_metadata(s)
                        sax_slicepath2metadata[s] = sax_metadata
                        sid = utils.get_slice_id(s)
                        roi = self.slice2roi[str(pid)][sid]
                        sax_slicepath2roi[s] = roi

                    # read ch2, ch4
                    if self.pid2ch2_path[pid]:
                        data_ch2 = data.read_slice(self.pid2ch2_path[pid])
                        metadata_ch2 = data.read_metadata(self.pid2ch2_path[pid])
                    else:
                        data_ch2, metadata_ch2 = None, None

                    if self.pid2ch4_path[pid]:
                        data_ch4 = data.read_slice(self.pid2ch4_path[pid])
                        metadata_ch4 = data.read_metadata(self.pid2ch4_path[pid])
                    else:
                        data_ch4, metadata_ch4 = None, None

                    if data_ch2 is None and data_ch4 is not None:
                        data_ch2 = data_ch4

                    if data_ch4 is None and data_ch2 is not None:
                        data_ch4 = data_ch2

                    # sample augmentation params per patient
                    random_params = data.sample_augmentation_parameters(self.transformation_params)
                    x_ch2_batch[i], x_ch4_batch[i], targets_zoom = data.transform_ch(data_ch2, metadata_ch2,
                                                                                     data_ch4, metadata_ch4,
                                                                                     saxslice2metadata=sax_slicepath2metadata,
                                                                                     transformation=self.transformation_params,
                                                                                     sax2roi=sax_slicepath2roi,
                                                                                     random_augmentation_params=random_params)
                    if self.id2labels:
                        y0_batch[i] = self.id2labels[pid][0] * targets_zoom
                        y1_batch[i] = self.id2labels[pid][1] * targets_zoom

                if self.full_batch:
                    if nb == self.batch_size:
                        yield [x_ch2_batch, x_ch4_batch], [y0_batch, y1_batch], patients_ids
                else:
                    yield [x_ch2_batch, x_ch4_batch], [y0_batch, y1_batch], patients_ids

            if not self.infinite:
                break
