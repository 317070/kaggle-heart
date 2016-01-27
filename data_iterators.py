import numpy as np
import itertools
import data
import buffering
import utils

VALIDATION_SPLIT_PATH = "validation_split_v1.pkl"


class DataLoader(object):
    def __init__(self, **kwargs):
        self.augmentation_transforms_test = [data.tform_identity]  # default to no test-time augmentation
        self.__dict__.update(kwargs)

    def load_train(self):
        images = data.load('train')
        labels = data.labels_train

        split = np.load(VALIDATION_SPLIT_PATH)
        indices_train = split['indices_train']
        indices_valid = split['indices_valid']

        self.images_train = images[indices_train]
        self.labels_train = labels[indices_train]

        self.images_valid = images[indices_valid]
        self.labels_valid = labels[indices_valid]

    def load_test(self):
        self.images_test = data.load('test')


class RescaledDataLoader(DataLoader):

    def create_random_gen(self, images, labels):
        gen = data.rescaled_patches_gen_augmented(images, labels, self.estimate_scale, patch_size=self.patch_size,
                                                  chunk_size=self.chunk_size, num_chunks=self.num_chunks_train,
                                                  augmentation_params=self.augmentation_params)

        def random_gen():
            for chunk_x, chunk_y, chunk_shape in gen:
                yield [chunk_x[:, None, :, :]], chunk_y

        return buffering.buffered_gen_threaded(random_gen())

    def create_fixed_gen(self, images, augment=False):
        augmentation_transforms = self.augmentation_transforms_test if augment else None
        gen = data.rescaled_patches_gen_fixed(images, self.estimate_scale, patch_size=self.patch_size,
                                              chunk_size=self.chunk_size,
                                              augmentation_transforms=augmentation_transforms)

        def fixed_gen():
            for chunk_x, chunk_shape, chunk_length in gen:
                yield [chunk_x[:, None, :, :]], chunk_length

        return buffering.buffered_gen_threaded(fixed_gen())