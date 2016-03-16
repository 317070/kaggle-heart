import matplotlib
# matplotlib.use('Qt4Agg')

import data
import numpy as np
from image2gif import writeGif
from PIL import Image, ImageDraw
from ImageDraw import Draw
import PIL
import glob
import re
from matplotlib import animation
import matplotlib.pyplot as plt
import utils
import data as data_test
from configuration import set_configuration, config

set_configuration('test_config')
patch_size = config().patch_size
train_transformation_params = config().train_transformation_params
valid_transformation_params = config().valid_transformation_params

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
slice2roi = utils.load_pkl('../pkl_train_slice2roi.pkl')
slice2roi_valid = utils.load_pkl('../pkl_validate_slice2roi.pkl')
slice2roi.update(slice2roi_valid)

patient_path = sorted(glob.glob(data_path + '/561/study'))
for p in patient_path:
    print p
    spaths = sorted(glob.glob(p + '/*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    slicepath2metadata = {}
    for s in spaths:
        d = data_test.read_slice(s)
        metadata = data_test.read_metadata(s)
        slicepath2metadata[s] = metadata

        pid = utils.get_patient_id(s)
        sid = utils.get_slice_id(s)
        roi = slice2roi[pid][sid]
        roi_center = roi['roi_center']
        print s
        print roi_center

        maxi = -10000
        for f in d:
            image = f.astype(np.float32)
            m = image.max()
            if m > maxi:
                maxi = m

        frames = []
        for f in d:
            image = f.astype(np.float32)  # convert to float
            image -= image.min()  # ensure the minimal value is 0.0
            image /= maxi  # maximum value in imag
            img = Image.fromarray(np.uint8(image * 255)).convert("RGB")

            draw = ImageDraw.Draw(img)
            draw.ellipse([roi_center[1]-2, roi_center[0]-2, roi_center[1] + 2, roi_center[0] + 2], fill='red')
            del draw
            frames.append(img)

        writeGif('%s%s.gif' % (pid, sid.replace('.pkl','')), frames[:1], duration=45 / 1000.0, dither=0)

        out_data, targets_zoom = data_test.transform_norm_rescale_after(d, metadata, train_transformation_params,
                                                                        roi=roi)
