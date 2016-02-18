import utils
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import glob
import re
import data_test
import utils


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/sax_*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(sax_\d+\.pkl)$', s).group(1)
        metadata = data_test.read_metadata(s)
        data = data_test.read_slice(s)
        patient_data.append({'data': data, 'metadata': metadata,
                             'slice_id': slice_id, 'patient_id': pid})
    return patient_data


def sort_slices(slices):
    nslices = len(slices)
    positions = np.zeros((nslices,))
    for i in xrange(nslices):
        positions[i] = slices[i]['metadata']['SliceLocation']
    sorted_slices = [s for pos, s in sorted(zip(positions.tolist(), slices),
                                            key=lambda x: x[0], reverse=True)]
    return sorted_slices


def group_slices(slice_stack):
    """
    Groups slices into stacks with the same image orientation
    :param slice_stack:
    :return: list of slice stacks
    """
    img_orientations = []
    for s in slice_stack:
        img_orientations.append(tuple(s['metadata']['ImageOrientationPatient']))
    img_orientations = list(set(img_orientations))
    if len(img_orientations) == 1:
        return [slice_stack]
    else:
        slice_groups = [[] for _ in xrange(len(img_orientations))]
        for s in slice_stack:
            group = img_orientations.index(tuple(s['metadata']['ImageOrientationPatient']))
            slice_groups[group].append(s)
        return slice_groups


def plot_roi(slice, roi_center, roi_radii):
    x_roi_center, y_roi_center = roi_center[0], roi_center[1]
    x_roi_radius, y_roi_radius = roi_radii[0], roi_radii[1]

    outdata = slice['data']
    print slice['slice_id']
    print slice['metadata']['SliceLocation']
    print slice['metadata']['ImageOrientationPatient']
    print slice['metadata']['PixelSpacing']
    print slice['data'].shape
    print '--------------------------------------'

    roi_mask = np.zeros_like(outdata[0])
    roi_mask[x_roi_center - x_roi_radius:x_roi_center + x_roi_radius,
    y_roi_center - y_roi_radius:y_roi_center + y_roi_radius] = 1

    outdata[:, roi_mask > 0.5] = 0.4 * outdata[:, roi_mask > 0.5]
    outdata[:, roi_mask > 0.5] = 0.4 * outdata[:, roi_mask > 0.5]

    fig = plt.figure(1)
    fig.canvas.set_window_title(slice['patient_id'] + slice['slice_id'])

    def init_out():
        im.set_data(outdata[0])

    def animate_out(i):
        im.set_data(outdata[i])
        return im

    im = fig.gca().imshow(outdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
    anim = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)
    plt.show()


def plot_patient_roi(data_path, slice2roi_valid, slice2roi_train):
    patient_paths = sorted(glob.glob(data_path + '/501/study'))
    for p in patient_paths:
        patient_data = get_patient_data(p)
        sd = patient_data[len(patient_data) / 2]
        slice_id = sd['slice_id']
        pid = sd['patient_id']
        if pid in slice2roi_valid:
            slice2roi = slice2roi_valid
        else:
            slice2roi = slice2roi_train
        roi_center = slice2roi[pid][slice_id]['roi_center']
        roi_radii = slice2roi[pid][slice_id]['roi_radii']
        print roi_center, roi_radii
        plot_roi(sd, roi_center, roi_radii)


if __name__ == '__main__':
    data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
    slice2roi_valid = utils.load_pkl('../pkl_validate_slice2roi.pkl')
    slice2roi_train = utils.load_pkl('../pkl_train_slice2roi.pkl')
    plot_patient_roi(data_path, slice2roi_valid, slice2roi_train)
