import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import glob
import re
import data
import utils


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/sax_*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(sax_\d+\.pkl)$', s).group(1)
        metadata = data.read_metadata(s)
        d = data.read_slice(s)
        patient_data.append({'data': d, 'metadata': metadata,
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


def plot_roi(slice_group, roi_center, roi_radii):
    x_roi_center, y_roi_center = roi_center[0], roi_center[1]
    x_roi_radius, y_roi_radius = roi_radii[0], roi_radii[1]
    print 'nslices', len(slice_group)

    for dslice in [slice_group[len(slice_group) / 2]]:
        outdata = dslice['data']
        # print dslice['slice_id']
        # print dslice['metadata']['SliceLocation']
        # print dslice['metadata']['ImageOrientationPatient']
        # print dslice['metadata']['PixelSpacing']
        # print dslice['data'].shape
        # print '--------------------------------------'

        roi_mask = np.zeros_like(outdata[0])
        roi_mask[x_roi_center - x_roi_radius:x_roi_center + x_roi_radius,
        y_roi_center - y_roi_radius:y_roi_center + y_roi_radius] = 1

        outdata[:, roi_mask > 0.5] = 0.4 * outdata[:, roi_mask > 0.5]
        outdata[:, roi_mask > 0.5] = 0.4 * outdata[:, roi_mask > 0.5]

        fig = plt.figure(1)
        fig.canvas.set_window_title(dslice['patient_id'] + dslice['slice_id'])

        def init_out():
            im.set_data(outdata[0])

        def animate_out(i):
            im.set_data(outdata[i])
            return im

        im = fig.gca().imshow(outdata[0], cmap='gist_gray_r', vmin=0, vmax=255)
        anim = animation.FuncAnimation(fig, animate_out, init_func=init_out, frames=30, interval=50)
        plt.show()


def get_slice2roi(data_path, plot=False):
    patient_paths = sorted(glob.glob(data_path + '/*/study'))
    slice2roi = {}
    for p in patient_paths:
        patient_data = get_patient_data(p)
        sorted_slices = sort_slices(patient_data)
        grouped_slices = group_slices(sorted_slices)

        # init patient dict
        pid = sorted_slices[0]['patient_id']
        # print pid
        slice2roi[pid] = {}

        # pixel spacing doesn't change within one patient
        pixel_spacing = sorted_slices[0]['metadata']['PixelSpacing'][0]

        for slice_group in grouped_slices:
            try:
                roi_center, roi_radii = data.extract_roi(slice_group, pixel_spacing)
            except:
                print 'Could not find ROI'
                roi_center, roi_radii = None, None
            print roi_center, roi_radii

            if plot and roi_center and roi_radii:
                plot_roi(slice_group, roi_center, roi_radii)

            for s in slice_group:
                sid = s['slice_id']
                slice2roi[pid][sid] = {'roi_center': roi_center, 'roi_radii': roi_radii}

    filename = data_path.split('/')[-1] + '_slice2roi.pkl'
    utils.save_pkl(slice2roi, filename)
    print 'saved to ', filename
    return slice2roi


if __name__ == '__main__':
    # data_paths = ['/data/dsb15_pkl/pkl_train', '/data/dsb15_pkl/pkl_validate']
    data_paths = ['/mnt/sda3/data/kaggle-heart/pkl_validate']
    for d in data_paths:
        get_slice2roi(d, plot=True)
