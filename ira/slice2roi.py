import numpy as np
import glob
import re
import data
import utils

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
except:
    print 'matplotlib not imported'


def orthogonal_projection_on_slice(percentual_coordinate, source_metadata, target_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array([[image_size[0], 0, 0, 0],
                             [0, image_size[1], 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array([[pixel_spacing[0], 0, 0, 0],
                             [0, pixel_spacing[1], 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape((2, 3))[::-1, :]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array([[Fa[0, 0], Fa[1, 0], 0, posa[0]],
                             [Fa[0, 1], Fa[1, 1], 0, posa[1]],
                             [Fa[0, 2], Fa[1, 2], 0, posa[2]],
                             [0, 0, 0, 1]]), point)
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array([[1, 0, 0, -posb[0]],
                             [0, 1, 0, -posb[1]],
                             [0, 0, 1, -posb[2]],
                             [0, 0, 0, 1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape((2, 3))[::-1, :]
    ff0 = np.sqrt(np.sum(Fb[0, :] * Fb[0, :]))
    ff1 = np.sqrt(np.sum(Fb[1, :] * Fb[1, :]))

    point = np.dot(np.array([[Fb[0, 0] / ff0, Fb[0, 1] / ff0, Fb[0, 2] / ff0, 0],
                             [Fb[1, 0] / ff1, Fb[1, 1] / ff1, Fb[1, 2] / ff1, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array([[1. / pixel_spacing[0], 0, 0, 0],
                             [0, 1. / pixel_spacing[1], 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]]), point)
    image_size = [target_metadata["Rows"], target_metadata["Columns"]]
    point = np.dot(np.array([[1. / image_size[0], 0, 0, 0],
                             [0, 1. / image_size[1], 0, 0],
                             [0, 0, 0, 0],
                             [0, 0, 0, 1]]), point)
    return point[:2, 0]  # percentual coordinate as well


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = utils.get_patient_id(patient_data_path)
    for s in spaths:
        slice_id = utils.get_slice_id(s)
        metadata = data.read_metadata(s)
        d = data.read_slice(s)
        patient_data.append({'data': d, 'metadata': metadata,
                             'slice_id': slice_id, 'patient_id': pid})
    return patient_data


def sort_sax_slices(slices):
    nslices = len(slices)
    positions = np.zeros((nslices,))
    for i in xrange(nslices):
        positions[i] = slices[i]['metadata']['SliceLocation']
    sorted_slices = [s for pos, s in sorted(zip(positions.tolist(), slices),
                                            key=lambda x: x[0], reverse=True)]
    return sorted_slices


def group_sax_slices(slice_stack):
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


def get_slice2roi(data_path, out_filename, num_circles=None, plot=False):
    patient_paths = sorted(glob.glob(data_path + '/*/study'))
    slice2roi = {}
    for p in patient_paths:
        patient_data = get_patient_data(p)

        # sax slices
        sax_slice_stack = []
        ch4, ch2 = None, None
        for s in patient_data:
            if 'sax' in s['slice_id']:
                sax_slice_stack.append(s)
            elif '4ch' in s['slice_id']:
                ch4 = s
            elif '2ch' in s['slice_id']:
                ch2 = s

        sorted_sax_slices = sort_sax_slices(sax_slice_stack)
        grouped_sax_slices = group_sax_slices(sorted_sax_slices)

        # init patient dict
        pid = sorted_sax_slices[0]['patient_id']
        # print pid
        slice2roi[pid] = {}

        for slice_group in grouped_sax_slices:
            # pixel spacing changes within one patient but not too much
            pixel_spacing = slice_group[0]['metadata']['PixelSpacing'][0]
            if num_circles:
                roi_center, roi_radii = data.extract_roi(slice_group, pixel_spacing, num_circles=num_circles)
            else:
                roi_center, roi_radii = data.extract_roi(slice_group, pixel_spacing)

            if plot:
                plot_roi(slice_group, roi_center, roi_radii)

            for s in slice_group:
                sid = s['slice_id']
                slice2roi[pid][sid] = {'roi_center': roi_center, 'roi_radii': roi_radii}

        # project found roi_centers on the 4ch and 2ch slice
        ch4_centers = []
        ch2_centers = []
        for slice in sorted_sax_slices:
            sid = slice['slice_id']
            roi_center = slice2roi[pid][sid]['roi_center']

            metadata_source = slice['metadata']
            hough_roi_center = (float(roi_center[0]) / metadata_source['Rows'],
                                float(roi_center[1]) / metadata_source['Columns'])
            if ch4 is not None:
                metadata_target = ch4['metadata']
                result = orthogonal_projection_on_slice(hough_roi_center, metadata_source, metadata_target)
                ch_roi_center = [float(result[0]) * metadata_target['Rows'],
                                 float(result[1]) * metadata_target['Columns']]
                ch4_centers.append(ch_roi_center)

            if ch2 is not None:
                metadata_target = ch2['metadata']
                result = orthogonal_projection_on_slice(hough_roi_center, metadata_source, metadata_target)
                ch_roi_center = [float(result[0]) * metadata_target['Rows'],
                                 float(result[1]) * metadata_target['Columns']]
                ch2_centers.append(ch_roi_center)

        if ch4 is not None:
            centers = np.array(ch4_centers)
            ch4_result_center = tuple(np.mean(centers, axis=0))
            ch4_result_radius = np.max(np.sqrt((centers - ch4_result_center) ** 2))
            ch4_result_radius = (ch4_result_radius, ch4_result_radius)
            sid = ch4['slice_id']
            slice2roi[pid][sid] = {'roi_center': ch4_result_center,
                                   'roi_radii': ch4_result_radius}
            if plot:
                plot_roi([ch4], ch4_result_center, ch4_result_radius)

        if ch2 is not None:
            centers = np.array(ch2_centers)
            ch2_result_center = tuple(np.mean(centers, axis=0))
            ch2_result_radius = np.max(np.sqrt((centers - ch2_result_center) ** 2))
            ch2_result_radius = (ch2_result_radius, ch2_result_radius)
            sid = ch2['slice_id']
            slice2roi[pid][sid] = {'roi_center': ch2_result_center,
                                   'roi_radii': ch2_result_radius}

            if plot:
                plot_roi([ch2], ch2_result_center, ch2_result_radius)

    utils.save_pkl(slice2roi, out_filename)
    print 'saved to ', out_filename
    return slice2roi


if __name__ == '__main__':
    data_paths = ['/data/dsb15_pkl/pkl_train', '/data/dsb15_pkl/pkl_validate']
    # data_paths = ['/mnt/sda3/data/kaggle-heart/pkl_validate']
    for d in data_paths:
        get_slice2roi(d, plot=False)
