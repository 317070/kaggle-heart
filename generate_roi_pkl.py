import argparse
import numpy as np
import glob
import re
from log import print_to_file
from scipy.fftpack import fftn, ifftn
from skimage.feature import peak_local_max, canny
from skimage.transform import hough_circle
import cPickle as pickle
from paths import TRAIN_DATA_PATH, LOGS_PATH, PKL_TRAIN_DATA_PATH, PKL_TEST_DATA_PATH
from paths import TEST_DATA_PATH


def orthogonal_projection_on_slice(percentual_coordinate, source_metadata, target_metadata):
    point = np.array([[percentual_coordinate[0]],
                      [percentual_coordinate[1]],
                      [0],
                      [1]])
    image_size = [source_metadata["Rows"], source_metadata["Columns"]]
    point = np.dot(np.array(  [[image_size[0],0,0,0],
                               [0,image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = source_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[pixel_spacing[0],0,0,0],
                               [0,pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    Fa = np.array(source_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    posa = source_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[Fa[0,0],Fa[1,0],0,posa[0]],
                               [Fa[0,1],Fa[1,1],0,posa[1]],
                               [Fa[0,2],Fa[1,2],0,posa[2]],
                               [0,0,0,1]]), point)
    posb = target_metadata["ImagePositionPatient"]
    point = np.dot(np.array(  [[1,0,0,-posb[0]],
                               [0,1,0,-posb[1]],
                               [0,0,1,-posb[2]],
                               [0,0,0,1]]), point)
    Fb = np.array(target_metadata["ImageOrientationPatient"]).reshape( (2,3) )[::-1,:]
    ff0 = np.sqrt(np.sum(Fb[0,:]*Fb[0,:]))
    ff1 = np.sqrt(np.sum(Fb[1,:]*Fb[1,:]))

    point = np.dot(np.array(  [[Fb[0,0]/ff0,Fb[0,1]/ff0,Fb[0,2]/ff0,0],
                               [Fb[1,0]/ff1,Fb[1,1]/ff1,Fb[1,2]/ff1,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    pixel_spacing = target_metadata["PixelSpacing"]
    point = np.dot(np.array(  [[1./pixel_spacing[0],0,0,0],
                               [0,1./pixel_spacing[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    image_size = [target_metadata["Rows"], target_metadata["Columns"]]
    point = np.dot(np.array(  [[1./image_size[0],0,0,0],
                               [0,1./image_size[1],0,0],
                               [0,0,0,0],
                               [0,0,0,1]]), point)
    return point[:2,0]  # percentual coordinate as well

#joni
minradius = 15
maxradius = 65

kernel_width = 5
center_margin = 8
num_peaks = 10
num_circles = 10  # 20
radstep = 2

#ira
minradius_mm=25
maxradius_mm=45
kernel_width=5
center_margin=8
num_peaks=10
num_circles=20
radstep=2


def extract_roi(data, pixel_spacing, minradius_mm=15, maxradius_mm=65, kernel_width=5, center_margin=8, num_peaks=10,
                num_circles=10, radstep=2):
    """
    Returns center and radii of ROI region in (i,j) format
    """
    # radius of the smallest and largest circles in mm estimated from the train set
    # convert to pixel counts
    minradius = int(minradius_mm / pixel_spacing)
    maxradius = int(maxradius_mm / pixel_spacing)

    ximagesize = data[0]['data'].shape[1]
    yimagesize = data[0]['data'].shape[2]

    xsurface = np.tile(range(ximagesize), (yimagesize, 1)).T
    ysurface = np.tile(range(yimagesize), (ximagesize, 1))
    lsurface = np.zeros((ximagesize, yimagesize))

    allcenters = []
    allaccums = []
    allradii = []

    for dslice in data:
        ff1 = fftn(dslice['data'])
        fh = np.absolute(ifftn(ff1[1, :, :]))
        fh[fh < 0.1 * np.max(fh)] = 0.0
        image = 1. * fh / np.max(fh)


        # find hough circles and detect two radii
        edges = canny(image, sigma=3)
        hough_radii = np.arange(minradius, maxradius, radstep)
        hough_res = hough_circle(edges, hough_radii)

        if hough_res.any():
            centers = []
            accums = []
            radii = []

            for radius, h in zip(hough_radii, hough_res):
                # For each radius, extract num_peaks circles
                peaks = peak_local_max(h, num_peaks=num_peaks)
                centers.extend(peaks)
                accums.extend(h[peaks[:, 0], peaks[:, 1]])
                radii.extend([radius] * num_peaks)

            # Keep the most prominent num_circles circles
            sorted_circles_idxs = np.argsort(accums)[::-1][:num_circles]

            for idx in sorted_circles_idxs:
                center_x, center_y = centers[idx]
                allcenters.append(centers[idx])
                allradii.append(radii[idx])
                allaccums.append(accums[idx])
                brightness = accums[idx]
                lsurface = lsurface + brightness * np.exp(
                    -((xsurface - center_x) ** 2 + (ysurface - center_y) ** 2) / kernel_width ** 2)

    lsurface = lsurface / lsurface.max()

    # select most likely ROI center
    roi_center = np.unravel_index(lsurface.argmax(), lsurface.shape)

    # determine ROI radius
    roi_x_radius = 0
    roi_y_radius = 0
    for idx in range(len(allcenters)):
        xshift = np.abs(allcenters[idx][0] - roi_center[0])
        yshift = np.abs(allcenters[idx][1] - roi_center[1])
        if (xshift <= center_margin) & (yshift <= center_margin):
            roi_x_radius = np.max((roi_x_radius, allradii[idx] + xshift))
            roi_y_radius = np.max((roi_y_radius, allradii[idx] + yshift))

    if roi_x_radius > 0 and roi_y_radius > 0:
        roi_radii = roi_x_radius, roi_y_radius
    else:
        roi_radii = None

    return roi_center, roi_radii


def read_slice(path):
    return pickle.load(open(path))['data']


def read_metadata(path):
    d = pickle.load(open(path))['metadata'][0]
    metadata = {k: d[k] for k in ['PixelSpacing', 'ImageOrientationPatient', 'ImagePositionPatient', 'SliceLocation',
                                  'PatientSex', 'PatientAge', 'Rows', 'Columns']}
    metadata['PixelSpacing'] = np.float32(metadata['PixelSpacing'])
    metadata['ImageOrientationPatient'] = np.float32(metadata['ImageOrientationPatient'])
    metadata['SliceLocation'] = np.float32(metadata['SliceLocation'])
    metadata['ImagePositionPatient'] = np.float32(metadata['ImagePositionPatient'])
    metadata['PatientSex'] = 1 if metadata['PatientSex'] == 'F' else 0
    metadata['PatientAge'] = int(metadata['PatientAge'][1:3])
    metadata['Rows'] = int(metadata['Rows'])
    metadata['Columns'] = int(metadata['Columns'])
    return metadata


def get_patient_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/sax_*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(sax_\d+\.pkl)$', s).group(1)
        metadata = read_metadata(s)
        d = read_slice(s)
        patient_data.append({'data': d, 'metadata': metadata,
                             'slice_id': slice_id, 'patient_id': pid})
    return patient_data


def get_patient_ch_data(patient_data_path):
    patient_data = []
    spaths = sorted(glob.glob(patient_data_path + '/*ch_*.pkl'),
                    key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    pid = re.search(r'/(\d+)/study$', patient_data_path).group(1)
    for s in spaths:
        slice_id = re.search(r'/(\d+ch_\d+\.pkl)$', s).group(1)
        metadata = read_metadata(s)
        d = read_slice(s)
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
    patient_paths = sorted(glob.glob(data_path + '*/study'))
    slice2roi = {}
    for p in patient_paths:
        patient_data = get_patient_data(p)
        sorted_slices = sort_slices(patient_data)
        grouped_slices = group_slices(sorted_slices)

        ch_data = get_patient_ch_data(p)
        ch4, ch2 = None,None
        for data in ch_data:
            if data['slice_id'].startswith("4"):
                ch4 = data
            elif data['slice_id'].startswith("2"):
                ch2 = data


        # init patient dict
        pid = sorted_slices[0]['patient_id']
        print "processing patient %s" % pid
        # print pid
        slice2roi[pid] = {}

        # pixel spacing doesn't change within one patient
        pixel_spacing = sorted_slices[0]['metadata']['PixelSpacing'][0]

        for slice_group in grouped_slices:
            try:
                roi_center, roi_radii = extract_roi(slice_group, pixel_spacing)
            except:
                print 'Could not find ROI'
                roi_center, roi_radii = None, None
            print roi_center, roi_radii

            if plot and roi_center and roi_radii:
                pass
                #plot_roi(slice_group, roi_center, roi_radii)

            for s in slice_group:
                sid = s['slice_id']
                slice2roi[pid][sid] = {'roi_center': roi_center, 'roi_radii': roi_radii}

        # project found roi_centers on the 4ch and 2ch slice
        ch4_centers = []
        ch2_centers = []
        for slice in sorted_slices:
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
            ch4_result_center = np.mean(centers, axis=0)
            ch4_result_radius = np.max(np.sqrt((centers - ch4_result_center)**2))
            sid = ch4['slice_id']
            slice2roi[pid][sid] = {'roi_center': tuple(ch4_result_center), 'roi_radii': (ch4_result_radius, ch4_result_radius)}

        if ch2 is not None:
            centers = np.array(ch2_centers)
            ch2_result_center = np.mean(centers, axis=0)
            ch2_result_radius = np.max(np.sqrt((centers - ch2_result_center)**2))
            sid = ch2['slice_id']
            slice2roi[pid][sid] = {'roi_center': tuple(ch2_result_center), 'roi_radii': (ch2_result_radius, ch2_result_radius)}


    filename = data_path.split('/')[-1] + '_slice2roi_joni.pkl'
    with open(filename, 'w') as f:
        pickle.dump(slice2roi, f)
    print 'saved to ', filename
    return slice2roi


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    #required.add_argument('-c', '--config',
    #                      help='configuration to run',
    #                      required=True)
    args = parser.parse_args()

    data_paths = [PKL_TRAIN_DATA_PATH, PKL_TEST_DATA_PATH]
    log_path = LOGS_PATH + "generate_roi.log"
    with print_to_file(log_path):
        for d in data_paths:
            get_slice2roi(d, plot=True)
        print "log saved to '%s'" % log_path

