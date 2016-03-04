import numpy as np
import skimage.transform
import re


def slice_location_finder(metadata_dict):
    """
    :param metadata_dict: dict with arbitrary keys, and metadata values
    :return: dict with "relative_position" and "middle_pixel_position" (and others)
    """
    datadict = dict()

    for key, metadata in metadata_dict.iteritems():
        # d1 = all_data['data']
        d2 = metadata
        image_orientation = [float(i) for i in metadata["ImageOrientationPatient"]]
        image_position = [float(i) for i in metadata["ImagePositionPatient"]]
        pixel_spacing = [float(i) for i in metadata["PixelSpacing"]]
        datadict[key] = {
            "orientation": image_orientation,
            "position": image_position,
            "pixel_spacing": pixel_spacing,
            "rows": int(d2["Rows"]),
            "columns": int(d2["Columns"]),
        }

    for key, data in datadict.iteritems():
        # calculate value of middle pixel
        F = np.array(data["orientation"]).reshape((2, 3))
        pixel_spacing = data["pixel_spacing"]
        i, j = data["columns"] / 2.0, data[
            "rows"] / 2.0  # reversed order, as per http://nipy.org/nibabel/dicom/dicom_orientation.html
        im_pos = np.array([[i * pixel_spacing[0], j * pixel_spacing[1]]], dtype='float32')
        pos = np.array(data["position"]).reshape((1, 3))
        position = np.dot(im_pos, F) + pos
        data["middle_pixel_position"] = position[0, :]

    # find the keys of the 2 points furthest away from each other
    if len(datadict) <= 1:
        for key, data in datadict.iteritems():
            data["relative_position"] = 0.0
    else:
        max_dist = -1.0
        max_dist_keys = []
        for key1, data1 in datadict.iteritems():
            for key2, data2 in datadict.iteritems():
                if key1 == key2:
                    continue
                p1 = data1["middle_pixel_position"]
                p2 = data2["middle_pixel_position"]
                distance = np.sqrt(np.sum((p1 - p2) ** 2))
                if distance > max_dist:
                    max_dist_keys = [key1, key2]
                    max_dist = distance
        # project the others on the line between these 2 points
        # sort the keys, so the order is more or less the same as they were
        max_dist_keys.sort(key=lambda x: int(re.search(r'/sax_(\d+)\.pkl$', x).group(1)))
        p_ref1 = datadict[max_dist_keys[0]]["middle_pixel_position"]
        p_ref2 = datadict[max_dist_keys[1]]["middle_pixel_position"]
        v1 = p_ref2 - p_ref1
        v1 = v1 / np.linalg.norm(v1)
        for key, data in datadict.iteritems():
            v2 = data["middle_pixel_position"] - p_ref1
            scalar = np.inner(v1, v2)
            data["relative_position"] = scalar

    sorted_indices = [key for key in sorted(datadict.iterkeys(), key=lambda x: datadict[x]["relative_position"])]

    sorted_distances = []
    for i in xrange(len(sorted_indices) - 1):
        res = []
        for d1, d2 in [(datadict[sorted_indices[i]], datadict[sorted_indices[i + 1]]),
                       (datadict[sorted_indices[i + 1]], datadict[sorted_indices[i]])]:
            F = np.array(d1["orientation"]).reshape((2, 3))
            n = np.cross(F[0, :], F[1, :])
            n = n / np.sqrt(np.sum(n * n))
            p = d2["middle_pixel_position"] - d1["position"]
            distance = np.abs(np.sum(n * p))
            res.append(distance)
        sorted_distances.append(np.mean(res))

    print sorted_distances

    return datadict, sorted_indices, sorted_distances


def patient_coor_from_slice(percentual_coordinate, source_metadata):
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

    return point[:3, 0]  # patient coordinate


def point_projection_on_slice(point, target_metadata):
    point = np.array([[point[0]],
                      [point[1]],
                      [point[2]],
                      [1]])
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
    return point[:2, 0]  # percentual coordinate as well


def get_chan_transformations(ch2_metadata=None,
                             ch4_metadata=None,
                             top_point_metadata=None,
                             bottom_point_metadata=None,
                             output_width=100):
    has_both_chans = False
    if ch2_metadata is None and ch4_metadata is None:
        raise "Need at least one of these slices"
    elif ch2_metadata and ch4_metadata is None:
        ch4_metadata = ch2_metadata
    elif ch4_metadata and ch2_metadata is None:
        ch2_metadata = ch4_metadata
    else:
        has_both_chans = True

    F2 = np.array(ch2_metadata["ImageOrientationPatient"]).reshape((2, 3))[::-1, :]
    F4 = np.array(ch4_metadata["ImageOrientationPatient"]).reshape((2, 3))[::-1, :]

    n2 = np.cross(F2[0, :], F2[1, :])
    n4 = np.cross(F4[0, :], F4[1, :])

    b2 = np.sum(n2 * np.array(ch2_metadata["ImagePositionPatient"]))
    b4 = np.sum(n4 * np.array(ch4_metadata["ImagePositionPatient"]))

    # find top and bottom of my view
    top_point = patient_coor_from_slice(top_point_metadata["hough_roi"], top_point_metadata)
    bottom_point = patient_coor_from_slice(bottom_point_metadata["hough_roi"], bottom_point_metadata)

    # if it has both chan's: middle line is the common line!
    if has_both_chans:
        F5 = np.cross(n2, n4)
        A = np.array([n2, n4])
        b = np.array([b2, b4])
        # print A, b
        P, rnorm, rank, s = np.linalg.lstsq(A, b)
        # print P, rnorm, rank, s

        # find top and bottom on the line
        A = np.array([F5]).T
        b = np.array(top_point)
        # print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A, b)
        # print sc, rnorm, rank, s

        top_point = sc[0] * F5 + P

        A = np.array([F5]).T
        b = np.array(bottom_point)
        # print A,b
        sc, rnorm, rank, s = np.linalg.lstsq(A, b)
        # print sc, rnorm, rank, s

        bottom_point = sc[0] * F5 + P

    ## FIND THE affine transformation ch2 needs:

    ch2_top_point = point_projection_on_slice(top_point, ch2_metadata)
    ch2_bottom_point = point_projection_on_slice(bottom_point, ch2_metadata)
    n = np.array([ch2_bottom_point[1] - ch2_top_point[1], ch2_top_point[0] - ch2_bottom_point[0]])
    ch2_third_point = ch2_top_point + n / 2

    A = np.array([[ch2_top_point[0], ch2_top_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch2_top_point[0], ch2_top_point[1], 1],
                  [ch2_bottom_point[0], ch2_bottom_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch2_bottom_point[0], ch2_bottom_point[1], 1],
                  [ch2_third_point[0], ch2_third_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch2_third_point[0], ch2_third_point[1], 1], ])
    b = np.array([0, 0.5 * output_width, output_width, 0.5 * output_width, 0, 0])
    # print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A, b)
    # print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [0, 0, 1]]))
    ch2_form_fix = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    # same for ch4
    ch4_top_point = point_projection_on_slice(top_point, ch4_metadata)
    ch4_bottom_point = point_projection_on_slice(bottom_point, ch4_metadata)
    n = np.array([ch4_bottom_point[1] - ch4_top_point[1], ch4_top_point[0] - ch4_bottom_point[0]])
    ch4_third_point = ch4_top_point + n / 2

    A = np.array([[ch4_top_point[0], ch4_top_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch4_top_point[0], ch4_top_point[1], 1],
                  [ch4_bottom_point[0], ch4_bottom_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch4_bottom_point[0], ch4_bottom_point[1], 1],
                  [ch4_third_point[0], ch4_third_point[1], 1, 0, 0, 0],
                  [0, 0, 0, ch4_third_point[0], ch4_third_point[1], 1], ])
    b = np.array([0, 0.5 * output_width, output_width, 0.5 * output_width, 0, 0])
    # print A,b
    sc, rnorm, rank, s = np.linalg.lstsq(A, b)
    # print sc, rnorm, rank, s
    # these need to be mixed up a little, because we have non-standard x-y-order
    tform_matrix = np.linalg.inv(np.array([[sc[4], sc[3], sc[5]],
                                           [sc[1], sc[0], sc[2]],
                                           [0, 0, 1]]))
    ch4_form_fix = skimage.transform.ProjectiveTransform(matrix=tform_matrix)

    return ch2_form_fix, ch4_form_fix


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
