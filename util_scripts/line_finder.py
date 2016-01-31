import numpy as np


def hash_orientation(orientation):
    l = list()
    for i in orientation:
        l.append("%.5f"%i) #orientation should be same up to 5 digits after comma
    return "[" + ",".join(l) + "]"

POSITION = 0
ORIENTATION = 1
INDEX = 2


def find_line(coordinates, orientations, labels=None):
    """
    Find which images are consecutive. They will be returned as:
    [[label1, label2], [label3, label4, label5], [label6], [label7]]
    One image can be in multiple lines.
    :param coordinates: list of the dicom ImagePositionPatient
    :param orientations: list of the dicom ImageOrientationPatient
    :param labels: list of the matching labels, which are the labels found in the result.
    :return:
    """
    if labels is None:
        labels = range(len(coordinates))

    p = zip(*(coordinates, orientations, range(len(coordinates))))
    groups_by_orientation = {}
    for point in p:
        hash = hash_orientation(point[ORIENTATION])
        if hash not in groups_by_orientation:
            groups_by_orientation[hash] = []
        groups_by_orientation[hash].append(point)
        
    #print "keys", groups_by_orientation.keys()

    ## put all collinear points together in groups with at least 3 points
    groups = []
    for key in groups_by_orientation.keys():
        points = groups_by_orientation[key]
        #print "group:", key

        if len(points) < 3:
            # they automatically form a line
            lines = [[points]]
        else:
            # go over all the points
            lines = [ [points[0], points[1]] ]
            points_done = [points[0], points[1]]
            for point in points[2:]:
                # go over all known lines:
                found = False
                for i in xrange(len(lines)):
                    a = np.array(lines[i][0][POSITION])
                    b = np.array(lines[i][1][POSITION])
                    c = np.array(point[POSITION])

                    # check if a,b,c are collinear
                    if collinear(a, b, c):
                        lines[i].append(point)
                        found = True

                if not found:
                    # we did not find a line
                    # add lines with this point and all previous points
                    for p in points_done:
                        lines.append( [point, p] )

                points_done.append(point)

        # only use longest line? move points not in that line to other group

        lines = sorted(lines, key=lambda line: -len(line))
        # throw away lines which are too short
        lines = [l for l in lines if len(l) > 2]
        
        # find points which are not on the long lines
        extra_points = groups_by_orientation[key]
        for line in lines:
            for point in lines[0]:
                if point in extra_points:  # it may have already been removed by a duplicate point!
                    extra_points.remove(point)
    
        # add these points as different groups
        for point in extra_points:
            groups.append([point])

        for line in lines:
            if len(line) > 2:
                # make sure the points are equidistant!
                # sort them such that they are equidistant and in logical order
                # when not equidistant, try as many good equidistant lines as possible
                
                #step 1: pick line to project on, and project the points on this line
                param = []
                for p in line:
                    t = p[POSITION][0] - line[0][POSITION][0]
                    param.append((t, p))
                param.sort()

                ''' #DONT DO EQUIDISTANT
                # now, go through this sorted list bubble wise, and find long equidistant lines
                candidates = [(0, param)]
                good_candidates = []
                while candidates:

                    current_try = candidates.pop(0)
                    if len(current_try[1]) < 3:
                        continue
                    if _is_equidistant(current_try[1]):
                        good_candidates.append(current_try[1])
                    else:
                        pivot = current_try[0]
                        param = current_try[1]
                        # once with pivot element removed
                        if pivot < len(param) or pivot>10: #stopping condition
                            next_try = (pivot, param[:pivot] + param[pivot+1:])
                            candidates.append(next_try)
                            # once with pivot element kept
                            next_try = (pivot+1, param)
                            candidates.append(next_try)


                # clean the results. Some lists are completely included.

                good_candidates = sorted(good_candidates, key=lambda line: -len(line))

                already_have = []
                filtered_candidates = []

                for l in good_candidates:
                    hash = []
                    for t, p in l:
                        hash.append( p[INDEX] )
                    found = False
                    for x in already_have:
                        if not [q for q in hash if q not in x]:
                            # we have seen this hash before
                            found = True
                    if not found:
                        already_have.append(hash)
                        filtered_candidates.append(l)


                for l in filtered_candidates:
                    groups.append(zip(*l)[1])
                '''
                groups.append(zip(*param)[1])

            else:  # line is shorter than 3
                groups.append(line)
        
    result = []
    for line in groups:

        #print len(line), "points in line:",
        #for point in line:
        #    print point[INDEX], ",",
        #print

        #check if there are at least 3 distinct points on the line
        g_points = []
        for p in line:
            if p[POSITION] not in g_points:
                g_points.append(p[POSITION])
        if len(g_points)>2:
            result.append([labels[point[INDEX]] for point in line])

    #print "result:",result
    return result


def _is_equidistant(points):
    p0 = points[0][0]
    offset = points[1][0] - points[0][0]
    for i in xrange(len(points)):
        #print abs(points[i][0] - (p0 + i*offset))
        if abs(points[i][0] - (p0 + i*offset)) > 1e-2:  # break tends to be between 1e-1 and 1e-5
            return False
    return True


def collinear(a, b, c):
    area = np.linalg.norm(np.cross(a-b, a-c))/2
    if area < 1e-2:  # break tends to be between 1e1 and 1e-4
        return True
    else:
        return False

if __name__ == "__main__":
    """
    posit = [[-84.854464499127, 38.299143032424, 175.58420681092], [-132.60280715963, -185.68275708306, -22.700323363531], [-119.3742216295, -204.77897994706, -38.129537321695], [-112.90679151475, -211.33719222733, -42.023404814371], [-106.43936140001, -217.89540069291, -45.917272307047], [-106.43936140001, -217.89540069291, -45.917272307047], [-99.971923655865, -224.45360915848, -49.811139799722], [-93.504493541119, -231.01182143876, -53.705007292398], [25.251157914357, -312.18187982658, 27.523793170467], [-87.037063426373, -237.57003371903, -57.598874785074], [-138.41456884974, -186.15733192081, -20.008025555127], [-131.94713873499, -192.71554229373, -23.901892094129], [-125.84165937364, -198.22077148148, -34.235673643716]]
    orient = [[0.7147936309144, -0.6925978528631, 0.0968415169939, 0.2230058641362, 0.0944919995583, -0.9702265954819], [0.9936544059965, -0.0733329714019, -0.0852830390475, 0.0991799171492, 0.9288841951456, 0.3568437978205], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.699190423035, 0.6746562288419, -0.2365834424065, -0.627347647016, 0.7376718075121, 0.2495500634847], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812], [0.6790124509842, 0.2625661033155, 0.6855662862175, 0.3473690222774, 0.7077837728256, -0.6151234780812]]
    print find_line(posit, orient)
    """
    import glob, os, dicom, re
    import cPickle as pickle
    patient_folder_list = glob.glob( os.path.expanduser('/mnt/storage/data/dsb15/pkl_train/*/') ) + glob.glob( os.path.expanduser('/mnt/storage/data/dsb15/pkl_validate/*/') )
    patient_folder_list.sort()
    result_dict = dict()
    for patient_folder in patient_folder_list:
        pkl_list = glob.glob('%s/study/sax_*.pkl' % patient_folder)

        positions = []
        orientations = []
        labels = []

        for pkl in pkl_list:
            data = pickle.load(open(pkl, 'rb'))
            im_pos = [float(i) for i in data['metadata'][0]["ImagePositionPatient"]]
            im_ori = [float(i) for i in data['metadata'][0]["ImageOrientationPatient"]]
            im_axi = int(re.findall(r'sax_\d+', pkl)[0].split('_')[1])

            positions.append(im_pos)
            orientations.append(im_ori)
            labels.append(im_axi)

        # sort such that labels are in rising order
        labels, positions, orientations = map(list, zip(*sorted(zip(labels, positions, orientations))))
        patient_name = int(re.findall(r'/\d+/', patient_folder)[0].split('/')[1])
        result = find_line(positions, orientations, labels)
        result = [r for r in result if len(r)>2]
        print "result:", patient_name, "->", result
        result_dict[patient_name] = result

    print "final result:", result_dict
    pickle.dump(result_dict, open('/mnt/storage/data/dsb15/4d-grouping.pkl', 'wb'))

