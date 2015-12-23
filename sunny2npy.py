import re
import os
import fnmatch
import shutil
import dicom
import cv2 #opencv2
import numpy as np
import cPickle as pickle

np.random.seed(317070)
import matplotlib.pyplot as plt

SAX_SERIES = {
    # challenge training
    "SC-HF-I-1": "0004",
    "SC-HF-I-2": "0106",
    "SC-HF-I-4": "0116",
    "SC-HF-I-5": "0156",
    "SC-HF-I-6": "0180",
    "SC-HF-I-7": "0209",
    "SC-HF-I-8": "0226",
    "SC-HF-I-9": "0241",
    "SC-HF-I-10": "0024",
    "SC-HF-I-11": "0043",
    "SC-HF-I-12": "0062",
    "SC-HF-I-40": "0134",
    "SC-HF-NI-3": "0379",
    "SC-HF-NI-4": "0501",
    "SC-HF-NI-7": "0523",
    "SC-HF-NI-11": "0270",
    "SC-HF-NI-12": "0286",
    "SC-HF-NI-13": "0304",
    "SC-HF-NI-14": "0331",
    "SC-HF-NI-15": "0359",
    "SC-HF-NI-31": "0401",
    "SC-HF-NI-33": "0424",
    "SC-HF-NI-34": "0446",
    "SC-HF-NI-36": "0474",
    "SC-HYP-1": "0550",
    "SC-HYP-3": "0650",
    "SC-HYP-6": "0767",
    "SC-HYP-7": "0007",
    "SC-HYP-8": "0796",
    "SC-HYP-9": "0003",
    "SC-HYP-10": "0579",
    "SC-HYP-11": "0601",
    "SC-HYP-12": "0629",
    "SC-HYP-37": "0702",
    "SC-HYP-38": "0734",
    "SC-HYP-40": "0755",
    "SC-N-2": "0898",
    "SC-N-3": "0915",
    "SC-N-5": "0963",
    "SC-N-6": "0984",
    "SC-N-7": "1009",
    "SC-N-9": "1031",
    "SC-N-10": "0851",
    "SC-N-11": "0878",
    "SC-N-40": "0944",
}

SUNNYBROOK_ROOT_PATH = os.path.expanduser("/mnt/storage/data/dsb15/lv-challenge")

TRAIN_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                    "Sunnybrook Cardiac MR Database ContoursPart3",
                                    "TrainingDataContours")
VALIDATION_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                        "Sunnybrook Cardiac MR Database ContoursPart2",
                                        "ValidationDataContours")

ONLINE_CONTOUR_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                                    "Sunnybrook Cardiac MR Database ContoursPart1",
                                    "OnlineDataContours")


print TRAIN_CONTOUR_PATH
TRAIN_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_training")

VALIDATION_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_validation")

ONLINE_IMG_PATH = os.path.join(SUNNYBROOK_ROOT_PATH,
                        "challenge_online")

def shrink_case(case):
    toks = case.split("-")
    def shrink_if_number(x):
        try:
            cvt = int(x)
            return str(cvt)
        except ValueError:
            return x
    return "-".join([shrink_if_number(t) for t in toks])

class Contour(object):
    def __init__(self, ctr_path):
        self.ctr_path = ctr_path
        match = re.search(r"/([^/]*)/contours-manual/IRCCI-expert/IM-0001-(\d{4})-icontour-manual.txt", ctr_path)
        self.case = shrink_case(match.group(1))
        self.img_no = int(match.group(2))
    
    def __str__(self):
        return "<Contour for case %s, image %d>" % (self.case, self.img_no)
    
    __repr__ = __str__

def load_contour(contour, img_path):
    filename = "IM-%s-%04d.dcm" % (SAX_SERIES[contour.case], contour.img_no)
    full_path = os.path.join(img_path, contour.case, filename)
    f = dicom.read_file(full_path)
    img = f.pixel_array.astype(np.int)
    ctrs = np.loadtxt(contour.ctr_path, delimiter=" ").astype(np.int32)
    label = np.zeros_like(img, dtype="uint8")
    cv2.fillPoly(label, [ctrs], 1)
    return img, label
    
def get_all_contours(contour_path):
    contours = [os.path.join(dirpath, f)
        for dirpath, dirnames, files in os.walk(contour_path)
        for f in fnmatch.filter(files, 'IM-0001-*-icontour-manual.txt')
    ]
    print("Shuffle data")
    np.random.shuffle(contours)
    print("Number of examples: {:d}".format(len(contours)))
    extracted = map(Contour, contours)
    return extracted

images = []
labels = []
def export_all_contours(contours, img_path):
    counter_img = 0
    counter_label = 0
    batchsz = 100
    print("Processing {:d} images and labels...".format(len(contours)))
    for i, ctr in enumerate(contours):
        img, label = load_contour(ctr, img_path)
        images.append(img)
        labels.append(label)
        #print ctr
        #if "SC-HYP-12" in ctr.__str__():
            #pass
        """
        if i>-1:
            plt.figure()
            mngr = plt.get_current_fig_manager()
            # to put it into the upper left corner for example:
            mngr.window.setGeometry(50, 100, 640, 545)
            plt.suptitle(ctr.__str__() + "  #%d" % i)
            plt.imshow(img * (1-label) + (np.max(img)-img)*(label) )
            #plt.imshow(label)
            plt.show()
            """


if __name__== "__main__":
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(TRAIN_CONTOUR_PATH)
    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")
    export_all_contours(ctrs, TRAIN_IMG_PATH)
    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(VALIDATION_CONTOUR_PATH)
    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")
    export_all_contours(ctrs, VALIDATION_IMG_PATH)

    print("Mapping ground truth contours to images...")
    ctrs = get_all_contours(ONLINE_CONTOUR_PATH)
    print("Done mapping ground truth contours to images")
    print("\nBuilding LMDB for train...")
    export_all_contours(ctrs, ONLINE_IMG_PATH)

    pickle.dump({'images': images,
                 'labels': labels},
                open("/mnt/storage/data/dsb15/pkl_annotated/data.pkl", "wb"),
                protocol=pickle.HIGHEST_PROTOCOL)