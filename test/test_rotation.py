import glob
import re
import data_test

data_path = '/mnt/sda3/data/kaggle-heart/pkl_validate'
# data_path = '/data/dsb15_pkl/pkl_splitted/train'
patient_path = glob.glob(data_path + '/*/study')
for p in patient_path:
    print p
    spaths = sorted(glob.glob(p + '/sax_*.pkl'), key=lambda x: int(re.search(r'/\w*_(\d+)*\.pkl$', x).group(1)))
    for s in spaths:
        data = data_test.read_slice(s)
        metadata = data_test.read_metadata(s)

        out_data = data_test.fix_image_orientation(data, metadata)
        if data.shape != out_data.shape:
            print s
            print data.shape
            print out_data.shape
            print '==================='
