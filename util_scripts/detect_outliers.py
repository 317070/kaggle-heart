"""Script to test out detecting outliers
"""

import glob
import re

import numpy as np

# detect wrong pixel widths
all_train_sax_files = glob.glob('/data/dsb15_pkl/pkl_train/*/study/sax*')
all_test_sax_files = glob.glob('/data/dsb15_pkl/pkl_validate/*/study/sax*')

all_sax_files = all_train_sax_files + all_test_sax_files

nr_slices = [0] * 700

for sax_file in all_sax_files:
  pat_id = int(re.search(r'/(\d+)/', sax_file).group(1))
  nr_slices[pat_id-1] += 1

pats_w_slices = {i: [] for i in xrange(30)}

for pid, nrs in enumerate(nr_slices):
  pats_w_slices[nrs].append(pid+1)

for x in pats_w_slices:
  print x, pats_w_slices[x]


#  metadata = np.load(sax_file)['metadata'][0]
#  rows = metadata['Rows']
#  cols = metadata['Columns']
#  if rows < cols:
#    rows, cols = cols, rows
#  pixwidth = metadata['PixelSpacing'][0]
#  
#  norm_rows = rows * pixwidth
#  norm_cols = cols * pixwidth
#  if norm_rows > 600 or norm_cols > 600:
#    print sax_file
