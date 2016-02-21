import cPickle as pickle
import matplotlib.pyplot as plt

d = pickle.load(open("pkl_train_slice2roi.pkl"))
d.update(pickle.load(open("pkl_validate_slice2roi.pkl")))

c = pickle.load(open("pkl_train_metadata.pkl"))
c.update(pickle.load(open("pkl_validate_metadata.pkl")))



slice_radius = [max(float(patient[key]['roi_radii'][0]) / c[pid][key]['PixelSpacing'][0],
                    float(patient[key]['roi_radii'][1]) / c[pid][key]['PixelSpacing'][1]) for pid, patient in d.iteritems() for key in patient.keys()]

plt.hist(slice_radius, 50)
plt.savefig('biggest_radii_overal_mm.png')
plt.show()


slice_radius = [max(float(patient[key]['roi_radii'][0]) / c[pid][key]['PixelSpacing'][0],
                    float(patient[key]['roi_radii'][1]) / c[pid][key]['PixelSpacing'][1]) for pid, patient in d.iteritems() for key in patient.keys() if "sax" in key]

plt.hist(slice_radius, 50)
plt.savefig('biggest_radii_sax_mm.png')
plt.show()


slice_radius = [max(float(patient[key]['roi_radii'][0]) / c[pid][key]['PixelSpacing'][0],
                    float(patient[key]['roi_radii'][1]) / c[pid][key]['PixelSpacing'][1]) for pid, patient in d.iteritems() for key in patient.keys() if "4ch" in key]

plt.hist(slice_radius, 50)
plt.savefig('biggest_radii_4ch_mm.png')
plt.show()


slice_radius = [max(float(patient[key]['roi_radii'][0]) / c[pid][key]['PixelSpacing'][0],
                    float(patient[key]['roi_radii'][1]) / c[pid][key]['PixelSpacing'][1]) for pid, patient in d.iteritems() for key in patient.keys() if "2ch" in key]

plt.hist(slice_radius, 50)
plt.savefig('biggest_radii_2ch_mm.png')
plt.show()