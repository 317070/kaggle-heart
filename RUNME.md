Step 1
======
Make sure you run on a linux Ubuntu 14.04, together with cuda 7.5, cudnn4, Theano and Lasagne installed.
You also need recent version (as of 2016) of scikit-learn, scikit-image, numpy, scipy, blz and pydicom.

Step 2
======

To set up the SETTINGS.json file correctly, select all paths on a disk with plenty of space (~40GB). Also, set the number of the submission to either 1 or 2.
We make 2 submissions, a first without validation set and fixed ensemble weights, a second with validation set with adapted ensemble weights.

Step 3
======
Generate the pickle files which are used for the rest of the competition with
> python generate_metadata_pkl.py

> python generate_roi_pkl.py

Step 4
======
First, optimize the following models in this order:

> python train.py -c je_os_fixedaggr_relloc_filtered

> python train.py -c je_ss_jonisc64small_360_gauss_longer

> python train.py -c je_meta_fixedaggr_filtered

> python train.py -c j6_2ch_128mm

> python train.py -c j6_2ch_96mm

> python train.py -c je_ss_jonisc80_framemax

> python train.py -c je_meta_fixedaggr_framemax_reg

> python train.py -c je_os_fixedaggr_rellocframe

> python train.py -c j6_2ch_128mm_96

> python train.py -c j6_4ch

> python train.py -c je_ss_jonisc80_leaky_convroll

> python train.py -c je_meta_fixedaggr_jsc80leakyconv

> python train.py -c je_os_fixedaggr_relloc_filtered_discs

> python train.py -c j6_4ch_32mm_specialist

> python train.py -c j6_4ch_128mm_specialist

> python train.py -c je_ss_jonisc80small_360_gauss_longer_augzoombright

> python train.py -c je_meta_fixedaggr_joniscale80small_augzoombright

> python train.py -c je_meta_fixedaggr_joniscale64small_filtered_longer

> python train.py -c je_meta_fixedaggr_jsc80leakyconv_augzoombright_short

> python train.py -c je_meta_fixedaggr_joniscale80small_augzoombright_betterdist

> python train.py -c je_os_segmentandintegrate_smartsigma_dropout

> python train.py -c j6_2ch_128mm_zoom

> python train.py -c j6_2ch_128mm_skew

> python train.py -c je_ss_jonisc64small_360


Also in the meantime, train these following models, which you can find in the directory 'ira'

> cd ira

> python train.py gauss_roi10_maxout_seqshift_96

> python train_meta.py meta_gauss_roi10_maxout_seqshift_96

> python predict_framework_transfer.py gauss_roi10_maxout_seqshift_96 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi10_maxout_seqshift_96 50 arithmetic

---

> python train.py gauss_roi_zoom_big

> python train_meta.py meta_gauss_roi_zoom_big

> python predict_framework_transfer.py gauss_roi_zoom_big 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi_zoom_big 50 arithmetic

---

> python train.py gauss_roi10_zoom_mask_leaky_after

> python train_meta.py meta_gauss_roi10_zoom_mask_leaky_after

> python predict_framework_transfer.py gauss_roi10_zoom_mask_leaky_after 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi10_zoom_mask_leaky_after 50 arithmetic

---

> python train.py gauss_roi10_maxout

> python train_meta.py meta_gauss_roi10_maxout

> python predict_framework_transfer.py gauss_roi10_maxout 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi10_maxout 50 arithmetic

---

> python train.py gauss_roi_zoom_mask_leaky_after

> python train_meta.py meta_gauss_roi_zoom_mask_leaky_after

> python predict_framework_transfer.py gauss_roi_zoom_mask_leaky_after 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky_after 50 arithmetic

---

> python train.py gauss_roi_zoom_mask_leaky_after

> python train_meta.py meta_gauss_roi_zoom_mask_leaky_after

> python predict_framework_transfer.py gauss_roi_zoom_mask_leaky_after 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky_after 50 arithmetic

---

> python train.py gauss_roi_zoom

> python train_meta.py meta_gauss_roi_zoom

> python predict_framework_transfer.py gauss_roi_zoom 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi_zoom 50 arithmetic

---

> python train.py ch2_zoom_leaky_after_maxout

> python predict_framework_transfer.py ch2_zoom_leaky_after_maxout 50 arithmetic

---

> python train.py ch2_zoom_leaky_after_nomask

> python predict_framework_transfer.py ch2_zoom_leaky_after_nomask 50 arithmetic

---

> python train.py ch2_zoom_leaky_after_nomask

> python predict_framework_transfer.py ch2_zoom_leaky_after_nomask 50 arithmetic

---

> python train.py gauss_roi10_zoom_mask_leaky_after

> python predict_framework_transfer.py gauss_roi10_zoom_mask_leaky_after 50 arithmetic


Step 5
======
Merge the resulting predictions with
> python merge_script_jeroen.py

in the case of the first submission.
Merge the resulting predictions with
> python merge_script.py
in the case of the second submission.

Step 6
======

????

Step 7
======
Submit the resulting submission file.