Step 1
======
Make sure you run on a linux Ubuntu 14.04, together with cuda 7.5, cudnn4, Theano and Lasagne installed.
You also need recent version of scikit-learn, scikit-image, numpy, scipy, blz, pydicom (as of 2016).

Step 2
======
We make 2 submissions, 1 without validation set and fixed ensemble weights, 1 with validation set with adapted ensemble weights.
If scripts from the first submission would suddenly perform worse, it should be fixed automatically in the second submission.

So set up the SETTINGS.json file correctly. And in SETTINGS.json, set the number of validation patients to 1 for the first submission.

Step 3
======
Generate the pickle files which are used for the rest of the competition with
> python generate_metadata_pkl.py

> python generate_roi_pkl.py

Step 4
======
First, optimize the following models:

> python train.py -c je_ss_normscale_patchcontrast

> python train.py -c je_ss_smcrps_nrmsc128_500_dropnorm

> python train.py -c je_ss_smcrps_nrmsc128_256_dropnorm

> python train.py -c je_ss_nrmsc128_maxconv

> python train.py -c je_ss_smcrps_jonisc64small_500_dropnorm

> python train.py -c je_ss_nrmsc128_bottleneck2

> python train.py -c je_ss_jonisc64small_gauss_longer

> python train.py -c je_ss_jonisc64small_360

> python train.py -c je_ss_jonisc64small_360_leaky

> python train.py -c je_ss_jonisc64small_360_gauss_longer

> python train.py -c je_ss_jonisc64smal_360_gauss_longer_augbright

> python train.py -c je_ss_jonisc80small_360_gauss_longer_augzoom

> python train.py -c je_ss_jonisc80small_360_gauss_longer_augzoombright

> python train.py -c je_ss_jonisc64_360_leaky

> python train.py -c je_ss_jonisc64_360_leaky_convroll

> python train.py -c je_ss_jonisc80_360_leaky_convroll

> python train.py -c je_ss_jonisc80_360_leaky_convroll_augzoombright

Then, optimize the following models:

> python train.py -c je_meta_jsc80leakyconv_augzoombright_betterdist_short

> python train.py -c je_meta_jsc80leakyconv_augzoombright_short

> python train.py -c je_meta_jsc80leakyconv_augzoombright_betterdist

> python train.py -c je_meta_jsc80leakyconv_augzoombright

> python train.py -c je_os_fixedaggr_jonisc80small_augzoombright_betterdists

> python train.py -c je_meta_fixedaggr_jonisc80small_augzoombright_betterdists

> python train.py -c je_os_fixedaggr_jonisc80small_augzoombright

> python train.py -c je_meta_fixedaggr_jonisc80small_augzoombright

> python train.py -c je_meta_fixedaggr_jsc64leakyconv_short

> python train.py -c je_meta_fixedaggr_betterdist

> python train.py -c je_os_segmentandintegrate_smartsigma

> python train.py -c je_meta_fixedaggr_jsc64leakyconv

> python train.py -c je_meta_fixedaggr_jsc80leakyconv

> python train.py -c je_meta_fixedaggr_joniscale80small_filtered_longer

> python train.py -c je_meta_fixedaggr_joniscale64small_filtered_longer

> python train.py -c je_meta_fixedaggr_filtered

> python train.py -c je_os_fixedaggr_relloc_filtered

> python train.py -c je_os_fixedaggr_rellocframe

> python train.py -c je_meta_fixedaggr_joniscale64small_360_gauss

> python train.py -c je_meta_joniscale64small_360


In the mean time, train the following models:

> python train.py -c j6_4ch

> python train.py -c j6_2ch

> python train.py -c j6_4chb

> python train.py -c j6_2chb

> python train.py -c j6_2ch_128mm

> python train.py -c j6_2ch_128mm_skew

> python train.py -c j6_2ch_128mm_skew_zoomb

> python train.py -c j6_sax

> python train.py -c j6_sax_skew

> python train.py -c j6_sax_skew_zoom

> python train.py -c j6_sax_96

> python train.py -c j6_4ch_128mm_specialist

> python train.py -c j6_2ch_128mm_specialist

> python train.py -c j6_2ch_gauss

> python train.py -c j6_4ch_gauss

> python train.py -c j6_2ch_96mm

Then the following meta models:
> python train.py -c j7_jeroen_ch


Also in the meantime, train these following models

> cd ira

> python train.py gauss_roi10_big_leaky_after_seqshift

> python train_meta.py meta_gauss_roi10_big_leaky_after_seqshift

> python predict_framework_transfer.py meta_gauss_roi10_big_leaky_after_seqshift 50 arithmetic

> python train.py gauss_roi_zoom_big

> python train_meta.py meta_gauss_roi_zoom_big

> python predict_framework_transfer.py meta_gauss_roi_zoom_big 50 arithmetic

> python train.py gauss_roi_zoom_mask_leaky_after

> python train_meta.py meta_gauss_roi_zoom_mask_leaky_after

> python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky_after 50 arithmetic

> python train.py gauss_roi10_zoom_mask_leaky_after

> python train_meta.py meta_gauss_roi10_zoom_mask_leaky_after

> python predict_framework_transfer.py meta_gauss_roi10_zoom_mask_leaky_after 50 arithmetic

> python train.py gauss_roi_zoom_mask_leaky

> python train_meta.py meta_gauss_roi_zoom_mask_leaky

> python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky 50 arithmetic

> python train.py gauss_roi10_maxout

> python train_meta.py meta_gauss_roi10_maxout

> python predict_framework_transfer.py meta_gauss_roi10_maxout 50 arithmetic


> python train.py ch2_roi10_zoom_leaky_after_nomask

> python predict_framework_transfer.py ch2_roi10_zoom_leaky_after_nomask 50 arithmetic

> python train.py ch2_zoom_leaky_after_nomask

> python predict_framework_transfer.py ch2_zoom_leaky_after_nomask 50 arithmetic

> python train.py gauss_roi10_maxout

> python predict_framework_transfer.py gauss_roi10_maxout 50 arithmetic

> python train.py gauss_roi10_zoom_mask_leaky_after

> python predict_framework_transfer.py gauss_roi10_zoom_mask_leaky_after 50 arithmetic


Step 5
======
Merge the resulting predictions with
> python merge_script_jeroen.py

Submit the resulting submission file as the first submission.

Step 7
======
Then, in SETTINGS.json, set the number of validation patients to 1/6th (~16%) of the train set.
After that, do step 4 and 5 again.

Step 8
======
Merge the resulting predictions with
> python merge_script.py

Submit the resulting submission file as the second submission.
