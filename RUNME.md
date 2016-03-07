Step 1
======
Make sure you run on a linux Ubuntu 14.04, together with cuda 7.5, cudnn4, Theano and Lasagne installed.
You also need recent version (as of 2016) of scikit-learn, scikit-image, numpy, scipy, blz and pydicom.

Step 2
======

To set up the SETTINGS.json file correctly, select all paths on a disk with plenty of space (~40GB). Also, set the number of the submission to either 1 (at first) or 2 (after submission 1 has been gerenated).
We make 2 submissions, a first with validation set and adapted ensemble weights, a second without validation set (i.e. the models are trained on all the training data) with fixed ensemble weights. The fixed ensemble weights are copied over from the first submissions. This means that submission 2 can only be generated after submission 1.

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

> python train.py -c je_ss_jonisc64_leaky_convroll

> python train.py -c je_ss_jonisc80small_360_gauss_longer_augzoombright

> python train.py -c je_meta_fixedaggr_joniscale80small_augzoombright

> python train.py -c je_meta_fixedaggr_joniscale64small_filtered_longer

> python train.py -c je_ss_jonisc80_leaky_convroll_augzoombright

> python train.py -c je_meta_fixedaggr_jsc80leakyconv_augzoombright_short

> python train.py -c je_meta_fixedaggr_joniscale80small_augzoombright_betterdist

> python train.py -c je_os_segmentandintegrate_smartsigma_dropout

> python train.py -c j6_2ch_128mm_zoom

> python train.py -c j6_2ch_128mm_skew

> python train.py -c je_ss_jonisc64small_360


Also in the meantime, train these following models, which you can find in the directory 'ira'

> cd ira

---

> python train.py gauss_roi10_maxout_seqshift_96

> python train_meta.py meta_gauss_roi10_maxout_seqshift_96

> python predict_framework_transfer.py gauss_roi10_maxout_seqshift_96 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi10_maxout_seqshift_96 50 arithmetic

---

> python train.py gauss_roi10_big_leaky_after_seqshift

> python train_meta.py meta_gauss_roi10_big_leaky_after_seqshift

> python predict_framework_transfer.py gauss_roi10_big_leaky_after_seqshift 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi10_big_leaky_after_seqshift 50 arithmetic

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

> python train.py gauss_roi_zoom_mask_leaky

> python train_meta.py meta_gauss_roi_zoom_mask_leaky

> python predict_framework_transfer.py gauss_roi_zoom_mask_leaky 50 arithmetic

> python predict_framework_transfer.py meta_gauss_roi_zoom_mask_leaky 50 arithmetic

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



Step 5
======
Merge the resulting predictions with
> python merge_script.py

This will generate the first submission file. The location of this file is specified by the SUBMISSION_PATH in SETTINGS.json. The path to the file will also be printed to the terminal.

Step 6
======
Merge the predictions with
> python merge_script_jeroen.py

This will output a file containing ensemble weights in the ENSEMBLE_WEIGHTS_PATH, as specified in the SETTINGS.json file. This contains the ensembling weights, which will be transferred over to the second submission ensemble. The path to the file will also be printed to the terminal.

Although not strictly necessairy, you could also keep track of which models get selected in the final ensemble. The script will print out 6 lists of models in total. The lists you're interested in are fourth and the sixth ones. If you do not do this, the end result will remain the same, but will take longer to compute.

Step 7
======
Change the SETTINGS.json file to set it up for the second submission. To do this, SUBMISSION_NR should be set to 2, and the output paths (SUBMISSION_NR, INTERMEDIATE_PREDICTIONS_PATH, PREDICTIONS_PATH) shoudl be changed to point to new empty folders. Make sure that INTERMEDIATE_PREDICTIONS_PATH and PREDICTIONS_PATH point to the same directory.

Step 8
======
Retrain all the models as in step 4. Optionally, you can omit retraining the models which did not come up in the lists you aquired in step 6, unless it is requried by another model. Retraining them anyways will not influence the final submission.

Step 9
======
Merge the predictions with
> python merge_script_jeroen.py PATH_TO_WEIGHTS_FILE

where PATH_TO_WEIGHTS_FILE points to the .pkl that was created by the same script in step 6 and was written to the ENSEMBLE_WEIGHTS_PATH folder. 
This script will output the final submission to the SUBMISSION_PATH folder. The file of intrest will be called 'ensemble_final.<some_numbers>.csv'. The other two files it produces can be ignored.

Step 10
=======
Submit the predictions that were generated in steps 5 and 9
 .
 