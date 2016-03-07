"""
Given a set of validation predictions, this script computes the optimal linear weights on the validation set.
It computes the weighted blend of test predictions, where some models are replaced by their bagged versions.
"""
import argparse
from functools import partial

import os
import lasagne
import numpy as np
import sys

import theano
from theano.sandbox.cuda import dnn
import theano.tensor as T
import scipy
from log import print_to_file
from paths import MODEL_PATH, SUBMISSION_PATH, LOGS_PATH, INTERMEDIATE_PREDICTIONS_PATH
from postprocess import make_monotone_distribution, test_if_valid_distribution
import glob
import cPickle as pickle
import utils
from data_loader import _TRAIN_LABELS_PATH, NUM_PATIENTS, test_patients_indices, validation_patients_indices, train_patients_indices
import csv
import string
import time
from validation_set import get_cross_validation_indices

def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data


def hash_mask(mask):
    mask = mask.flatten()
    res = np.sum(2**np.arange(len(mask))*mask)
    return res

def softmax(z):
    z = z-np.max(z)
    res = np.exp(z) / np.sum(np.exp(z))
    return res

def convolve1d(image, filter, window_size):
    flat_image = image.reshape((-1, 1, 1, 600))
    filter = filter.reshape((1,1,1,-1))  # (num_filters, num_input_channels, filter_rows, filter_columns)
    conved = dnn.dnn_conv(img=flat_image,
                          kerns=filter,
                          subsample=(1,1),
                          border_mode=(0, (window_size-1)/2),
                          conv_mode='conv',
                          algo='time_once',
                          )
    return conved.reshape(image.shape)


def generate_information_weight_matrix(expert_predictions,
                                       average_distribution,
                                       eps=1e-14,
                                       use_KL = True,
                                       use_entropy = True,
                                       expert_weights=None):

    if use_KL:
        KL_weight = 1.0
    else:
        KL_weight = 0.0
    if use_entropy:
        cross_entropy_weight = 1.0
    else:
        cross_entropy_weight = 0.0
    pdf = utils.cdf_to_pdf(expert_predictions)
    average_pdf = utils.cdf_to_pdf(average_distribution)
    average_pdf[average_pdf<=0] = np.min(average_pdf[average_pdf>0])/2  # KL is not defined when Q=0 and P is not
    inside = pdf * (np.log(pdf) - np.log(average_pdf[None,None,:]))
    inside[pdf<=0] = 0  # (xlog(x) of zero is zero)
    KL_distance_from_average = np.sum(inside, axis=2)  # (NUM_EXPERTS, NUM_VALIDATIONS)
    assert np.isfinite(KL_distance_from_average).all()

    clipped_predictions = np.clip(expert_predictions, 0.0, 1.0)
    cross_entropy_per_sample = - (    average_distribution[None,None,:]  * np.log(   clipped_predictions+eps) +\
                                  (1.-average_distribution[None,None,:]) * np.log(1.-clipped_predictions+eps) )

    cross_entropy_per_sample[cross_entropy_per_sample<0] = 0  # (NUM_EXPERTS, NUM_VALIDATIONS, 600)
    assert np.isfinite(cross_entropy_per_sample).all()
    if expert_weights is None:
        weights = cross_entropy_weight*cross_entropy_per_sample + KL_weight*KL_distance_from_average[:,:,None]  #+  # <- is too big?
    else:
        weights = (cross_entropy_weight*cross_entropy_per_sample + KL_weight*KL_distance_from_average[:,:,None]) * expert_weights[:,None,None]  #+  # <- is too big?

    #make sure the ones without predictions don't get weight, unless absolutely necessary
    weights[np.where((expert_predictions == average_distribution[None,None,:]).all(axis=2))] = 1e-14
    return weights



def optimize_expert_weights(expert_predictions,
                            average_distribution,
                            mask_matrix=None,
                            targets=None,
                            num_cross_validation_masks=2,
                            num_folds=1,
                            eps=1e-14,
                            cutoff=0.01,
                            do_optimization=True,
                            expert_weights=None,
                            optimal_params=None,
                            special_average=False,
                            *args, **kwargs):
    """
    :param expert_predictions: experts x validation_samples x 600 x
    :param mask_matrix: experts x validation_samples x
    :param targets: validation_samples x 600 x
    :param average_distribution: 600 x
    :param eps:
    :return:
    """
    if expert_weights is not None:
        mask_matrix = mask_matrix[expert_weights>cutoff,:]  # remove
        expert_predictions = expert_predictions[expert_weights>cutoff,:,:]  # remove

    NUM_EXPERTS = expert_predictions.shape[0]
    NUM_FILTER_PARAMETERS = 2
    WINDOW_SIZE = 599

    # optimizing weights
    X = theano.shared(expert_predictions.astype('float32'))  # source predictions = (NUM_EXPERTS, NUM_VALIDATIONS, 600)
    x_coor = theano.shared(np.linspace(-(WINDOW_SIZE-1)/2, (WINDOW_SIZE-1)/2, num=WINDOW_SIZE, dtype='float32'))  # targets = (NUM_VALIDATIONS, 600)

    NUM_VALIDATIONS = expert_predictions.shape[1]
    ind = theano.shared(np.zeros((NUM_VALIDATIONS,), dtype='int32'))  # targets = (NUM_VALIDATIONS, 600)

    if optimal_params is None:
        params_init = np.concatenate([ np.ones((NUM_EXPERTS,), dtype='float32'),
                                       np.ones((NUM_FILTER_PARAMETERS,), dtype='float32') ])
    else:
        params_init = optimal_params.astype('float32')

    params = theano.shared(params_init.astype('float32'))
    #params = T.vector('params', dtype='float32')  # expert weights = (NUM_EXPERTS,)

    C = 0.0001
    if not special_average:
        # Create theano expression
        # inputs:
        W = params[:NUM_EXPERTS]
        weights = T.nnet.softmax(W.dimshuffle('x',0)).dimshuffle(1, 0)
        preds = X.take(ind, axis=1)
        mask = theano.shared(mask_matrix.astype('float32')).take(ind, axis=1)
        # expression
        masked_weights = mask * weights
        tot_masked_weights = T.clip(masked_weights.sum(axis=0), 1e-7, utils.maxfloat)
        preds_weighted_masked = preds * masked_weights.dimshuffle(0, 1, 'x')
        cumulative_distribution = preds_weighted_masked.sum(axis=0) / tot_masked_weights.dimshuffle(0, 'x')
        # loss
        l1_loss = weights.sum()
    else:
        # calculate the weighted average for each of these experts
        weights = generate_information_weight_matrix(expert_predictions, average_distribution)  # = (NUM_EXPERTS, NUM_VALIDATIONS, 600)
        weight_matrix = theano.shared((mask_matrix[:,:,None]*weights).astype('float32'))
        pdf = utils.cdf_to_pdf(expert_predictions)
        x_log = np.log(pdf)
        x_log[pdf<=0] = np.log(eps)
        # Compute the mean
        X_log = theano.shared(x_log.astype('float32'))  # source predictions = (NUM_EXPERTS, NUM_VALIDATIONS, 600)
        X_log_i = X_log.take(ind, axis=1)
        w_i = weight_matrix.take(ind, axis=1)

        W = params[:NUM_EXPERTS]
        w_i = w_i * T.nnet.softmax(W.dimshuffle('x',0)).dimshuffle(1, 0, 'x')

        #the different predictions, are the experts
        geom_av_log = T.sum(X_log_i * w_i, axis=0) / (T.sum(w_i, axis=0) + eps)
        geom_av_log = geom_av_log - T.max(geom_av_log,axis=-1).dimshuffle(0,'x')  # stabilizes rounding errors?

        geom_av = T.exp(geom_av_log)

        geom_pdf = geom_av/T.sum(geom_av,axis=-1).dimshuffle(0,'x')
        l1_loss = 0
        cumulative_distribution = T.cumsum(geom_pdf, axis=-1)

    if not do_optimization:
        ind.set_value(range(NUM_VALIDATIONS))
        f_eval = theano.function([], cumulative_distribution)
        cumulative_distribution = f_eval()
        return cumulative_distribution[0]
    else:
        # convert to theano_values (for regularization)
        t_valid = theano.shared(targets.astype('float32'))  # targets = (NUM_VALIDATIONS, 600)
        t_train = theano.shared(targets.astype('float32'))  # targets = (NUM_VALIDATIONS, 600)

    CRPS_train = T.mean((cumulative_distribution - t_train.take(ind, axis=0))**2) + C * l1_loss
    CRPS_valid = T.mean((cumulative_distribution - t_valid.take(ind, axis=0))**2)

    iter_optimize = theano.function([], CRPS_train, on_unused_input="ignore", updates=lasagne.updates.adam(CRPS_train, [params], 1.0))
    f_val = theano.function([], CRPS_valid)

    def optimize_my_params():
        for _ in xrange(40 if special_average else 100):  # early stopping
            score = iter_optimize()
        result = params.get_value()
        return result, score


    if num_cross_validation_masks==0:

        ind.set_value(range(NUM_VALIDATIONS))
        params.set_value(params_init)
        optimal_params, train_score = optimize_my_params()
        final_weights = -1e10 * np.ones(expert_weights.shape,)
        final_weights[np.where(expert_weights>cutoff)] = optimal_params[:NUM_EXPERTS]
        final_params = np.concatenate(( final_weights, optimal_params[NUM_EXPERTS:]))
        return softmax(final_weights), train_score, final_params
    else:
        final_params = []
        final_losses = []
        print
        print
        print
        for fold in xrange(num_folds):
            for i_cross_validation in xrange(num_cross_validation_masks):
                print "\r\033[F\033[F\033[Fcross_validation %d/%d"%(fold*num_cross_validation_masks+i_cross_validation+1, num_folds*num_cross_validation_masks)
                val_indices = get_cross_validation_indices(range(NUM_VALIDATIONS),
                                                       validation_index=i_cross_validation,
                                                       number_of_splits=num_cross_validation_masks,
                                                       rng_seed=fold,
                                                       )

                indices = [i for i in range(NUM_VALIDATIONS) if i not in val_indices]


                #out, crps, d = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)
                ind.set_value(indices)
                params.set_value(params_init)
                result, train_score = optimize_my_params()

                final_params.append(result)

                ind.set_value(val_indices)
                validation_score = f_val()
                print "              Current train value: %.6f" % train_score
                print "         Current validation value: %.6f" % validation_score
                final_losses.append(validation_score)

        optimal_params = np.mean(final_params, axis=0)
        average_loss   = np.mean(final_losses)

        expert_weights_result = softmax(optimal_params[:NUM_EXPERTS])
        filter_param_result = optimal_params[NUM_EXPERTS:NUM_EXPERTS+NUM_FILTER_PARAMETERS]
        #print "filter param result:", filter_param_result

        return expert_weights_result, average_loss, optimal_params  # (NUM_EXPERTS,)



def get_validate_crps(predictions, labels):
    errors = []
    for patient in validation_patients_indices:
        prediction = predictions[patient-1]
        if "systole_average" in prediction and prediction["systole_average"] is not None:
            assert patient == labels[patient-1, 0]
            error = utils.CRSP(prediction["systole_average"], labels[patient-1, 1])
            errors.append(error)
            error = utils.CRSP(prediction["diastole_average"], labels[patient-1, 2])
            errors.append(error)
    if len(errors)>0:
        errors = np.array(errors)
        estimated_CRSP = np.mean(errors)
        return estimated_CRSP
    else:
        return utils.maxfloat



def calculate_tta_average(predictions, average_method, average_systole, average_diastole):
    already_printed = False
    for prediction in predictions:
        if prediction["systole"].size>0 and prediction["diastole"].size>0:
            assert np.isfinite(prediction["systole"][None,:,:]).all()
            assert np.isfinite(prediction["diastole"][None,:,:]).all()
            prediction["systole_average"] = average_method(prediction["systole"][None,:,:], average=average_systole)
            prediction["diastole_average"] = average_method(prediction["diastole"][None,:,:], average=average_diastole)
            try:
                test_if_valid_distribution(prediction["systole_average"])
                test_if_valid_distribution(prediction["diastole_average"])
            except:
                if not already_printed:
                    #print "WARNING: These distributions are not distributions"
                    already_printed = True
                prediction["systole_average"] = make_monotone_distribution(prediction["systole_average"])
                prediction["diastole_average"] = make_monotone_distribution(prediction["diastole_average"])
                try:
                    test_if_valid_distribution(prediction["systole_average"])
                    test_if_valid_distribution(prediction["diastole_average"])
                except:
                    prediction["systole_average"] = None
                    prediction["diastole_average"] = None
        else:
            # average distributions get zero weight later on
            #prediction["systole_average"] = average_systole
            #prediction["diastole_average"] = average_diastole
            prediction["systole_average"] = None
            prediction["diastole_average"] = None




def merge_all_prediction_files(prediction_file_location = INTERMEDIATE_PREDICTIONS_PATH,
                               redo_tta = True):

    submission_path = SUBMISSION_PATH + "final_submission-%s.csv" % time.time()

    # calculate the average distribution
    regular_labels = _load_file(_TRAIN_LABELS_PATH)

    average_systole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,1]]), axis=0))
    average_diastole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,2]]), axis=0))

    ss_expert_pkl_files = sorted([]
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi10_maxout_seqshift_96.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi10_big_leaky_after_seqshift.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi_zoom_big.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi10_zoom_mask_leaky_after.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi10_maxout.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi_zoom_mask_leaky_after.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.ch2_zoom_leaky_after_maxout.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.ch2_zoom_leaky_after_nomask.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi_zoom_mask_leaky.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.gauss_roi_zoom.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc64small_360_gauss_longer.pkl")
        +glob.glob(prediction_file_location+"j6_2ch_128mm.pkl")
        +glob.glob(prediction_file_location+"j6_2ch_96mm.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc80_framemax.pkl")
        +glob.glob(prediction_file_location+"j6_2ch_128mm_96.pkl")
        +glob.glob(prediction_file_location+"j6_4ch.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc80_leaky_convroll.pkl")
        +glob.glob(prediction_file_location+"j6_4ch_32mm_specialist.pkl")
        +glob.glob(prediction_file_location+"j6_4ch_128mm_specialist.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc64_leaky_convroll.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc80small_360_gauss_longer_augzoombright.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc80_leaky_convroll_augzoombright.pkl")
        +glob.glob(prediction_file_location+"j6_2ch_128mm_zoom.pkl")
        +glob.glob(prediction_file_location+"j6_2ch_128mm_skew.pkl")
        +glob.glob(prediction_file_location+"je_ss_jonisc64small_360.pkl")
    )


    fp_expert_pkl_files = sorted([]
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi10_maxout_seqshift_96.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi10_big_leaky_after_seqshift.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi_zoom_big.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi10_zoom_mask_leaky_after.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi10_maxout.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi_zoom_mask_leaky_after.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi_zoom_mask_leaky.pkl")
        +glob.glob(prediction_file_location+"ira_configurations.meta_gauss_roi_zoom.pkl")
        +glob.glob(prediction_file_location+"je_os_fixedaggr_relloc_filtered.pkl")
        +glob.glob(prediction_file_location+"je_os_fixedaggr_rellocframe.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_filtered.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_framemax_reg.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_jsc80leakyconv.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_jsc80leakyconv_augzoombright_short.pkl")
        +glob.glob(prediction_file_location+"je_os_fixedaggr_relloc_filtered_discs.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_joniscale80small_augzoombright.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_joniscale64small_filtered_longer.pkl")
        +glob.glob(prediction_file_location+"je_meta_fixedaggr_joniscale80small_augzoombright_betterdist.pkl")
        +glob.glob(prediction_file_location+"je_os_segmentandintegrate_smartsigma_dropout.pkl")
    )

    everything = sorted([]
        +glob.glob(prediction_file_location+"*.pkl")
    )
    expert_pkl_files = ss_expert_pkl_files + fp_expert_pkl_files

    print "found %d/44 files" % len(expert_pkl_files)
    """
    # filter expert_pkl_files
    for file in expert_pkl_files[:]:
        try:
            with open(file, 'r') as f:
                print "testing file",file.split('/')[-1]
                data = pickle.load(f)
                if 'predictions' not in data.keys():
                    expert_pkl_files.remove(file)
                    print "                -> removed"
        except:
            print sys.exc_info()[0]
            expert_pkl_files.remove(file)
            print "                -> removed"
    """
    NUM_EXPERTS = len(expert_pkl_files)
    NUM_VALIDATIONS = len(validation_patients_indices)
    NUM_TESTS = len(test_patients_indices)

    systole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')
    diastole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')

    systole_masked_expert_predictions_matrix = np.ones((NUM_EXPERTS, NUM_VALIDATIONS), dtype='bool')
    diastole_masked_expert_predictions_matrix = np.ones((NUM_EXPERTS, NUM_VALIDATIONS), dtype='bool')

    test_systole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_TESTS, 600), dtype='float32')
    test_diastole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_TESTS, 600), dtype='float32')

    test_systole_masked_expert_predictions_matrix = np.ones((NUM_EXPERTS, NUM_TESTS), dtype='bool')
    test_diastole_masked_expert_predictions_matrix = np.ones((NUM_EXPERTS, NUM_TESTS), dtype='bool')

    for i,file in enumerate(expert_pkl_files):
        with open(file, 'r') as f:
            print
            print "loading file",file.split('/')[-1]
            predictions = pickle.load(f)['predictions']

        if redo_tta:
            best_average_method = normalav
            best_average_crps = utils.maxfloat
            for average_method in [geomav,
                                   normalav,
                                   prodav,
                                   weighted_geom_method
                                   ]:
                calculate_tta_average(predictions, average_method, average_systole, average_diastole)

                crps = get_validate_crps(predictions, regular_labels)
                print string.rjust(average_method.__name__,25),"->",crps
                if crps<best_average_crps:
                    best_average_method = average_method
                    best_average_crps = crps

            print " I choose you,", best_average_method.__name__
            calculate_tta_average(predictions, best_average_method, average_systole, average_diastole)
            print " validation loss:", get_validate_crps(predictions, regular_labels)

        for j,patient in enumerate(validation_patients_indices):
            prediction = predictions[patient-1]
            # average distributions get zero weight later on
            if "systole_average" in prediction and prediction["systole_average"] is not None:
                systole_expert_predictions_matrix[i,j,:] = prediction["systole_average"]
            else:
                systole_masked_expert_predictions_matrix[i,j] = False

            if "diastole_average" in prediction and prediction["diastole_average"] is not None:
                diastole_expert_predictions_matrix[i,j,:] = prediction["diastole_average"]
            else:
                diastole_masked_expert_predictions_matrix[i,j] = False

        for j,patient in enumerate(test_patients_indices):
            prediction = predictions[patient-1]
            # average distributions get zero weight later on
            if "systole_average" in prediction and prediction["systole_average"] is not None:
                test_systole_expert_predictions_matrix[i,j,:] = prediction["systole_average"]
            else:
                test_systole_masked_expert_predictions_matrix[i,j] = False

            if "diastole_average" in prediction and prediction["diastole_average"] is not None:
                test_diastole_expert_predictions_matrix[i,j,:] = prediction["diastole_average"]
            else:
                test_diastole_masked_expert_predictions_matrix[i,j] = False

        del predictions  # can be LOADS of data



    cv = [id-1 for id in validation_patients_indices]
    systole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,1].flatten()])
    systole_expert_weight, first_pass_sys_loss, systole_optimal_params = get_optimal_ensemble_weights_for_these_experts(
        expert_mask=np.ones((NUM_EXPERTS,), dtype='bool'),
        prediction_matrix=systole_expert_predictions_matrix,
        mask_matrix=systole_masked_expert_predictions_matrix,
        labels=systole_valid_labels,
        average_distribution=average_systole,
        )

    cv = [id-1 for id in validation_patients_indices]
    diastole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,2].flatten()])
    diastole_expert_weight, first_pass_dia_loss, diastole_optimal_params = get_optimal_ensemble_weights_for_these_experts(
        expert_mask=np.ones((NUM_EXPERTS,), dtype='bool'),
        prediction_matrix=diastole_expert_predictions_matrix,
        mask_matrix=diastole_masked_expert_predictions_matrix,
        labels=diastole_valid_labels,
        average_distribution=average_diastole,
        )


    # print the final weight of every expert
    print "  Systole:  Diastole: Name:"
    for expert_name, systole_weight, diastole_weight in zip(expert_pkl_files, systole_expert_weight, diastole_expert_weight):
        print string.rjust("%.3f%%" % (100*systole_weight), 10),
        print string.rjust("%.3f%%" % (100*diastole_weight), 10),
        print expert_name.split('/')[-1]

    print
    print "estimated leaderboard loss: %f" % ((first_pass_sys_loss + first_pass_dia_loss)/2)
    print

    print
    print "Average the experts according to these weights to find the final distribution"
    final_predictions = [{
                            "patient": i+1,
                            "final_systole": None,
                            "final_diastole": None
                        } for i in xrange(NUM_PATIENTS)]


    generate_final_predictions(
        final_predictions=final_predictions,
        prediction_tag="final_systole",
        expert_predictions_matrix=systole_expert_predictions_matrix,
        masked_expert_predictions_matrix=systole_masked_expert_predictions_matrix,
        test_expert_predictions_matrix=test_systole_expert_predictions_matrix,
        test_masked_expert_predictions_matrix=test_systole_masked_expert_predictions_matrix,
        optimal_params=systole_optimal_params,
        average_distribution=average_systole,
        valid_labels=systole_valid_labels,
        expert_pkl_files=expert_pkl_files,
        expert_weight=systole_expert_weight,
        disagreement_cutoff=0.01 # 0.01
    )

    generate_final_predictions(
        final_predictions=final_predictions,
        prediction_tag="final_diastole",
        expert_predictions_matrix=diastole_expert_predictions_matrix,
        masked_expert_predictions_matrix=diastole_masked_expert_predictions_matrix,
        test_expert_predictions_matrix=test_diastole_expert_predictions_matrix,
        test_masked_expert_predictions_matrix=test_diastole_masked_expert_predictions_matrix,
        optimal_params=diastole_optimal_params,
        average_distribution=average_diastole,
        valid_labels=diastole_valid_labels,
        expert_pkl_files=expert_pkl_files,
        expert_weight=diastole_expert_weight,
        disagreement_cutoff=0.015  # diastole has about 50% more error
    )

    print
    print "Calculating training and validation set scores for reference"


    validation_dict = {}
    for patient_ids, set_name in [(validation_patients_indices, "validation")]:
        errors = []
        for patient in patient_ids:
            prediction = final_predictions[patient-1]

            if "final_systole" in prediction:
                assert patient == regular_labels[patient-1, 0]
                error1 = utils.CRSP(prediction["final_systole"], regular_labels[patient-1, 1])
                errors.append(error1)
                prediction["systole_crps_error"] = error1
                error2 = utils.CRSP(prediction["final_diastole"], regular_labels[patient-1, 2])
                errors.append(error2)
                prediction["diastole_crps_error"] = error1
                prediction["average_crps_error"] = 0.5*error1 + 0.5*error2

        if len(errors)>0:
            errors = np.array(errors)
            estimated_CRSP = np.mean(errors)
            print "  %s kaggle loss: %f" % (string.rjust(set_name, 12), estimated_CRSP)
            validation_dict[set_name] = estimated_CRSP
        else:
            print "  %s kaggle loss: not calculated" % (string.rjust(set_name, 12))
    print "WARNING: both of the previous are overfitted!"

    print
    print "estimated leaderboard loss: %f" % ((first_pass_sys_loss + first_pass_dia_loss)/2)
    print


    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d'%i for i in xrange(600)])
        for prediction in final_predictions:
            # the submission only has patients 501 to 700
            if prediction["patient"] in test_patients_indices:
                if "final_diastole" not in prediction or "final_systole" not in prediction:
                    raise Exception("Not all test-set patients were predicted")
                csvwriter.writerow(["%d_Diastole" % prediction["patient"]] + ["%.18f" % p for p in prediction["final_diastole"].flatten()])
                csvwriter.writerow(["%d_Systole" % prediction["patient"]] + ["%.18f" % p for p in prediction["final_systole"].flatten()])
    print "submission file dumped"



def get_optimal_ensemble_weights_for_these_experts(expert_mask,
                                                   prediction_matrix,
                                                   mask_matrix,
                                                   labels,
                                                   average_distribution):

    selected_expert_predictions = prediction_matrix[expert_mask, :, :]
    selected_mask_matrix = mask_matrix[expert_mask, :]

    last_loss = -1
    for pass_index in xrange(1,11):
        print "  PASS %d: " % pass_index,

        if pass_index==1:
            expert_weight, loss, optimal_params = optimize_expert_weights(
                selected_expert_predictions,
                average_distribution=average_distribution,
                mask_matrix=selected_mask_matrix,
                targets=labels,
                num_cross_validation_masks=selected_expert_predictions.shape[1],
                fold=1,
            )
            first_loss = loss
        else:
            expert_weight, loss, optimal_params = optimize_expert_weights(
                selected_expert_predictions,
                average_distribution=average_distribution,
                mask_matrix=selected_mask_matrix,
                targets=labels,
                num_cross_validation_masks=0,
                expert_weights=expert_weight,
            )
        print "Loss: %.6f" % loss
        if last_loss==loss:
            break
        last_loss = loss


    resulting_optimal_params = np.zeros((len(expert_mask), ))
    resulting_optimal_params[expert_mask] = optimal_params[:len(expert_weight)]
    resulting_optimal_params = np.append(resulting_optimal_params, optimal_params[len(expert_weight):])

    resulting_expert_weights = np.zeros((len(expert_mask), ))
    resulting_expert_weights[expert_mask] = expert_weight

    return resulting_expert_weights, first_loss, resulting_optimal_params



def generate_final_predictions(
        final_predictions,
        prediction_tag,
        expert_predictions_matrix,
        masked_expert_predictions_matrix,
        test_expert_predictions_matrix,
        test_masked_expert_predictions_matrix,
        optimal_params,
        average_distribution,
        valid_labels,
        expert_pkl_files,
        expert_weight,
        disagreement_cutoff):
    cache_dict = dict()

    for final_prediction in final_predictions:
        patient_id = final_prediction['patient']
        if patient_id in train_patients_indices:
            continue

        print "   final prediction of patient %d" % patient_id
        reoptimized = False
        # get me the data for this patient
        # (NUM_EXPERTS, 600)
        if patient_id in validation_patients_indices:
            idx = validation_patients_indices.index(patient_id)
            prediction_matrix = expert_predictions_matrix[:, idx, :]
            mask_matrix = masked_expert_predictions_matrix[:, idx]

        elif patient_id in test_patients_indices:
            idx = test_patients_indices.index(patient_id)
            prediction_matrix = test_expert_predictions_matrix[:, idx, :]
            mask_matrix = test_masked_expert_predictions_matrix[:, idx]

        else:
            raise "This patient is neither train, validation or test?"


        # Is there an expert in the set, which did not predict this patient?
        # if so, re-optimize our weights on the validation set!
        # so, if a model is unavailble, but should be available, re-optimize!
        if np.logical_and(np.logical_not(mask_matrix),  (expert_weight!=0.0)).any():
            if hash_mask(mask_matrix) in cache_dict:
                (optimal_params_for_this_patient, expert_weight) = cache_dict[hash_mask(mask_matrix)]
            else:
                expert_weight, _, optimal_params_for_this_patient = get_optimal_ensemble_weights_for_these_experts(
                    expert_mask=mask_matrix,
                    prediction_matrix=expert_predictions_matrix,
                    mask_matrix=masked_expert_predictions_matrix,
                    labels=valid_labels,
                    average_distribution=average_distribution,
                    )
                cache_dict[hash_mask(mask_matrix)] = (optimal_params_for_this_patient, expert_weight)
            reoptimized = True
        else:
            optimal_params_for_this_patient = optimal_params



        while True: # do-while: break condition at the end
            last_mask_matrix = mask_matrix

            final_prediction[prediction_tag] = optimize_expert_weights(
                            expert_predictions=prediction_matrix[:,None,:],
                            mask_matrix=mask_matrix[:,None],
                            average_distribution=average_distribution,
                            do_optimization=False,
                            optimal_params=optimal_params_for_this_patient,
                            )

            ## find the disagreement between models
            # and remove the ones who disagree too much with the average
            disagreement = find_disagreement(
                expert_predictions=prediction_matrix,
                mask_matrix=mask_matrix,
                ground_truth=final_prediction[prediction_tag]
            )
            mask_matrix = (disagreement<disagreement_cutoff)

            ## RETRAIN THESE ENSEMBLE WEIGHTS ON THE VALIDATION SET IF NEEDED!
            if ((last_mask_matrix!=mask_matrix).any()
               and np.sum(mask_matrix)!=0):

                if hash_mask(mask_matrix) in cache_dict:
                    (optimal_params_for_this_patient, expert_weight) = cache_dict[hash_mask(mask_matrix)]
                else:
                    expert_weight, _, optimal_params_for_this_patient = get_optimal_ensemble_weights_for_these_experts(
                        expert_mask=mask_matrix,
                        prediction_matrix=expert_predictions_matrix,
                        mask_matrix=masked_expert_predictions_matrix,
                        labels=valid_labels,
                        average_distribution=average_distribution,
                        )
                    cache_dict[hash_mask(mask_matrix)] = (optimal_params_for_this_patient, expert_weight)
                reoptimized = True
                continue
            else:
                break
        try:
            test_if_valid_distribution(final_prediction[prediction_tag])
        except:
            final_prediction[prediction_tag] = make_monotone_distribution(final_prediction[prediction_tag])
            test_if_valid_distribution(final_prediction[prediction_tag])

        if reoptimized:
            print "    Weight:  Name:"
            for expert_name, weight in zip(expert_pkl_files, expert_weight):
                if weight>0.:
                    print string.rjust("%.3f%%" % (100*weight), 12),
                    print expert_name.split('/')[-1]


def find_disagreement(expert_predictions,mask_matrix,ground_truth):
    """
    :param expert_predictions: (NUM_EXPERTS, 600)
    :param mask_matrix: (NUM_EXPERTS, )
    :param ground_truth: (600, )
    :return: (NUM_EXPERTS, )
    """
    expert_predictions = expert_predictions[mask_matrix]

    def get_crps(x):
        return np.mean((x - ground_truth)**2)

    crps = np.apply_along_axis(get_crps, axis=1, arr=expert_predictions)
    result = utils.maxfloat * np.ones(mask_matrix.shape)
    result[mask_matrix] = crps
    return result

def get_expert_disagreement(expert_predictions, expert_weights=None, cutoff=0.01):
    """
    :param expert_predictions: experts x 600 x
    :return:
    """
    #if not expert_weights is None:
    #    expert_predictions = expert_predictions[expert_weights>cutoff,:]  # remove
    NUM_EXPERTS = expert_predictions.shape[0]
    cross_crps = 0
    for i in xrange(NUM_EXPERTS):
        for j in xrange(i,NUM_EXPERTS):
            cross_crps += np.mean((expert_predictions[i,:] - expert_predictions[j,:])**2)
    cross_crps /= (NUM_EXPERTS * (NUM_EXPERTS - 1)) / 2
    return cross_crps

#############################################
#                 AVERAGES
#############################################


def geomav(x, *args, **kwargs):
    x = x[0]
    if len(x) == 0:
        return np.zeros(600)
    res = np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))
    return res


def normalav(x, *args, **kwargs):
    x = x[0]
    if len(x) == 0:
        return np.zeros(600)
    return np.mean(x, axis=0)


def prodav(x, *args, **kwargs):
    x = x[0]
    if len(x) == 0:
        return np.zeros(600)
    return np.cumsum(utils.norm_prod(utils.cdf_to_pdf(x)))


def weighted_geom_no_entr(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    if len(prediction_matrix.flatten()) == 0:
        return np.zeros(600)
    weights = generate_information_weight_matrix(prediction_matrix, average, expert_weights=expert_weights, use_entropy=False, *args, **kwargs)
    assert np.isfinite(weights).all()
    pdf = utils.cdf_to_pdf(prediction_matrix)
    x_log = np.log(pdf)
    x_log[pdf<=0] = np.log(eps)
    # Compute the mean
    geom_av_log = np.sum(x_log * weights, axis=(0,1)) / (np.sum(weights, axis=(0,1)) + eps)
    geom_av_log = geom_av_log - np.max(geom_av_log)  # stabilizes rounding errors?

    geom_av = np.exp(geom_av_log)
    res = np.cumsum(geom_av/np.sum(geom_av))
    return res


def weighted_geom_method(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    if len(prediction_matrix.flatten()) == 0:
        return np.zeros(600)
    weights = generate_information_weight_matrix(prediction_matrix, average, expert_weights=expert_weights, *args, **kwargs)
    assert np.isfinite(weights).all()
    pdf = utils.cdf_to_pdf(prediction_matrix)
    x_log = np.log(pdf)
    x_log[pdf<=0] = np.log(eps)
    # Compute the mean
    geom_av_log = np.sum(x_log * weights, axis=(0,1)) / (np.sum(weights, axis=(0,1)) + eps)
    geom_av_log = geom_av_log - np.max(geom_av_log)  # stabilizes rounding errors?

    geom_av = np.exp(geom_av_log)
    res = np.cumsum(geom_av/np.sum(geom_av))
    return res


def weighted_arithm_no_entr(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    if len(prediction_matrix.flatten()) == 0:
        return np.zeros(600)
    weights = generate_information_weight_matrix(prediction_matrix, average, expert_weights=expert_weights, use_entropy=False, *args, **kwargs)
    assert np.isfinite(weights).all()
    res = np.sum(prediction_matrix * weights, axis=(0,1)) / (np.sum(weights, axis=(0,1)) + eps)
    return res


def weighted_arithm_method(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    if len(prediction_matrix.flatten()) == 0:
        return np.zeros(600)
    weights = generate_information_weight_matrix(prediction_matrix, average, expert_weights=expert_weights, *args, **kwargs)
    assert np.isfinite(weights).all()
    res = np.sum(prediction_matrix * weights, axis=(0,1)) / (np.sum(weights, axis=(0,1)) + eps)
    return res








if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    #required.add_argument('-c', '--config',
    #                      help='configuration to run',
    #                      required=True)
    args = parser.parse_args()

    log_path = LOGS_PATH + "merging-%s.log" % time.time()
    with print_to_file(log_path):
        print "Current git version:", utils.get_git_revision_hash()
        merge_all_prediction_files()
        print "log saved to '%s'" % log_path

