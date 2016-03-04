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
from postprocess import make_monotone_distribution, test_if_valid_distribution
import glob
import cPickle as pickle
import utils
from data_loader import _TRAIN_LABELS_PATH, NUM_PATIENTS, validation_patients_indices, train_patients_indices
import csv
import string
import time
from validation_set import get_cross_validation_indices

# TODO: not hardcoded!
test_patients_indices = xrange(501,701)

def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data


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
                                       KL_weight = 1.0,
                                       cross_entropy_weight=1.0,
                                       expert_weights=None):

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
                            targets=None,
                            num_cross_validation_masks=2,
                            num_folds=1,
                            eps=1e-14,
                            cutoff=0.01,
                            do_optimization=True,
                            expert_weights=None,
                            optimal_params=None,
                            *args, **kwargs):
    """
    :param expert_predictions: experts x validation_samples x 600 x
    :param targets: validation_samples x 600 x
    :param average_distribution: 600 x
    :param eps:
    :return:
    """
    if not expert_weights is None:
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

    # calculate the weighted average for each of these experts
    weights = generate_information_weight_matrix(expert_predictions, average_distribution)

    weight_matrix = theano.shared(weights.astype('float32'))

    # generate a bunch of cross validation masks
    pdf = utils.cdf_to_pdf(expert_predictions)
    x_log = np.log(pdf)
    x_log[pdf<=0] = np.log(eps)
    # Compute the mean
    X_log = theano.shared(x_log.astype('float32'))  # source predictions = (NUM_EXPERTS, NUM_VALIDATIONS, 600)
    X_log_i = X_log.take(ind, axis=1)
    w_i = weight_matrix.take(ind, axis=1)

    W = params[:NUM_EXPERTS]
    filter_params = params[NUM_EXPERTS:NUM_EXPERTS+NUM_FILTER_PARAMETERS]
    w_i = w_i * T.nnet.softmax(W.dimshuffle('x',0)).dimshuffle(1, 0, 'x')

    #the different predictions, are the experts
    geom_av_log = T.sum(X_log_i * w_i, axis=0) / (T.sum(w_i, axis=0) + eps)
    geom_av_log = geom_av_log - T.max(geom_av_log,axis=-1).dimshuffle(0,'x')  # stabilizes rounding errors?

    geom_av = T.exp(geom_av_log)

    geom_pdf = geom_av/T.sum(geom_av,axis=-1).dimshuffle(0,'x')
    filter = T.clip( 0.75 / (T.abs_(filter_params[0]) + eps) * (1-(x_coor/( T.abs_(filter_params[0])+eps))**2), 0.0, np.float32(np.finfo(np.float64).max))
    #gauss filter =(1./(np.sqrt(2*np.pi))).astype('float32')/filter_params[0] * T.exp(-((x_coor/filter_params[0])**2)/2 )

    #geom_pdf = convolve1d(geom_pdf, filter, WINDOW_SIZE)


    cumulative_distribution = T.cumsum(geom_pdf, axis=-1)

    #exponent = np.exp(filter_params[1]-1)+eps

    #cumulative_distribution = ((1-(1-cumulative_distribution)**exponent) + (cumulative_distribution**exponent))/2

    # TODO: test this
    # nans
    #cumulative_distribution = (T.erf( T.erfinv( T.clip(cumulative_distribution*2-1, -1+eps, 1-eps) ) * filter_params[1] )+1)/2

    if not do_optimization:
        ind.set_value(range(NUM_VALIDATIONS))
        f_eval = theano.function([], cumulative_distribution)
        cumulative_distribution = f_eval()
        return cumulative_distribution[0]
    else:
        # convert to theano_values (for regularization)
        t_valid = theano.shared(targets.astype('float32'))  # targets = (NUM_VALIDATIONS, 600)

        target_values = theano.shared(np.argmax(targets, axis=-1).astype('int32'))
        noise = T.shared_randomstreams.RandomStreams(seed=317070)
        #target_values += noise.random_integers(size=target_values.shape, low=-10, high=10, dtype='int32')
        target_values += T.iround(noise.normal(size=target_values.shape, std=2, dtype='float32'))
        t_train = T.cumsum(T.extra_ops.to_one_hot(target_values, 600, dtype='float32'), axis=-1)


    CRPS_train = T.mean((cumulative_distribution - t_train.take(ind, axis=0))**2)
    CRPS_valid = T.mean((cumulative_distribution - t_valid.take(ind, axis=0))**2)

    print "compiling the Theano functions"
    iter_optimize = theano.function([], CRPS_train, on_unused_input="ignore", updates=lasagne.updates.adam(CRPS_train, [params], 1.0))
    f_val = theano.function([], CRPS_valid)

    def optimize_my_params():
        for _ in xrange(1000):  # early stopping
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
        print "filter size:", np.abs(optimal_params[NUM_EXPERTS])
        return softmax(final_weights), train_score, final_params
    else:
        final_params = []
        final_losses = []

        for fold in xrange(num_folds):
            for i_cross_validation in xrange(num_cross_validation_masks):
                print "cross_validation %d/%d"%(fold*num_cross_validation_masks+i_cross_validation+1, num_folds*num_cross_validation_masks)
                val_indices = get_cross_validation_indices(range(NUM_VALIDATIONS),
                                                       validation_index=i_cross_validation,
                                                       number_of_splits=num_cross_validation_masks,
                                                       rng_seed=fold,
                                                       )

                indices = [i for i in range(NUM_VALIDATIONS) if i not in val_indices]


                print "starting optimization"
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
        print "filter param result:", filter_param_result

        return expert_weights_result, average_loss, optimal_params  # (NUM_EXPERTS,)


def geomav(x, average=None):
    return np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x[0])))


def normalav(x, average=None):
    return np.mean(x[0], axis=0)


def prodav(x, average=None):
    return np.cumsum(utils.norm_prod(utils.cdf_to_pdf(x[0])))


def weighted_average_method(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    weights = generate_information_weight_matrix(prediction_matrix, average, expert_weights=expert_weights)
    assert np.isfinite(weights).all()
    pdf = utils.cdf_to_pdf(prediction_matrix)
    if True:
        x_log = np.log(pdf)
        x_log[pdf<=0] = np.log(eps)
        # Compute the mean
        geom_av_log = np.sum(x_log * weights, axis=(0,1)) / (np.sum(weights, axis=(0,1)) + eps)
        geom_av_log = geom_av_log - np.max(geom_av_log)  # stabilizes rounding errors?

        geom_av = np.exp(geom_av_log)
        res = np.cumsum(geom_av/np.sum(geom_av))
    else:
        res = np.cumsum(np.sum(pdf * weights, axis=(0,1)) / np.sum(weights, axis=(0,1)))
    return res




def merge_all_prediction_files(prediction_file_location = "/mnt/storage/metadata/kaggle-heart/predictions/",
                               redo_tta = False):

    submission_path = "/mnt/storage/metadata/kaggle-heart/submissions/final_submission-%s.csv" % time.time()

#    average_method = weighted_average_method
    average_method = normalav
    redo_tta = True

    # calculate the average distribution
    regular_labels = _load_file(_TRAIN_LABELS_PATH)

    average_systole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,1]]), axis=0))
    average_diastole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,2]]), axis=0))

    # creating new expert opinions from the tta's generated by the experts
    expert_pkl_files = sorted(
        glob.glob(prediction_file_location+"je_ss_jonisc64small_360.pkl")
        + glob.glob(prediction_file_location+"je_meta_fixedaggr_jsc64leakyconv.pkl")
        #+glob.glob(prediction_file_location+"ira_*.pkl")  # buggy
        #+glob.glob(prediction_file_location+"j6*.pkl")
        #+glob.glob(prediction_file_location+"j7*.pkl")
        #+glob.glob(prediction_file_location+"je_os_fixedaggr_joniscale64small_360_gauss.pkl")
        #+glob.glob(prediction_file_location+"je_ss_smcrps_nrmsc_500_dropnorm.pkl")
        #+glob.glob(prediction_file_location+"je_ss_normscale_patchcontrast.pkl")
        #+glob.glob(prediction_file_location+"je_ss_smcrps_nrmsc_500_dropoutput")
        #+glob.glob(prediction_file_location+"je_ss_smcrps_jonisc128_500_dropnorm.pkl")
        #+glob.glob(prediction_file_location+"j5_normscale.pkl")
    )

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
            expert_pkl_files.remove(file)
            print "                -> removed"
    """

    NUM_EXPERTS = len(expert_pkl_files)
    NUM_VALIDATIONS = len(validation_patients_indices)

    systole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')
    diastole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')

    average_systole_predictions_per_file = [None] * NUM_EXPERTS
    average_diastole_predictions_per_file = [None] * NUM_EXPERTS

    for i,file in enumerate(expert_pkl_files):
        with open(file, 'r') as f:
            print
            print "loading file",file.split('/')[-1]
            predictions = pickle.load(f)['predictions']

        if redo_tta:
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
                            print "WARNING: These distributions are not distributions"
                            already_printed = True
                        prediction["systole_average"] = make_monotone_distribution(prediction["systole_average"])
                        prediction["diastole_average"] = make_monotone_distribution(prediction["diastole_average"])
                        test_if_valid_distribution(prediction["systole_average"])
                        test_if_valid_distribution(prediction["diastole_average"])
                else:
                    # average distributions get zero weight later on
                    prediction["systole_average"] = average_systole
                    prediction["diastole_average"] = average_diastole
                    #prediction["systole_average"] = None
                    #prediction["diastole_average"] = None

        validation_dict = {}
        for patient_ids, set_name in [(validation_patients_indices, "validation"),
                                          (train_patients_indices,  "train")]:
            errors = []
            for patient in patient_ids:
                prediction = predictions[patient-1]
                if "systole_average" in prediction:
                    assert patient == regular_labels[patient-1, 0]
                    error = utils.CRSP(prediction["systole_average"], regular_labels[patient-1, 1])
                    errors.append(error)
                    error = utils.CRSP(prediction["diastole_average"], regular_labels[patient-1, 2])
                    errors.append(error)
            if len(errors)>0:
                errors = np.array(errors)
                estimated_CRSP = np.mean(errors)
                print "  %s kaggle loss: %f" % (string.rjust(set_name, 12), estimated_CRSP)
                validation_dict[set_name] = estimated_CRSP
            else:
                print "  %s kaggle loss: not calculated" % (string.rjust(set_name, 12))


        for j,patient in enumerate(validation_patients_indices):
            prediction = predictions[patient-1]
            # average distributions get zero weight later on
            systole_expert_predictions_matrix[i,j,:] = prediction["systole_average"] if "systole_average" in prediction else average_systole
            diastole_expert_predictions_matrix[i,j,:] = prediction["diastole_average"] if "diastole_average" in prediction else average_diastole

        # average distributions get zero weight later on
        average_systole_predictions_per_file[i] = [prediction["systole_average"] if "systole_average" in prediction else average_systole for prediction in predictions]
        average_diastole_predictions_per_file[i] = [prediction["diastole_average"] if "diastole_average" in prediction else average_diastole for prediction in predictions]
        del predictions  # can be LOADS of data


    for pass_index in xrange(1,11):
        print
        print "  PASS %d " % pass_index
        print "==========="

        cv = [id-1 for id in validation_patients_indices]

        systole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,1].flatten()])
        if pass_index==1:
            systole_expert_weight, sys_loss, systole_optimal_params = optimize_expert_weights(
                                                            systole_expert_predictions_matrix,
                                                            average_distribution=average_systole,
                                                            targets=systole_valid_labels,
                                                            num_cross_validation_masks=NUM_VALIDATIONS,
                                                            fold=1,
                                                            )
            first_pass_sys_loss = sys_loss
        else:
            systole_expert_weight, sys_loss, systole_optimal_params = optimize_expert_weights(
                                                        systole_expert_predictions_matrix,
                                                        average_distribution=average_systole,
                                                        targets=systole_valid_labels,
                                                        num_cross_validation_masks=0,
                                                        expert_weights=systole_expert_weight,
                                                        )

        diastole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,2].flatten()])
        if pass_index==1:
            diastole_expert_weight, dia_loss, diastole_optimal_params = optimize_expert_weights(
                                                            diastole_expert_predictions_matrix,
                                                            average_distribution=average_diastole,
                                                            targets=diastole_valid_labels,
                                                            num_cross_validation_masks=NUM_VALIDATIONS,
                                                            fold=1,
                                                            )
            first_pass_dia_loss = dia_loss
        else:
            diastole_expert_weight, dia_loss, diastole_optimal_params = optimize_expert_weights(
                                                        diastole_expert_predictions_matrix,
                                                        average_distribution=average_diastole,
                                                        targets=diastole_valid_labels,
                                                        num_cross_validation_masks=0,
                                                        expert_weights=diastole_expert_weight,
                                                        )
        print
        print "   Final systole loss: %.6f" % sys_loss
        print "  Final diastole loss: %.6f" % dia_loss
        print "                     + --------------"
        print "                       %.6f" % ((sys_loss + dia_loss) / 2)
        print

        # print the final weight of every expert
        print "  Systole:  Diastole: Name:"
        for expert_name, systole_weight, diastole_weight in zip(expert_pkl_files, systole_expert_weight, diastole_expert_weight):
            print string.rjust("%.3f%%" % (100*systole_weight), 10),
            print string.rjust("%.3f%%" % (100*diastole_weight), 10),
            print expert_name.split('/')[-1]


    print
    print "Average the experts according to these weights to find the final distribution"
    final_predictions = [{
                            "patient": i+1,
                            "final_systole": None,
                            "final_diastole": None
                        } for i in xrange(NUM_PATIENTS)]

    already_printed = False
    for final_prediction in final_predictions:
        patient_id = final_prediction['patient']
        print "\r   final prediction of patient %d" % patient_id,
        systole_prediction_matrix = np.array([average_systole_predictions_per_file[i][patient_id-1] for i in xrange(NUM_EXPERTS)])
        diastole_prediction_matrix = np.array([average_diastole_predictions_per_file[i][patient_id-1] for i in xrange(NUM_EXPERTS)])

        final_prediction["final_systole"] = optimize_expert_weights(
                        systole_prediction_matrix[:,None,:],
                        average_distribution=average_systole,
                        do_optimization=False,
                        optimal_params=systole_optimal_params,
                        )

        final_prediction["final_diastole"] = optimize_expert_weights(
                        diastole_prediction_matrix[:,None,:],
                        average_distribution=average_diastole,
                        do_optimization=False,
                        optimal_params=diastole_optimal_params,
                        )
        try:
            test_if_valid_distribution(final_prediction["final_systole"])
            test_if_valid_distribution(final_prediction["final_diastole"])
        except:
            final_prediction["final_systole"] = make_monotone_distribution(final_prediction["final_systole"])
            final_prediction["final_diastole"] = make_monotone_distribution(final_prediction["final_diastole"])
            test_if_valid_distribution(final_prediction["final_systole"])
            test_if_valid_distribution(final_prediction["final_diastole"])

    print
    print "Calculating training and validation set scores for reference"


    validation_dict = {}
    for patient_ids, set_name in [(validation_patients_indices, "validation"),
                                      (train_patients_indices,  "train")]:
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

    for patient_ids, set_name in [(validation_patients_indices, "validation")]:
        print "patient:  disagreement:  CRPS-contr:"
        for patient in sorted(patient_ids, key=lambda id: -final_predictions[id-1]["average_crps_error"]):
            prediction = final_predictions[patient-1]
            if "final_systole" in prediction:
                systole_prediction_matrix = np.array([average_systole_predictions_per_file[i][patient-1] for i in xrange(NUM_EXPERTS)])
                diastole_prediction_matrix = np.array([average_diastole_predictions_per_file[i][patient-1] for i in xrange(NUM_EXPERTS)])
                disagreement = 0.5*(get_expert_disagreement(systole_prediction_matrix, systole_expert_weight)
                                  + get_expert_disagreement(diastole_prediction_matrix, diastole_expert_weight))
                print string.rjust("%d" % patient,8),
                print string.rjust("%.5f" % disagreement,15),
                if "systole_crps_error" in prediction:
                    error1 = prediction["systole_crps_error"]
                    error2 = prediction["diastole_crps_error"]
                    crps_contribution = 1000*(error1+error2) / 2. / len(patient_ids)
                    print string.rjust("%.5f" % crps_contribution,13),
                print

    print
    print "estimated leaderboard loss: %f" % ((first_pass_sys_loss + first_pass_dia_loss)/2)
    print


    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d'%i for i in xrange(600)])
        for prediction in final_predictions:
            # the submission only has patients 501 to 700
            if 500 < prediction["patient"] <= 700:
                if "final_diastole" not in prediction or "final_systole" not in prediction:
                    raise Exception("Not all test-set patients were predicted")
                csvwriter.writerow(["%d_Diastole" % prediction["patient"]] + ["%.18f" % p for p in prediction["final_diastole"].flatten()])
                csvwriter.writerow(["%d_Systole" % prediction["patient"]] + ["%.18f" % p for p in prediction["final_systole"].flatten()])
    print "submission file dumped"



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




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    required = parser.add_argument_group('required arguments')
    #required.add_argument('-c', '--config',
    #                      help='configuration to run',
    #                      required=True)
    args = parser.parse_args()

    with print_to_file("/mnt/storage/metadata/kaggle-heart/logs/merging.log"):
        print "Current git version:", utils.get_git_revision_hash()
        merge_all_prediction_files()
        print "log saved to '%s'" % ("/mnt/storage/metadata/kaggle-heart/logs/merging.log")

