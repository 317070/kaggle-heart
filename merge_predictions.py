"""
Given a set of validation predictions, this script computes the optimal linear weights on the validation set.
It computes the weighted blend of test predictions, where some models are replaced by their bagged versions.
"""
import argparse
from functools import partial

import os
import numpy as np
import sys

import theano
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


def _load_file(path):
    with open(path, "r") as f:
        data = pickle.load(f)
    return data
"""
            average_method = partial(np.mean, axis=0)
            average_method = lambda x: np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))
"""

def softmax(z):
    z = z-np.max(z)
    res = np.exp(z) / np.sum(np.exp(z))
    return res


def optimize_expert_weights(expert_predictions,
                            targets,
                            average_distribution,
                            num_cross_validation_masks,
                            num_folds=1,
                            eps=1e-5,
                            *args, **kwargs):
    """
    :param expert_predictions: experts x validation_samples x 600 x
    :param targets: validation_samples x 600 x
    :param average_distribution: 600 x
    :param eps:
    :return:
    """
    NUM_VALIDATIONS = targets.shape[0]
    NUM_EXPERTS = expert_predictions.shape[0]
    # optimizing weights
    X = theano.shared(expert_predictions.astype('float32'))  # source predictions = (NUM_EXPERTS, NUM_VALIDATIONS, 600)
    t = theano.shared(targets.astype('float32'))  # targets = (NUM_VALIDATIONS, 600)
    W = T.vector('W', dtype='float32')  # expert weights = (NUM_EXPERTS,)

    # calculate the weighted average for each of these experts
    pdf = utils.cdf_to_pdf(expert_predictions)
    average_pdf = utils.cdf_to_pdf(average_distribution)
    average_pdf[average_pdf==0] = eps  # KL is not defined when Q=0 and P is not
    inside = pdf * (np.log(pdf) - np.log(average_pdf[None,None,:]))
    inside[pdf==0] = 0  # (xlog(x) of zero is zero)
    KL_distance_from_average = np.sum(inside, axis=2)  # (NUM_EXPERTS, NUM_VALIDATIONS)
    #print "a:", KL_distance_from_average.shape

    #print KL_distance_from_average
    cross_entropy_per_sample = - (    average_distribution[None,None,:]  * np.log(   expert_predictions+eps) +\
                                  (1.-average_distribution[None,None,:]) * np.log(1.-expert_predictions+eps) )
    cross_entropy_per_sample[cross_entropy_per_sample<0] = 0  # (NUM_EXPERTS, NUM_VALIDATIONS, 600)
    #print "b:", cross_entropy_per_sample.shape
    #print "t:", targets.shape
    weights = cross_entropy_per_sample + KL_distance_from_average[:,:,None]  # (NUM_EXPERTS, NUM_VALIDATIONS, 600)

    weight_matrix = theano.shared(weights.astype('float32'))

    # generate a bunch of cross validation masks
    x_log = np.log(pdf)
    x_log[pdf==0] = np.log(eps)
    # Compute the mean
    X_log = theano.shared(x_log)  # source predictions = (NUM_EXPERTS, NUM_VALIDATIONS, 600)

    final_weights = []
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

            def func_grad_on_indices(ind):
                X_log_i = X_log.take(ind, axis=1)
                t_i = t.take(ind, axis=0)
                w_i = weight_matrix.take(ind, axis=1)

                w_i = w_i * T.nnet.softmax(W.dimshuffle('x',0)).dimshuffle(1, 0, 'x')

                #the different predictions, are the experts
                geom_av_log = T.sum(X_log_i * w_i, axis=0) / (T.sum(w_i, axis=0) + eps)
                geom_av_log = geom_av_log - T.max(geom_av_log,axis=-1).dimshuffle(0,'x')  # stabilizes rounding errors?

                geom_av = T.exp(geom_av_log)
                res = T.cumsum(geom_av/T.sum(geom_av,axis=-1).dimshuffle(0,'x'),axis=-1)

                CRPS = T.mean((res - t_i)**2)
                grad = T.grad(CRPS, W)
                return CRPS, grad

            CRPS_train, grad_train = func_grad_on_indices(indices)
            CRPS_valid, grad_valid = func_grad_on_indices(val_indices)

            print "compiling the Theano functions"
            f = theano.function([W], CRPS_train, allow_input_downcast=True)
            g = theano.function([W], grad_train, allow_input_downcast=True)
            f_val = theano.function([W], CRPS_valid, allow_input_downcast=True)
            print "starting optimization"
            w_init = np.ones((NUM_EXPERTS,), dtype='float32')
            #out, crps, d = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)
            result = scipy.optimize.fmin_bfgs(f, w_init, fprime=g, epsilon=1e-08, maxiter=10000)
            final_weights.append(softmax(result))
            validation_score = f_val(result)
            print "         Current validation value: %.6f" % validation_score
            final_losses.append(validation_score)

    #optimal_weights = utils.geometric_average(final_weights)
    #optimal_weights = np.percentile(final_weights, axis=0)
    optimal_weights = np.mean(final_weights, axis=0)
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    average_loss    = np.mean(final_losses)
    return optimal_weights, average_loss  # (NUM_EXPERTS,)






def weighted_average_method(prediction_matrix, average, eps=1e-5, expert_weights=None, *args, **kwargs):
    pdf = utils.cdf_to_pdf(prediction_matrix)
    average_pdf = utils.cdf_to_pdf(average)
    average_pdf[average_pdf==0] = eps
    inside = pdf * (np.log(pdf) - np.log(average_pdf[None,:]))
    inside[pdf==0] = 0
    KL_distance_from_average = np.sum(inside,axis=1)

    #print KL_distance_from_average
    cross_entropy_per_sample = - ( average[None,:] * np.log(prediction_matrix+eps) + (1.-average[None,:]) * np.log(1.-prediction_matrix+eps))
    cross_entropy_per_sample[cross_entropy_per_sample<0] = 0  # numerical stability

    # geometric mean
    if expert_weights is None:
        weights = cross_entropy_per_sample + KL_distance_from_average[:,None]  #+  # <- is too big?
    else:
        weights = (cross_entropy_per_sample + KL_distance_from_average[:,None]) * expert_weights[:,None]  #+  # <- is too big?
    if True:
        x_log = np.log(pdf)
        x_log[pdf==0] = np.log(eps)
        # Compute the mean
        geom_av_log = np.sum(x_log * weights, axis=0) / (np.sum(weights, axis=0) + eps)
        geom_av_log = geom_av_log - np.max(geom_av_log)  # stabilizes rounding errors?

        geom_av = np.exp(geom_av_log)
        res = np.cumsum(geom_av/np.sum(geom_av))
    else:
        res = np.cumsum(np.sum(pdf * weights, axis=0) / np.sum(weights, axis=0))
    return res




def merge_all_prediction_files(prediction_file_location = "/mnt/storage/metadata/kaggle-heart/predictions/",
                               redo_tta = True):

    submission_path = "/mnt/storage/metadata/kaggle-heart/submissions/final_submission-%s.csv" % time.time()

    # calculate the average distribution
    regular_labels = _load_file(_TRAIN_LABELS_PATH)

    average_systole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,1]]), axis=0))
    average_diastole = make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in regular_labels[:,2]]), axis=0))

    # creating new expert opinions from the tta's generated by the experts
    expert_pkl_files = sorted(glob.glob(prediction_file_location+"je_ss_smcrps_nrmsc_500_dropnorm*.pkl")+
                              glob.glob(prediction_file_location+"je_ss_normscale_patchcontrast.pkl")+
                              glob.glob(prediction_file_location+"je_ss_smcrps_nrmsc_500_dropoutput")+
                              glob.glob(prediction_file_location+"je_ss_smcrps_jonisc128_500_dropnorm.pkl")+
                              glob.glob(prediction_file_location+"j5_normscale.pkl")
                              )

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

    NUM_EXPERTS = len(expert_pkl_files)
    NUM_VALIDATIONS = len(validation_patients_indices)

    systole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')
    diastole_expert_predictions_matrix = np.zeros((NUM_EXPERTS, NUM_VALIDATIONS, 600), dtype='float32')

    average_systole_predictions_per_file = [None] * NUM_EXPERTS
    average_diastole_predictions_per_file = [None] * NUM_EXPERTS

    for i,file in enumerate(expert_pkl_files):
        with open(file, 'r') as f:
            print "loading file",file.split('/')[-1]
            predictions = pickle.load(f)['predictions']

        if redo_tta:
            already_printed = False
            for prediction in predictions:
                if prediction["systole"].size>0 and prediction["diastole"].size>0:
                    average_method = weighted_average_method
                    prediction["systole_average"] = average_method(prediction["systole"], average=average_systole)
                    prediction["diastole_average"] = average_method(prediction["diastole"], average=average_diastole)
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
                    #TODO: is this smart to do? Are they really filtered out?
                    prediction["systole_average"] = average_systole
                    prediction["diastole_average"] = average_diastole

        for j,patient in enumerate(validation_patients_indices):
            prediction = predictions[patient-1]
            systole_expert_predictions_matrix[i,j,:] = prediction["systole_average"]
            diastole_expert_predictions_matrix[i,j,:] = prediction["diastole_average"]

        average_systole_predictions_per_file[i] = [prediction["systole_average"] for prediction in predictions]
        average_diastole_predictions_per_file[i] = [prediction["diastole_average"] for prediction in predictions]
        del predictions  # can be LOADS of data

    cv = [id-1 for id in validation_patients_indices]
    systole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,1].flatten()])

    systole_expert_weight, sys_loss = optimize_expert_weights(systole_expert_predictions_matrix,
                                                    systole_valid_labels,
                                                    average_systole,
                                                    num_cross_validation_masks=NUM_VALIDATIONS,
                                                    fold=1,
                                                    )

    diastole_valid_labels = np.array([utils.cumulative_one_hot(v) for v in regular_labels[cv,2].flatten()])
    diastole_expert_weight, dia_loss = optimize_expert_weights(diastole_expert_predictions_matrix,
                                                    diastole_valid_labels,
                                                    average_diastole,
                                                    num_cross_validation_masks=NUM_VALIDATIONS,
                                                    fold=1,
                                                    )

    print "   Final systole loss:",sys_loss
    print "  Final diastole loss:",dia_loss
    print "                     + --------------"
    print "                      ",(sys_loss + dia_loss) / 2
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
        systole_prediction_matrix = np.array([average_systole_predictions_per_file[i][patient_id-1] for i in xrange(NUM_EXPERTS)])
        diastole_prediction_matrix = np.array([average_diastole_predictions_per_file[i][patient_id-1] for i in xrange(NUM_EXPERTS)])
        final_prediction["final_systole"] = average_method(systole_prediction_matrix, average=average_systole, expert_weights=systole_expert_weight)
        final_prediction["final_diastole"] = average_method(diastole_prediction_matrix, average=average_diastole, expert_weights=diastole_expert_weight)

        try:
            test_if_valid_distribution(final_prediction["final_systole"])
            test_if_valid_distribution(final_prediction["final_diastole"])
        except:
            if not already_printed:
                print "WARNING: These FINAL distributions are not distributions"
                already_printed = True
            final_prediction["final_systole"] = make_monotone_distribution(final_prediction["final_systole"])
            final_prediction["final_diastole"] = make_monotone_distribution(final_prediction["final_diastole"])

    print
    print "Calculating training and validation set scores for reference"
    print "WARNING: both of the following are overfitted!"

    validation_dict = {}
    for patient_ids, set_name in [(validation_patients_indices, "validation"),
                                      (train_patients_indices,  "train")]:
        errors = []
        for patient in patient_ids:
            prediction = final_predictions[patient-1]
            if "final_systole" in prediction:
                assert patient == regular_labels[patient-1, 0]
                error = utils.CRSP(prediction["final_systole"], regular_labels[patient-1, 1])
                errors.append(error)
                error = utils.CRSP(prediction["final_diastole"], regular_labels[patient-1, 2])
                errors.append(error)
        if len(errors)>0:
            errors = np.array(errors)
            estimated_CRSP = np.mean(errors)
            print "  %s kaggle loss: %f" % (string.rjust(set_name, 12), estimated_CRSP)
            validation_dict[set_name] = estimated_CRSP
        else:
            print "  %s kaggle loss: not calculated" % (string.rjust(set_name, 12))




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

