"""Ensembling script using weighted averaging and outlier detection. 

This script loads all the prediction files in the predictions folder.
For every file, it computes the best way to do the averaging across TTAs, by
looking at the validation set. Then, it makes 3 ensembles. The first ensemble
contains every single model. The second ensemble excludes patient models, since
these are generally not robust to outliers. When these two disagree too much on
a given patient, that patient is considered an outlier. The final ensemble is
trained on all patients that are not outliers. Outliers are predicted using the 
second ensemble. 

To create an ensemble, a set of weights (summing ot one) is optimised using
projected gradient descend. Then, the top models are selected, and the weights
are re-optimised for only those models.

This script also exports the weights and other results. If these are passed as 
an argument, the script will work without a validation set, and simply use 
the averaging methods and ensemble weights as defined in the weights file.

Usage:
    To simply generate an ensemble and export the weights:
    > python merge_predictions_jeroen

    To reuse weights that were computed by an earlier run of this script:
    > python merge_predictions_jeroen PATH_TO_WEIGHTS_FILE
"""
import cPickle as pickle
import csv
import glob
import os
import time

import numpy as np
import theano
import theano.tensor as T

import data_loader
import paths
import postprocess
import utils
import sys


###### PARAMTERS #######
#======================#

DO_XVAL = True  # Flag specifying whether to do cross-validation or not.

# Loading all the predictions takes a while. Once they are loaded, the
# intermediate results are saved to the following location:
METADATAS_LOCATION = 'metadatas.pkl'
# ... and if the following flag is set to True, the predictions are loaded from
# that location instead. This makes loading faster, but new predictions will not
# be processed.
RELOAD_METADATAS = False  

# Some paths
PREDICTIONS_PATH = paths.INTERMEDIATE_PREDICTIONS_PATH
SUBMISSION_PATH = paths.SUBMISSION_PATH
WEIGHTS_PATH = paths.ENSEMBLE_WEIGHTS_PATH

# Printing to console
PRINT_FOLDS = False  # Print the result of every xval fold
PRINT_EVERY = 1000  # Print itermediate results during training

# Projected SGD parameters
LEARNING_RATE = 1  # Learning rate of gradient descend
NR_EPOCHS = 100
C_reg1 = 0.0000  # L1 regularisation parameters
C_reg2 = 0.01  # L2 regularisation parameters
GRAD_CLIP = 1

EPS_TOP = 0.000001  # Minnimal possible weight
SELECT_TOP = 15  # Nr of models to select

OUTLIER_THRESHOLD = 0.020  # Disagreement threshold

print C_reg1
print C_reg2


###### LOADING PREDICTION FILES ########
#======================================#


def filter_jeroen_patientmodels(filename):
    return (
        filename.startswith('je_os')
        or filename.startswith('je_meta')
        or filename.startswith('je_rs'))


def filter_ira_patientmodels(filename):
    return filename.startswith('ira_configurations.meta')


def filter_lio_patientmodels(filename):
    return filename.startswith('je_lio_rnn')


def filter_jonas_bullcrap(filename):
    return (
        filename.startswith('j0')
        or filename.startswith('j1')
        or filename.startswith('j2')
        or filename.startswith('j3')
        or filename.startswith('j4')
        or filename.startswith('j5'))


def filter_jonas_patientmodels(filename):
    return filename.startswith('j7')


ONLY_SLICE_MODELS_FILTERS = (
    filter_jeroen_patientmodels,
    filter_ira_patientmodels,
    filter_lio_patientmodels,
    filter_jonas_bullcrap,
    filter_jonas_patientmodels)


ALL_MODELS_EXCEPT_IRA_FILTERS = (
    filter_jonas_bullcrap,
    filter_ira_patientmodels)


ALL_MODELS_EXCEPT_CRAPPY_ONES_FILTERS = (
    filter_jonas_bullcrap,)


FILTERS_TO_USE = ALL_MODELS_EXCEPT_CRAPPY_ONES_FILTERS


def _get_all_prediction_files():
    """Returns all the prediction files in the predictions folder.
    The files are filtered first.
    """
    # Get all of them
    all_files = glob.glob(PREDICTIONS_PATH + '*')
    # And filter them
    res = []
    for f in all_files:
        filename = os.path.basename(f)
        use_file = True
        for filt in FILTERS_TO_USE:
            if filt(filename):
                use_file = False
                break
        if use_file:
            res.append(f)

    return res


def _construct_all_patient_ids():
    all_patient_ids = {key: [] for key in data_loader.patient_folders}
    for pid in data_loader.id_to_index_map:
        pset, pindex = data_loader.id_to_index_map[pid]
        all_patient_ids[pset].append(pid)
    for arr in all_patient_ids.values():
        arr.sort()
    return all_patient_ids


# This map contains the ids for all patients in the train, test and validation split
all_patient_ids = _construct_all_patient_ids()


def load_prediction_file(path):
    try:
        m = np.load(path)
        m['path'] = path
        return m
    except:
        return False


def _is_empty_prediction(patient_prediction):
    return (
      len(patient_prediction['systole']) == 0
      and len(patient_prediction['diastole']) == 0
      and 'patient' in patient_prediction)


not_predicted_sets = {}


def _register_model(pats_not_predicted):
    not_predicted = tuple(sorted(pats_not_predicted['validation'] + pats_not_predicted['test']))
    not_predicted_sets[not_predicted] = not_predicted_sets.get(not_predicted, 0) + 1


def _compute_pats_not_predicted(metadata):
    pats_not_predicted = {set:[] for set in data_loader.patient_folders}
    pats_predicted = {set:[] for set in data_loader.patient_folders}
    for prediction in metadata['predictions']:
        pid = prediction['patient']
        pset = data_loader.id_to_index_map[pid][0]
        if _is_empty_prediction(prediction):
            pats_not_predicted[pset].append(pid)
        else:
            pats_predicted[pset].append(pid)
    return pats_not_predicted, pats_predicted


##### TTA AVERAGING ######
#========================#


average_systole = postprocess.make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:,1]]), axis=0))
average_diastole = postprocess.make_monotone_distribution(np.mean(np.array([utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:,2]]), axis=0))


def generate_information_weight_matrix(expert_predictions, average_distribution, eps=1e-14, KL_weight = 1.0, cross_entropy_weight=1.0, expert_weights=None):
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


def weighted_average_method(prediction_matrix, average, eps=1e-14, expert_weights=None, *args, **kwargs):
    if len(prediction_matrix) == 0:
        return np.zeros(600)
    prediction_matrix = prediction_matrix[None, :, :]
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


def geomav(x, **kwargs):
    if len(x) == 0:
        return np.zeros(600)
    res = np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))
    return res


def normalav(x, **kwargs):
    if len(x) == 0:
        return np.zeros(600)
    return np.mean(x, axis=0)


def prodav(x, **kwargs):
    if len(x) == 0:
        return np.zeros(600)
    return np.cumsum(utils.norm_prod(utils.cdf_to_pdf(x)))


AVERAGING_METHODS = (normalav, geomav, weighted_average_method)


def _validate_metadata(metadata, pats_predicted):
    """Compute validation score for given patients.
    """
    errors = []
    for pid in pats_predicted:
        prediction_pid = metadata['predictions'][pid-1]
        assert prediction_pid['patient'] == pid
        error_sys = utils.CRSP(prediction_pid["systole_average"], data_loader.regular_labels[pid-1, 1])
        error_dia = utils.CRSP(prediction_pid["diastole_average"], data_loader.regular_labels[pid-1, 2])
        errors += [error_sys, error_dia]
    return np.mean(errors)


def _compute_tta(metadata, averaging_method):
    for prediction_pid in metadata['predictions']:
        pid = prediction_pid['patient']
        prediction_pid["systole_average"] = averaging_method(prediction_pid["systole"], average=average_systole)
        prediction_pid["diastole_average"] = averaging_method(prediction_pid["diastole"], average=average_diastole)


def _compute_nr_ttas(metadata, pats_predicted):
    if pats_predicted:
        return len(metadata['predictions'][pats_predicted[0]]["systole"])
    else:
        return -1

def _make_valid_distributions(metadata):
    for prediction in metadata['predictions']:
        prediction["systole_average"] = postprocess.make_monotone_distribution(prediction["systole_average"])
        postprocess.test_if_valid_distribution(prediction["systole_average"])
        prediction["diastole_average"] = postprocess.make_monotone_distribution(prediction["diastole_average"])
        postprocess.test_if_valid_distribution(prediction["diastole_average"])


def _add_tta_score_to_metadata(metadata, tta_score, best_method):
    metadata['tta_score'] = tta_score
    metadata['best_method'] = best_method.func_name


def _compute_best_tta(metadata, pats_predicted, best_method=None):
    print "    Using %d patients, averaging over %d TTAs" % (
        len(pats_predicted), _compute_nr_ttas(metadata, pats_predicted))
    if not best_method:
        best_score = 1
        best_method = None
        for averaging_method in AVERAGING_METHODS:
            _compute_tta(metadata, averaging_method)
            err = _validate_metadata(metadata, pats_predicted)
            print "    - %s: %.5f" % (averaging_method.func_name, err)
            if err < best_score:
                best_score, best_method = err, averaging_method
    else:
        print "    Using predefined averaging"
        for averaging_method in AVERAGING_METHODS:
            if best_method == averaging_method.func_name:
                best_method = averaging_method
                break
    _compute_tta(metadata, best_method)
    _make_valid_distributions(metadata)
    tta_score = _validate_metadata(metadata, pats_predicted)
    print "    Choosing %s (%.5f)" % (best_method.func_name, tta_score)
    _add_tta_score_to_metadata(metadata, tta_score, best_method)


def _remove_ttas(metadata, tags_to_remove=('systole', 'diastole')):
    for prediction in metadata['predictions']:
        # Tag the prediction to keep track of whether it was empty or not
        prediction['is_empty'] = _is_empty_prediction(prediction)
        for tag in tags_to_remove:
            if tag in prediction:
                del prediction[tag]


### LOADING AND PROCESSING PREDICTIONS ###
#========================================#


def _process_prediction_file(path, tta_methods=None, noskip=False):
    metadata = load_prediction_file(path)
    if not metadata:
        print "  Couldn't load file"
        return

    # Compute for which patients there are no predictions
    pats_not_predicted, pats_predicted = _compute_pats_not_predicted(metadata)
    _register_model(pats_not_predicted)

    nr_val_predicted = len(pats_predicted['validation'])
    nr_test_predicted = len(pats_predicted['test'])
    print "  val: %3d/%3d,  test: %3d/%3d" % (
        nr_val_predicted, data_loader.NUM_VALID_PATIENTS,
        nr_test_predicted, data_loader.NUM_TEST_PATIENTS,)
    if not noskip and (nr_val_predicted == 0 or nr_test_predicted == 0):
        print "  Skipping this model, not enough predictions."
        return

    # Compute best way of averaging
    if tta_methods:
        print "  Using predetermined way of doing TTA"
        best_method = tta_methods[metadata['configuration_file']]
        _compute_best_tta(metadata, pats_predicted['validation'], best_method)
    else:
        print "  Trying out different ways of doing TTA:"
        _compute_best_tta(metadata, pats_predicted['validation'])

    # Clean up the metadata file and return it
    _remove_ttas(metadata)
    return metadata


def _load_and_process_metadata_files(tta_methods=None, noskip=False):
    all_prediction_files = sorted(_get_all_prediction_files())[:]
    nr_prediction_files = len(all_prediction_files)

    print "Using the following files:"
    print
    print "\n".join(map(os.path.basename, all_prediction_files))

    useful_files = []

    for idx, path in enumerate(all_prediction_files):
        print
        print 'Processing %s (%d/%d)' % (os.path.basename(path), idx+1, nr_prediction_files)
        m = _process_prediction_file(path, tta_methods, noskip=noskip)
        if m:
            useful_files.append(m)

    print
    print "Loaded %d files" % len(useful_files)
    return useful_files


######### ENSEMBLING #########
#============================#


def _create_prediction_matrix(metadatas):
    nr_models = len(metadatas)
    nr_patients = data_loader.NUM_PATIENTS

    res_mask = np.zeros((nr_models, nr_patients))
    res_sys = np.zeros((nr_models, nr_patients, 600))
    res_dia = np.zeros((nr_models, nr_patients, 600))

    for i, metadata in enumerate(metadatas):
        for j, prediction in enumerate(metadata['predictions']):
            res_mask[i, j] = not prediction["is_empty"]
            res_sys[i, j] = prediction["systole_average"]
            res_dia[i, j] = prediction["diastole_average"]
            assert prediction["patient"] == j+1

    return res_mask, res_sys, res_dia


def _create_label_matrix():
    systole_valid_labels = np.array(
        [utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:,1].flatten()])
    diastole_valid_labels = np.array(
        [utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:,2].flatten()])
    return systole_valid_labels, diastole_valid_labels


def _get_train_val_test_ids():
    sets = [data_loader.id_to_index_map[pid][0] for pid in range(1, data_loader.NUM_PATIENTS+1)]

    return np.array(sets)=="train", np.array(sets)=="validation", np.array(sets)=="test"


def _find_weights(w_init, preds_matrix, targets_matrix, mask_matrix, eps=0.0, use_reg=True, slice_models_idx=None):

    nr_models = len(w_init)
    nr_patients = mask_matrix.shape[1]

    learning_rate = LEARNING_RATE
    nr_epochs = NR_EPOCHS

    if PRINT_FOLDS: print "      Compiling function"

    # Create theano expression
    # inputs:
    weights = theano.shared(w_init.astype('float32'))
    preds = theano.shared(preds_matrix.astype('float32'))
    targets = theano.shared(targets_matrix.astype('float32'))
    mask = theano.shared(mask_matrix.astype('float32'))

    # expression
    masked_weights = mask * weights.dimshuffle(0, 'x')
    tot_masked_weights = masked_weights.sum(axis=0)
    preds_weighted_masked = preds * masked_weights.dimshuffle(0, 1, 'x')
    av_preds = preds_weighted_masked.sum(axis=0) / tot_masked_weights.dimshuffle(0, 'x')
    # loss
    l1_loss = weights.sum()
    l2_loss = (weights**2).sum()
    loss = ((av_preds - targets)**2).mean() + use_reg * C_reg1 * l1_loss + use_reg * C_reg2 * l2_loss

    # Update function (keeping the weights normalised)
    grad_weights = theano.grad(loss, weights)
    grad_weights_norm = (grad_weights ** 2).sum()
    grad_weights = grad_weights * GRAD_CLIP / T.clip(grad_weights_norm, GRAD_CLIP, utils.maxfloat)
    delta = T.clip(learning_rate * grad_weights, -.1, .1)
    updated_weights = T.clip((weights - delta), eps, 10)  # Dont allow weights smaller than 0
    updated_weights_normalised = T.clip(updated_weights / updated_weights.sum(), eps, 1)  # Renormalise
    updates = {weights: updated_weights_normalised}
    iter_train = theano.function([], loss, updates=updates)

    # Do training
    if PRINT_FOLDS: print "      Training"
    for iteration in xrange(nr_epochs):
        train_err = iter_train()
        w_value = np.array(weights.eval())
        if (iteration+1) % PRINT_EVERY == 0:
            print iteration
            if PRINT_FOLDS: print "      train_error: %.4f, weights: %s" % (train_err, str((w_value*1000).astype('int32')))

    return np.array(weights.eval())  # Convert cudaNArray to numpy if necessairy



def _compute_predictions_ensemble(weights, preds, mask):
    masked_weights = mask * weights[:, np.newaxis]
    tot_masked_weights = masked_weights.sum(axis=0)
    preds_weighted_masked = preds * masked_weights[:, :, np.newaxis]
    av_preds = preds_weighted_masked.sum(axis=0) / tot_masked_weights[:, np.newaxis]
    return av_preds


def _eval_weights(weights, preds, targets, mask):
    av_preds = _compute_predictions_ensemble(weights, preds, mask)

    crps = ((av_preds - targets)**2).mean()
    return crps


def _ensemble_result_to_metadata(slice_ensemble_results):
    predictions = [
        {
            "patient": idx + 1,
            "is_empty": False,
            "systole_average": sys,
            "diastole_average": dia,
        } for idx, (sys, dia) in enumerate(zip(
            slice_ensemble_results["predictions_systole"],
            slice_ensemble_results["predictions_diastole"]))]

    return {
        "predictions": predictions,
        "configuration_file": "slice_ensemble",
        "best_method": None,
    }


def has_nan(x):
    return np.isnan(np.sum(x))


def get_weight_vector(metadatas, weights):
    w_sys = np.array([weights.get(metadata['configuration_file'], [0,0])[0] for metadata in metadatas])
    w_dia = np.array([weights.get(metadata['configuration_file'], [0,0])[1] for metadata in metadatas])
    return w_sys, w_dia


def _create_ensembles(metadatas, outliers=None, slice_ensemble_results=None, weights=None):

    if outliers is not None and slice_ensemble_results is not None:
        ensemble_metadata = _ensemble_result_to_metadata(slice_ensemble_results)
        metadatas.append(ensemble_metadata)

    # aggregate predictions and targets
    mask, preds_sys, preds_dia = _create_prediction_matrix(metadatas)
    targets_sys, targets_dia = _create_label_matrix()

    # take outliers into account
    if outliers is not None and slice_ensemble_results is not None:
        print "Taking outliers into account"
        # remove outliers
        mask = mask * np.logical_not(outliers)[np.newaxis, :]
        mask[-1, :] = outliers

    if weights is not None:
        print "Loading predefined weights"
        w_sys, w_dia = get_weight_vector(metadatas, weights)
        print
        print "dia    sys "
        for metadata, weight_sys, weight_dia in zip(metadatas, w_sys, w_dia):
            print "%5.2f  %5.2f : %s" % (weight_sys*100, weight_dia*100, metadata["configuration_file"])
        return {
            "weights": weights,
            "predictions_systole": _compute_predictions_ensemble(w_sys, preds_sys, mask),
            "predictions_diastole": _compute_predictions_ensemble(w_dia, preds_dia, mask),
        }

    # initialise weights
    nr_models = len(metadatas)
    w_init = np.ones((nr_models,), dtype='float32')/nr_models

    # split data
    _, val_idx, test_idx = _get_train_val_test_ids()
    mask_val, mask_test = mask[:, val_idx], mask[:, test_idx]
    targets_sys_val = targets_sys[val_idx[:len(targets_sys)], :]
    targets_dia_val = targets_dia[val_idx[:len(targets_dia)], :]
    preds_sys_val, preds_sys_test = preds_sys[:, val_idx, :], preds_sys[:, test_idx, :]
    preds_dia_val, preds_dia_test = preds_dia[:, val_idx, :], preds_dia[:, test_idx, :]


    ## MAKE ENSEMBLE USING THE FULL VALIDATION SET
    print "Fitting weights on the entire validation set"
    w_sys = _find_weights(w_init, preds_sys_val, targets_sys_val, mask_val, eps=EPS_TOP)
    w_dia = _find_weights(w_init, preds_dia_val, targets_dia_val, mask_val, eps=EPS_TOP)

    # Print the result
    print
    print "dia    sys "
    sort_key = lambda x: - x[1] - x[2]
    for metadata, weight_sys, weight_dia in sorted(zip(metadatas, w_sys, w_dia), key=sort_key):
        print "%5.2f  %5.2f : %s" % (weight_sys*100, weight_dia*100, metadata["configuration_file"])


    ## SELECT THE TOP MODELS AND RETRAIN ONCE MORE
    print
    print "Selecting the top %d models and retraining" % SELECT_TOP
    sorted_models, sorted_w_sys, sorted_w_dia = zip(*sorted(zip(metadatas, w_sys, w_dia), key=sort_key))
    top_models = list(sorted_models[:SELECT_TOP])
    sorted_w_sys = list(sorted_w_sys)
    sorted_w_dia = list(sorted_w_dia)
    if slice_ensemble_results is not None and not ensemble_metadata in top_models:
        print "Forcing slice ensemble to be in the mix"
        top_models[-1] = ensemble_metadata
        sorted_w_sys[SELECT_TOP-1] = EPS_TOP
        sorted_w_dia[SELECT_TOP-1] = EPS_TOP

    w_init_sys_top = np.array(sorted_w_sys[:SELECT_TOP]) / np.array(sorted_w_sys[:SELECT_TOP]).sum()
    w_init_dia_top = np.array(sorted_w_dia[:SELECT_TOP]) / np.array(sorted_w_dia[:SELECT_TOP]).sum()
    mask_top, preds_sys_top, preds_dia_top = _create_prediction_matrix(top_models)

    slice_models_idx = None
    if outliers is not None and slice_ensemble_results is not None:        # remove outliers
        mask_top = mask_top * np.logical_not(outliers)[np.newaxis, :]
        for idx, model in enumerate(top_models):
            if model is ensemble_metadata:
                mask_top[idx, :] = outliers
                slice_models_idx = idx

    mask_top_val, mask_top_test = mask_top[:, val_idx], mask_top[:, test_idx]
    preds_sys_top_val, preds_sys_top_test = preds_sys_top[:, val_idx, :], preds_sys_top[:, test_idx, :]
    preds_dia_top_val, preds_dia_top_test = preds_dia_top[:, val_idx, :], preds_dia_top[:, test_idx, :]

    w_sys_top = _find_weights(w_init_sys_top, preds_sys_top_val, targets_sys_val, mask_top_val, eps=EPS_TOP)
    w_dia_top = _find_weights(w_init_dia_top, preds_dia_top_val, targets_dia_val, mask_top_val, eps=EPS_TOP)

    sys_err = _eval_weights(w_sys_top, preds_sys_top_val, targets_sys_val, mask_top_val)
    dia_err = _eval_weights(w_dia_top, preds_dia_top_val, targets_dia_val, mask_top_val)


    print
    print "dia    sys "
    sort_key = lambda x: - x[1] - x[2]
    for metadata, weight_sys, weight_dia in zip(top_models, w_sys_top, w_dia_top):
        print "%5.2f  %5.2f : %s" % (weight_sys*100, weight_dia*100, metadata["configuration_file"])


    ## LOOXVAL
    if DO_XVAL:
        print "  Doing leave one patient out xval (systole)"
        nr_val_patients = len(targets_sys_val)
        
        train_errs_sys = []
        val_errs_sys = []
        w_iters_sys = []

        train_errs_dia = []
        val_errs_dia = []
        w_iters_dia = []
        
        for val_pid in xrange(nr_val_patients):
            if PRINT_FOLDS: print "    - fold %d" % val_pid
            preds_sys_val_i = np.hstack((preds_sys_val[:, :val_pid], preds_sys_val[:, val_pid+1:]))
            targets_sys_val_i = np.vstack((targets_sys_val[:val_pid], targets_sys_val[val_pid+1:]))
            preds_dia_val_i = np.hstack((preds_dia_val[:, :val_pid], preds_dia_val[:, val_pid+1:]))
            targets_dia_val_i = np.vstack((targets_dia_val[:val_pid], targets_dia_val[val_pid+1:]))
            mask_val_i = np.hstack((mask_val[:, :val_pid], mask_val[:, val_pid+1:]))
                
            # Fit weights on entire thing
            w_sys_i = _find_weights(w_init, preds_sys_val_i, targets_sys_val_i, mask_val_i, eps=EPS_TOP)
            w_dia_i = _find_weights(w_init, preds_dia_val_i, targets_dia_val_i, mask_val_i, eps=EPS_TOP)

            # Select models
            sorted_models_i, sorted_w_sys_i, sorted_w_dia_i = zip(*sorted(zip(metadatas, w_sys_i, w_dia_i), key=sort_key))
            top_models_i = list(sorted_models_i[:SELECT_TOP])
            sorted_w_sys_i = list(sorted_w_sys_i)
            sorted_w_dia_i = list(sorted_w_dia_i)
            if slice_ensemble_results is not None and not ensemble_metadata in top_models_i:
                top_models_i[-1] = ensemble_metadata
                sorted_w_sys_i[SELECT_TOP-1] = EPS_TOP
                sorted_w_dia_i[SELECT_TOP-1] = EPS_TOP

            w_init_sys_top_i = np.array(sorted_w_sys_i[:SELECT_TOP]) / np.array(sorted_w_sys_i[:SELECT_TOP]).sum()
            w_init_dia_top_i = np.array(sorted_w_dia_i[:SELECT_TOP]) / np.array(sorted_w_dia_i[:SELECT_TOP]).sum()
            
            mask_top_, preds_sys_top_, preds_dia_top_ = _create_prediction_matrix(top_models_i)
            
            slice_models_idx_i = None
            if outliers is not None and slice_ensemble_results is not None:
                # remove outliers
                mask_top_ = mask_top_ * np.logical_not(outliers)[np.newaxis, :]
                for idx, model in enumerate(top_models):
                    if model is ensemble_metadata:
                        mask_top_[idx, :] = outliers
                        slice_models_idx_i = idx 

            mask_top_val_ = mask_top_[:, val_idx]
            mask_top_val_i = np.hstack((mask_top_val_[:, :val_pid], mask_top_val_[:, val_pid+1:]))
            
            preds_sys_top_val_ = preds_sys_top_[:, val_idx, :]
            preds_sys_top_val_i = np.hstack((preds_sys_top_val_[:, :val_pid], preds_sys_top_val_[:, val_pid+1:]))

            preds_dia_top_val_ = preds_dia_top_[:, val_idx, :]
            preds_dia_top_val_i = np.hstack((preds_dia_top_val_[:, :val_pid], preds_dia_top_val_[:, val_pid+1:]))
            
            w_sys_top_i = _find_weights(w_init_sys_top_i, preds_sys_top_val_i, targets_sys_val_i, mask_top_val_i, eps=EPS_TOP)
            w_dia_top_i = _find_weights(w_init_dia_top_i, preds_dia_top_val_i, targets_dia_val_i, mask_top_val_i, eps=EPS_TOP)

            sys_err_i = _eval_weights(w_sys_top_i, preds_sys_top_val[:, val_pid:val_pid+1], targets_sys_val[val_pid:val_pid+1], mask_top_val_[:, val_pid:val_pid+1])
            dia_err_i = _eval_weights(w_dia_top_i, preds_dia_top_val[:, val_pid:val_pid+1], targets_dia_val[val_pid:val_pid+1], mask_top_val_[:, val_pid:val_pid+1])
            sys_err_train_i = _eval_weights(w_sys_top_i, preds_sys_top_val_i, targets_sys_val_i, mask_top_val_i)
            dia_err_train_i = _eval_weights(w_dia_top_i, preds_dia_top_val_i, targets_dia_val_i, mask_top_val_i)

            if PRINT_FOLDS: print "      (sys) train_err: %.4f,  val_err: %.4f" % (sys_err_train_i, sys_err_i)
            if PRINT_FOLDS: print "      (dia) train_err: %.4f,  val_err: %.4f" % (dia_err_train_i, dia_err_i)

            train_errs_sys.append(sys_err_train_i)
            val_errs_sys.append(sys_err_i)
            w_iters_sys.append(w_sys_top_i)
            train_errs_dia.append(dia_err_train_i)
            val_errs_dia.append(dia_err_i)
            w_iters_dia.append(w_dia_top_i)        


        expected_systole_loss = np.mean(val_errs_sys)
        expected_diastole_loss = np.mean(val_errs_dia)
        print " Results systole:"
        print "  average train err: %.4f" % np.mean(train_errs_sys)
        print "  average valid err: %.4f" % np.mean(val_errs_sys)
        print " Results diastole:"
        print "  average train err: %.4f" % np.mean(train_errs_dia)
        print "  average valid err: %.4f" % np.mean(val_errs_dia)
        print " Results average:"
        print "  average train err: %.4f" % (np.mean(train_errs_dia)/2.0 + np.mean(train_errs_sys)/2.0)
        print "  average valid err: %.4f" % (np.mean(val_errs_dia)/2.0 + np.mean(val_errs_sys)/2.0)

        
    print "Final scores on the validation set:"
    print "  systole:  %.4f" % sys_err
    print "  diastole: %.4f" % dia_err
    print "  average:  %.4f" % ((sys_err + dia_err) / 2.0)
    if DO_XVAL:
        print "Expected leaderboard scores:"
        print "  systole:  %.4f" % expected_systole_loss
        print "  diastole: %.4f" % expected_diastole_loss
        print "  average:  %.4f" % ((expected_systole_loss + expected_diastole_loss) / 2.0)


    ## COMPUTE TEST PREDICTIONS
    preds_sys = _compute_predictions_ensemble(w_sys_top, preds_sys_top_test, mask_top_test)
    preds_dia = _compute_predictions_ensemble(w_dia_top, preds_dia_top_test, mask_top_test)


    ## STORE RESULTS IN DICTS AND RETURN
    weights = {
        metadata["configuration_file"]: (weight_sys, weight_dia) for metadata, weight_sys, weight_dia in zip(top_models, w_sys_top, w_dia_top)
    }

    return {
        "weights": weights,
        "predictions_systole": _compute_predictions_ensemble(w_sys_top, preds_sys_top, mask_top),
        "predictions_diastole": _compute_predictions_ensemble(w_dia_top, preds_dia_top, mask_top),
    }


def dump_metadatas(metadatas):
    with open(METADATAS_LOCATION, 'w') as f:
        pickle.dump(metadatas, f, pickle.HIGHEST_PROTOCOL)
    print "metadatas file dumped"


def load_metadatas():
    metadatas = np.load(METADATAS_LOCATION)
    print "Loaded metadatas file"
    return metadatas


def _filter_metadatas(metadatas, filters):
    res = []
    for m in metadatas:
        filename = os.path.basename(m['path'])
        use_file = True
        for filt in filters:
            if filt(filename):
                use_file = False
                break
        if use_file:
            res.append(m)
    return res


def dump_predictions(result_ensemble, submission_path):

    _, _, test_idx = _get_train_val_test_ids()
    test_ids = np.where(test_idx)[0] + 1

    preds_sys = result_ensemble["predictions_systole"][np.where(test_idx)]
    preds_dia = result_ensemble["predictions_diastole"][np.where(test_idx)]
    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d'%i for i in xrange(600)])
        for pid, pat_pred_sys, pat_pred_dia in zip(test_ids, preds_sys, preds_dia):
            pat_pred_sys = postprocess.make_monotone_distribution(pat_pred_sys)
            pat_pred_dia = postprocess.make_monotone_distribution(pat_pred_dia)
            csvwriter.writerow(["%d_Diastole" % pid] + ["%.18f" % p for p in pat_pred_dia.flatten()])
            csvwriter.writerow(["%d_Systole" % pid] + ["%.18f" % p for p in pat_pred_sys.flatten()])
    print "submission file dumped"


def compute_outliers(result_ensemble_everything, result_ensemble_slices, threshold=OUTLIER_THRESHOLD):
    crps_sys = ((result_ensemble_everything["predictions_systole"] - result_ensemble_slices["predictions_systole"])**2).mean(axis=1)
    crps_dia = ((result_ensemble_everything["predictions_diastole"] - result_ensemble_slices["predictions_diastole"])**2).mean(axis=1)
    crps_av = 0.5*crps_sys + 0.5*crps_dia

    is_outlier = crps_av > threshold
    print "  outliers (%.4f): " % threshold, str(np.where(is_outlier)[0]+1)
    return is_outlier


def combine_results(result_ensemble_everything, result_ensemble_slices, outliers):
    pred_sys = result_ensemble_everything["predictions_systole"]
    pred_sys[np.where(outliers)] = result_ensemble_slices["predictions_systole"][np.where(outliers)]
    pred_dia = result_ensemble_everything["predictions_diastole"]
    pred_dia[np.where(outliers)] = result_ensemble_slices["predictions_diastole"][np.where(outliers)]
    res = {
        "predictions_systole": pred_sys,
        "predictions_diastole": pred_dia,
    }
    return res


def main(weights_dict=None):
    # Load all metadatas
    print
    print "LOADING PREDICTION FILES"
    print
    if RELOAD_METADATAS:
        metadatas = load_metadatas()
    else:
        if weights_dict:
            metadatas = _load_and_process_metadata_files(weights_dict['tta_methods'], noskip=True)
        else:
            metadatas = _load_and_process_metadata_files()
        dump_metadatas(metadatas)
    metadatas_slices = _filter_metadatas(metadatas, ONLY_SLICE_MODELS_FILTERS)

    # Create an ensemble of everything
    print
    print "CREATING ENSEMBLE (ALL)"
    print
    if not weights_dict:
        result_ensemble_everything = _create_ensembles(metadatas)
    else:
        result_ensemble_everything = _create_ensembles(metadatas, weights=weights_dict['weights_all'])

    # Create an ensemble of only slicemodels
    print
    print "CREATING ENSEMBLE (SLICES)"
    print
    if not weights_dict:
        result_ensemble_slices = _create_ensembles(metadatas_slices)
    else:
        result_ensemble_slices = _create_ensembles(metadatas_slices, weights=weights_dict['weights_slices'])

    # Compute outliers
    print
    print "COMPUTING OUTLIERS"
    print
    if not weights_dict:
        outliers = compute_outliers(result_ensemble_slices, result_ensemble_everything)
    else:
        outliers = weights_dict['outliers']

    # Retrain
    print
    print "CREATING ENSEMBLE (SLICE + ALL w/ outliers)"
    print
    if not weights_dict:
        result_ensemble_final = _create_ensembles(metadatas, outliers=outliers, slice_ensemble_results=result_ensemble_slices)
    else:
        result_ensemble_final = _create_ensembles(metadatas, outliers=outliers, slice_ensemble_results=result_ensemble_slices, weights=weights_dict['weights_final'])

    # Dump predictions
    dump_predictions(result_ensemble_everything, SUBMISSION_PATH + 'ensemble_w_outliers.' + str(time.time()) + '.csv')
    dump_predictions(result_ensemble_slices, SUBMISSION_PATH + 'ensemble_slices.' + str(time.time()) + '.csv')
    dump_predictions(result_ensemble_final, SUBMISSION_PATH + 'ensemble_final.' + str(time.time()) + '.csv')

    # Dump weights and stuff
    tta_methods = {metadata['configuration_file']: metadata['best_method'] for metadata in metadatas}
    weights_all = result_ensemble_everything["weights"]
    weights_slices = result_ensemble_slices["weights"]
    weights_final = result_ensemble_final["weights"]
    weights_path = WEIGHTS_PATH + 'weights_ensemble.' + str(time.time()) + '.pkl'

    with open(weights_path, 'w') as f:
        pickle.dump({
                        'tta_methods': tta_methods,
                        'weights_all': weights_all,
                        'weights_slices': weights_slices,
                        'weights_final': weights_final,
                        'outliers': outliers,
                    }, f, pickle.HIGHEST_PROTOCOL)
    print "Weights file dumped to %s" % weights_path



if __name__ == '__main__':
    if len(sys.argv) > 1:
        print "Reusing weights!"
        weights_path = sys.argv[1]
        weights_dict = np.load(weights_path)
        main(weights_dict)
    else:
        main()
