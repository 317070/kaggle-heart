import cPickle as pickle
import csv
import glob
import os
import time
import numpy as np
import theano
import theano.tensor as T
import data_loader
import postprocess
import utils
import paths

METADATAS_LOCATION = 'metadatas.pkl'
RELOAD_METADATAS = False
DO_XVAL = True

PREDICTIONS_PATH = paths.INTERMEDIATE_PREDICTIONS_PATH
SUBMISSION_PATH = paths.SUBMISSION_PATH

PRINT_EVERY = 1000  # Print itermediate results during training
LEARNING_RATE = 10  # Learning rate of gradient descend
NR_EPOCHS = 100
C = 0.0001  # L1 regularisation parameters

SELECT_TOP = 10  # Nr of models to select


def _get_all_prediction_files():
    all_files = glob.glob(PREDICTIONS_PATH + '*')
    res = []
    for f in all_files:
        if f in MODELS_TO_USE:
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


all_patient_ids = _construct_all_patient_ids()


def load_prediction_file(path):
    try:
        return np.load(path)
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
    pats_not_predicted = {set: [] for set in data_loader.patient_folders}
    pats_predicted = {set: [] for set in data_loader.patient_folders}
    for prediction in metadata['predictions']:
        pid = prediction['patient']
        pset = data_loader.id_to_index_map[pid][0]
        if _is_empty_prediction(prediction):
            pats_not_predicted[pset].append(pid)
        else:
            pats_predicted[pset].append(pid)
    return pats_not_predicted, pats_predicted


## AVERAGING
def geomav(x):
    if len(x) == 0:
        return np.zeros(600)
    res = np.cumsum(utils.norm_geometric_average(utils.cdf_to_pdf(x)))
    return res


def normalav(x):
    if len(x) == 0:
        return np.zeros(600)
    return np.mean(x, axis=0)


AVERAGING_METHODS = (normalav, geomav)


def _validate_metadata(metadata, pats_predicted):
    """Compute validation score for given patients.
    """
    errors = []
    for pid in pats_predicted:
        prediction_pid = metadata['predictions'][pid - 1]
        assert prediction_pid['patient'] == pid
        error_sys = utils.CRSP(prediction_pid["systole_average"], data_loader.regular_labels[pid - 1, 1])
        error_dia = utils.CRSP(prediction_pid["diastole_average"], data_loader.regular_labels[pid - 1, 2])
        errors += [error_sys, error_dia]
    return np.mean(errors)


def _compute_tta(metadata, averaging_method):
    for prediction_pid in metadata['predictions']:
        pid = prediction_pid['patient']
        prediction_pid["systole_average"] = averaging_method(prediction_pid["systole"])
        prediction_pid["diastole_average"] = averaging_method(prediction_pid["diastole"])


def _compute_nr_ttas(metadata, pats_predicted):
    return len(metadata['predictions'][pats_predicted[0]]["systole"])


def _make_valid_distributions(metadata):
    for prediction in metadata['predictions']:
        prediction["systole_average"] = postprocess.make_monotone_distribution(prediction["systole_average"])
        postprocess.test_if_valid_distribution(prediction["systole_average"])
        prediction["diastole_average"] = postprocess.make_monotone_distribution(prediction["diastole_average"])
        postprocess.test_if_valid_distribution(prediction["diastole_average"])


def _add_tta_score_to_metadata(metadata, tta_score):
    metadata['tta_score'] = tta_score


def _compute_best_tta(metadata, pats_predicted):
    print "    Using %d patients, averaging over %d TTAs" % (
        len(pats_predicted), _compute_nr_ttas(metadata, pats_predicted))
    best_score = 1
    best_method = None
    for averaging_method in AVERAGING_METHODS:
        _compute_tta(metadata, averaging_method)
        err = _validate_metadata(metadata, pats_predicted)
        print "    - %s: %.5f" % (averaging_method.func_name, err)
        if err < best_score:
            best_score, best_method = err, averaging_method
    _compute_tta(metadata, best_method)
    _make_valid_distributions(metadata)
    tta_score = _validate_metadata(metadata, pats_predicted)
    print "    Choosing %s (%.5f)" % (best_method.func_name, tta_score)
    _add_tta_score_to_metadata(metadata, tta_score)


def _remove_ttas(metadata, tags_to_remove=('systole', 'diastole')):
    for prediction in metadata['predictions']:
        # Tag the prediction to keep track of whether it was empty or not
        prediction['is_empty'] = _is_empty_prediction(prediction)
        for tag in tags_to_remove:
            if tag in prediction:
                del prediction[tag]


def _process_prediction_file(path):
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
    if nr_val_predicted == 0 or nr_test_predicted == 0:
        print "  Skipping this model, not enough predictions."
        return

    # Compute best way of averaging
    print "  Trying out different ways of doing TTA:"
    _compute_best_tta(metadata, pats_predicted['validation'])

    # Clean up the metadata file and return it
    _remove_ttas(metadata)
    return metadata


def _load_and_process_metadata_files():
    all_prediction_files = sorted(_get_all_prediction_files())[:]
    nr_prediction_files = len(all_prediction_files)

    print "Using the following files:"
    print
    print "\n".join(map(os.path.basename, all_prediction_files))

    useful_files = []

    for idx, path in enumerate(all_prediction_files):
        print
        print 'Processing %s (%d/%d)' % (os.path.basename(path), idx + 1, nr_prediction_files)
        m = _process_prediction_file(path)
        if m:
            useful_files.append(m)

    print
    print "Loaded %d files" % len(useful_files)
    return useful_files


## ENSEMBLING


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
            assert prediction["patient"] == j + 1

    return res_mask, res_sys, res_dia


def _create_label_matrix():
    systole_valid_labels = np.array(
        [utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:, 1].flatten()])
    diastole_valid_labels = np.array(
        [utils.cumulative_one_hot(v) for v in data_loader.regular_labels[:, 2].flatten()])
    return systole_valid_labels, diastole_valid_labels


def _get_train_val_test_ids():
    sets = [data_loader.id_to_index_map[pid][0] for pid in range(1, data_loader.NUM_PATIENTS + 1)]

    return np.array(sets) == "train", np.array(sets) == "validation", np.array(sets) == "test"


def _find_weights(w_init, preds_matrix, targets_matrix, mask_matrix):
    nr_models = len(w_init)
    nr_patients = mask_matrix.shape[1]

    learning_rate = LEARNING_RATE
    nr_epochs = NR_EPOCHS

    print "      Compiling function"

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
    loss = ((av_preds - targets) ** 2).mean() + C * l1_loss

    # Update function (keeping the weights normalised)
    grad_weights = theano.grad(loss, weights)
    updated_weights = T.clip(weights - learning_rate * grad_weights, 0, 100)  # Dont allow weights smaller than 0
    updated_weights_normalised = updated_weights / updated_weights.sum()  # Renormalise
    updates = {weights: updated_weights_normalised}
    iter_train = theano.function([], loss, updates=updates)

    # Do training
    print "      Training"
    for iteration in xrange(nr_epochs):
        train_err = iter_train()
        w_value = weights.eval()
        if (iteration + 1) % PRINT_EVERY == 0:
            print iteration
            print "      train_error: %.4f, weights: %s" % (train_err, str((w_value * 1000).astype('int32')))

    return np.array(weights.eval())  # Convert cudaNArray ot numpy if necessairy


def _compute_predictions_ensemble(weights, preds, mask):
    masked_weights = mask * weights[:, np.newaxis]
    tot_masked_weights = masked_weights.sum(axis=0)
    preds_weighted_masked = preds * masked_weights[:, :, np.newaxis]
    av_preds = preds_weighted_masked.sum(axis=0) / tot_masked_weights[:, np.newaxis]
    return av_preds


def _eval_weights(weights, preds, targets, mask):
    av_preds = _compute_predictions_ensemble(weights, preds, mask)

    crps = ((av_preds - targets) ** 2).mean()
    return crps


def _create_ensembles(metadatas):
    # aggregate predictions and targets
    mask, preds_sys, preds_dia = _create_prediction_matrix(metadatas)
    targets_sys, targets_dia = _create_label_matrix()
    print mask.mean()

    # initialise weights
    nr_models = len(metadatas)
    w_init = np.ones((nr_models,), dtype='float32') / nr_models

    # split data
    _, val_idx, test_idx = _get_train_val_test_ids()
    mask_val, mask_test = mask[:, val_idx], mask[:, test_idx]
    targets_sys_val = targets_sys[val_idx[:len(targets_sys)], :]
    targets_dia_val = targets_dia[val_idx[:len(targets_dia)], :]
    preds_sys_val, preds_sys_test = preds_sys[:, val_idx, :], preds_sys[:, test_idx, :]
    preds_dia_val, preds_dia_test = preds_dia[:, val_idx, :], preds_dia[:, test_idx, :]

    ## CREATE SYSTOLE AND DIASTOLE PREDICTION USING LOOXVAL
    if DO_XVAL:
        print "Making systole ensemble"
        print "  Doing leave one patient out xval"
        nr_val_patients = len(targets_sys_val)
        train_errs_sys = []
        val_errs_sys = []
        w_iters_sys = []
        for val_pid in xrange(nr_val_patients):
            print "    - fold %d" % val_pid
            w_iter = _find_weights(
                w_init,
                np.hstack((preds_sys_val[:, :val_pid], preds_sys_val[:, val_pid + 1:])),
                np.vstack((targets_sys_val[:val_pid], targets_sys_val[val_pid + 1:])),
                np.hstack((mask_val[:, :val_pid], mask_val[:, val_pid + 1:])),
            )
            train_err = _eval_weights(
                w_iter,
                np.hstack((preds_sys_val[:, :val_pid], preds_sys_val[:, val_pid + 1:])),
                np.vstack((targets_sys_val[:val_pid], targets_sys_val[val_pid + 1:])),
                np.hstack((mask_val[:, :val_pid], mask_val[:, val_pid + 1:])),
            )
            val_err = _eval_weights(w_iter,
                                    preds_sys_val[:, val_pid:val_pid + 1],
                                    targets_sys_val[val_pid:val_pid + 1],
                                    mask_val[:, val_pid:val_pid + 1],
                                    )
            print "      train_err: %.4f,  val_err: %.4f" % (train_err, val_err)
            train_errs_sys.append(train_err)
            val_errs_sys.append(val_err)
            w_iters_sys.append(w_iter)

        expected_systole_loss = np.mean(val_errs_sys)
        print "  average train err: %.4f" % np.mean(train_errs_sys)
        print "  average valid err: %.4f" % np.mean(val_errs_sys)

        print "Making diastole ensemble"
        print "  Doing leave one patient out xval"
        nr_val_patients = len(targets_dia_val)
        train_errs_dia = []
        val_errs_dia = []
        w_iters_dia = []
        for val_pid in xrange(nr_val_patients):
            print "    - fold %d" % val_pid
            w_iter = _find_weights(
                w_init,
                np.hstack((preds_dia_val[:, :val_pid], preds_dia_val[:, val_pid + 1:])),
                np.vstack((targets_dia_val[:val_pid], targets_dia_val[val_pid + 1:])),
                np.hstack((mask_val[:, :val_pid], mask_val[:, val_pid + 1:])),
            )
            train_err = _eval_weights(
                w_iter,
                np.hstack((preds_dia_val[:, :val_pid], preds_dia_val[:, val_pid + 1:])),
                np.vstack((targets_dia_val[:val_pid], targets_dia_val[val_pid + 1:])),
                np.hstack((mask_val[:, :val_pid], mask_val[:, val_pid + 1:])),
            )
            val_err = _eval_weights(w_iter,
                                    preds_dia_val[:, val_pid:val_pid + 1],
                                    targets_dia_val[val_pid:val_pid + 1],
                                    mask_val[:, val_pid:val_pid + 1],
                                    )
            print "      train_err: %.4f,  val_err: %.4f" % (train_err, val_err)
            train_errs_dia.append(train_err)
            val_errs_dia.append(val_err)
            w_iters_dia.append(w_iter)

        expected_diastole_loss = np.mean(val_errs_dia)
        print "  average train err: %.4f" % np.mean(train_errs_dia)
        print "  average valid err: %.4f" % np.mean(val_errs_dia)

        ## MAKE ENSEMBLE USING THE FULL VALIDATION SET
        print "Fitting weights on the entire validation set"
        w_sys = _find_weights(w_init, preds_sys_val, targets_sys_val, mask_val)
        w_dia = _find_weights(w_init, preds_dia_val, targets_dia_val, mask_val)

        # Print the result
        print
        print "dia   sys "
        sort_key = lambda x: - x[1] - x[2]
        for metadata, weight_sys, weight_dia in sorted(zip(metadatas, w_sys, w_dia), key=sort_key):
            print "%4.1f  %4.1f : %s" % (weight_sys * 100, weight_dia * 100, metadata["configuration_file"])

    ## SELECT THE TOP MODELS AND RETRAIN ONCE MORE
    print
    print "Selecting the top %d models and retraining" % SELECT_TOP
    sorted_models, sorted_w_sys, sorted_w_dia = zip(*sorted(zip(metadatas, w_sys, w_dia), key=sort_key))
    top_models = sorted_models[:SELECT_TOP]

    w_init_sys_top = np.array(sorted_w_sys[:SELECT_TOP]) / np.array(sorted_w_sys[:SELECT_TOP]).sum()
    w_init_dia_top = np.array(sorted_w_dia[:SELECT_TOP]) / np.array(sorted_w_dia[:SELECT_TOP]).sum()
    mask_top, preds_sys_top, preds_dia_top = _create_prediction_matrix(top_models)

    mask_top_val, mask_top_test = mask_top[:, val_idx], mask_top[:, test_idx]
    preds_sys_top_val, preds_sys_top_test = preds_sys_top[:, val_idx, :], preds_sys_top[:, test_idx, :]
    preds_dia_top_val, preds_dia_top_test = preds_dia_top[:, val_idx, :], preds_dia_top[:, test_idx, :]

    w_sys_top = _find_weights(w_init_sys_top, preds_sys_top_val, targets_sys_val, mask_top_val)
    w_dia_top = _find_weights(w_init_dia_top, preds_dia_top_val, targets_dia_val, mask_top_val)

    sys_err = _eval_weights(w_sys_top, preds_sys_top_val, targets_sys_val, mask_top_val)
    dia_err = _eval_weights(w_dia_top, preds_dia_top_val, targets_dia_val, mask_top_val)

    print
    print "dia   sys "
    sort_key = lambda x: - x[1] - x[2]
    for metadata, weight_sys, weight_dia in zip(top_models, w_sys_top, w_dia_top):
        print "%4.1f  %4.1f : %s" % (weight_sys * 100, weight_dia * 100, metadata["configuration_file"])

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
    test_ids = np.where(test_idx)[0] + 1
    preds_sys = _compute_predictions_ensemble(w_sys_top, preds_sys_top_test, mask_top_test)
    preds_dia = _compute_predictions_ensemble(w_dia_top, preds_dia_top_test, mask_top_test)

    submission_path = SUBMISSION_PATH + "final_submission-%s.csv" % time.time()
    print "dumping submission file to %s" % submission_path
    with open(submission_path, 'w') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvwriter.writerow(['Id'] + ['P%d' % i for i in xrange(600)])
        for pid, pat_pred_sys, pat_pred_dia in zip(test_ids, preds_sys, preds_dia):
            pat_pred_sys = postprocess.make_monotone_distribution(pat_pred_sys)
            pat_pred_dia = postprocess.make_monotone_distribution(pat_pred_dia)
            csvwriter.writerow(["%d_Diastole" % pid] + ["%.18f" % p for p in pat_pred_dia.flatten()])
            csvwriter.writerow(["%d_Systole" % pid] + ["%.18f" % p for p in pat_pred_sys.flatten()])
    print "submission file dumped"


def dump_metadatas(metadatas):
    with open(METADATAS_LOCATION, 'w') as f:
        pickle.dump(metadatas, f, pickle.HIGHEST_PROTOCOL)
    print "metadatas file dumped"


def load_metadatas():
    metadatas = np.load(METADATAS_LOCATION)
    print "Loaded metadatas file"
    return metadatas


def main():
    if RELOAD_METADATAS:
        metadatas = load_metadatas()
    else:
        metadatas = _load_and_process_metadata_files()
        dump_metadatas(metadatas)
    _create_ensembles(metadatas)


if __name__ == '__main__':
    metamodels = sorted([]
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi10_maxout_seqshift_96.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi10_big_leaky_after_seqshift.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi10_zoom_mask_leaky_after.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi10_maxout.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi_zoom_big.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi_zoom_mask_leaky_after.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi_zoom_mask_leaky.pkl")
                        + glob.glob(PREDICTIONS_PATH + "ira_configurations.meta_gauss_roi_zoom.pkl")
                        # + glob.glob(PREDICTIONS_PATH + "je_os_fixedaggr_rellocframe.pkl")
                        # + glob.glob(PREDICTIONS_PATH + "je_meta_fixedaggr_jsc80leakyconv.pkl")
                        # + glob.glob(PREDICTIONS_PATH + "je_meta_fixedaggr_framemax_reg.pkl")
                        # + glob.glob(PREDICTIONS_PATH + "je_os_fixedaggr_relloc_filtered.pkl")
                        # + glob.glob(PREDICTIONS_PATH + "je_os_fixedaggr_relloc_filtered_discs.pkl")
                        )
    slice_models = sorted([]
                          + glob.glob(PREDICTIONS_PATH + "j6_2ch_128mm_skew.pkl")
                          + glob.glob(PREDICTIONS_PATH + "je_ss_jonisc64small_360.pkl")
                          + glob.glob(PREDICTIONS_PATH + "j6_2ch_96mm.pkl")
                          + glob.glob(PREDICTIONS_PATH + "j6_2ch_128mm_96.pkl")
                          + glob.glob(PREDICTIONS_PATH + "j6_4ch_32mm_specialist.pkl")
                          + glob.glob(PREDICTIONS_PATH + "ira_configurations.ch2_zoom_leaky_after_maxout.pkl")
                          + glob.glob(PREDICTIONS_PATH + "ira_configurations.gauss_roi_zoom_mask_leaky.pkl")
                          + glob.glob(PREDICTIONS_PATH + "ira_configurations.gauss_roi_zoom_mask_leaky_after.pkl")
                          + glob.glob(PREDICTIONS_PATH + "ira_configurations.gauss_roi10_big_leaky_after_seqshift.pkl")
                          + glob.glob(PREDICTIONS_PATH + "ira_configurations.gauss_roi10_maxout.pkl")
                          )

    # meta
    MODELS_TO_USE = metamodels
    main()
    # ss
    MODELS_TO_USE = slice_models
    main()
    # mixed_models
    MODELS_TO_USE = metamodels + slice_models
    main()
