import os
import glob
import theano
import theano.tensor as T
import numpy as np
import scipy
from sklearn.cross_validation import StratifiedKFold
import cPickle
import utils


def cv_save_optimal_weights(predictions_paths, target_path, n_cvfolds=10):
    """
    On each fold of cross-validation it finds optimal weights
    for all the models and saves these weights for further experiments.

    """

    s = np.load("validation_split_v1.pkl")
    targets = data.labels_train[s['indices_valid']]

    print "Loading validation set predictions"
    predictions_list = [np.load(path) for path in predictions_paths]
    predictions_stack = np.array(predictions_list).astype(theano.config.floatX)  # num_sources x num_datapoints x 121
    n_models = predictions_stack.shape[0]
    print 'number of models:', n_models
    print

    print "Compute individual prediction errors"
    individual_prediction_errors = [utils.log_loss(p, targets) for p in predictions_list]
    for i in xrange(n_models):
        print individual_prediction_errors[i], predictions_paths[i]
    print
    del predictions_list

    print "Compiling Theano functions"
    X = theano.shared(predictions_stack)  # source predictions
    t = theano.shared(utils.one_hot(targets))  # targets
    idx = theano.shared(np.zeros(predictions_stack.shape[1], dtype='int64'))

    W = T.vector('W')

    s = T.nnet.softmax(W).reshape((W.shape[0], 1, 1))

    weighted_avg_predictions = T.sum(X[:, idx, :] * s, axis=0)

    error = nn_plankton.log_loss(weighted_avg_predictions, t[idx])
    grad = T.grad(error, W)

    f = theano.function([W], error)
    g = theano.function([W], grad)

    skf = StratifiedKFold(targets, n_folds=n_cvfolds)
    weights_kfolds = []
    idx_valid_kfolds = []

    for train_idx, valid_idx in skf:
        idx.set_value(train_idx)

        w_init = np.zeros(n_models).astype(theano.config.floatX)
        out, _, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

        # softmax weights all models
        out_s = utils.softmax(out)
        weights_kfolds.append(out_s)

        # save valid idx for later
        idx_valid_kfolds.append(valid_idx)

        z = zip(out_s, predictions_paths)
        zs = sorted(z, key=lambda x: x[0], reverse=True)
        for i, j in zs[:10]:
            print '%0.5f\t%s' % (i, os.path.basename(j))
            # print '\'%s\',' % predictions_paths[k]
        print '----------------------------------------------------------------------------------'

    with open(target_path, 'w') as file:
        cPickle.dump({
            'weights_kfolds': weights_kfolds,
            'predictions_paths': predictions_paths,
            'idx_valid_kfolds': idx_valid_kfolds
        }, file, cPickle.HIGHEST_PROTOCOL)

    print ' weights stored in %s' % target_path


def cv_uniform_blend_top(metadata_path, target_path, max_ntop_models=26):
    """
    On each fold select top N models based on the optimal (for that fold) weights.
    Blend N models uniformly and calculate validation error (validation indices should be provided)
    """
    with open(metadata_path) as f:
        d = cPickle.load(f)
    predictions_paths = d['predictions_paths']
    idx_valid_kfolds = d['idx_valid_kfolds']
    weights_kfolds = d['weights_kfolds']

    s = np.load("validation_split_v1.pkl")
    targets = data.labels_train[s['indices_valid']]

    print "Loading validation set predictions"

    predictions_list = [np.load(path) for path in predictions_paths]
    predictions_stack = np.array(predictions_list).astype(theano.config.floatX)  # num_sources x num_datapoints x 121
    del predictions_list
    n_models = predictions_stack.shape[0]
    print 'number of models:', n_models
    print

    print 'Uniform blending'
    avg_losses_valid = []
    ntop_range = range(1, min(n_models + 1, max_ntop_models))
    for n_top_models in ntop_range:
        valid_losses_kfolds = []
        for idx_valid, weights in zip(idx_valid_kfolds, weights_kfolds):
            # find cut-off for top n models
            sorted_weights = np.sort(weights)[::-1]
            cut_off = sorted_weights[n_top_models] if n_top_models < n_models else 0

            # uniform top n weights
            out_uniform = np.copy(weights)
            out_uniform[np.where(out_uniform <= cut_off)[0]] = -np.inf
            out_uniform[np.where(out_uniform > cut_off)[0]] = 1.0
            out_uniform_s = utils.softmax(out_uniform)
            weighted_prediction = np.sum(predictions_stack[:, idx_valid, :] * out_uniform_s.reshape((n_models, 1, 1)),
                                         axis=0)
            valid_losses_kfolds.append(utils.log_loss(weighted_prediction, targets[idx_valid]))

        avg_loss_valid = np.mean(valid_losses_kfolds)
        avg_losses_valid.append(avg_loss_valid)
        print n_top_models, avg_loss_valid

    with open(target_path, 'w') as file:
        cPickle.dump({
            'ntop_models': ntop_range,
            'losses_valid': avg_losses_valid,
        }, file, cPickle.HIGHEST_PROTOCOL)
    print ' losses stored in %s', target_path


def cv_weighted_blend_top(metadata_path, target_path, max_ntop_models=26):
    """
    On each fold select top N models based on the optimal (for that fold) weights.
    Optimize weights again only for N models and calculate validation error
    (validation indices should be provided)

    """

    with open(metadata_path) as f:
        d = cPickle.load(f)
    predictions_paths = d['predictions_paths']
    idx_valid_kfolds = d['idx_valid_kfolds']
    weights_kfolds = d['weights_kfolds']

    s = np.load("validation_split_v1.pkl")
    targets = data.labels_train[s['indices_valid']]
    n_models = len(predictions_paths)

    ntop_range = range(1, min(n_models + 1, max_ntop_models))
    print 'Weighted blending'
    avg_losses_valid = []
    for n_top_models in ntop_range:
        valid_losses_kfolds = []
        for idx_valid, weights in zip(idx_valid_kfolds, weights_kfolds):
            # find cut-off for top n models
            sorted_weights = np.sort(weights)[::-1]
            cut_off = sorted_weights[n_top_models] if n_top_models < n_models else 0
            top_idx = np.array(np.where(weights > cut_off)[0])

            # load predictions from top N models
            predictions_list = [np.load(predictions_paths[i]) for i in top_idx]
            predictions_stack = np.array(predictions_list).astype(
                theano.config.floatX)  # num_sources x num_datapoints x 121

            # find train indices
            idx_all = range(predictions_stack.shape[1])
            idx_train = np.setdiff1d(idx_all, idx_valid)

            # compile theano again and again .... todo
            X = theano.shared(predictions_stack)
            t = theano.shared(utils.one_hot(targets))
            idx = theano.shared(idx_train)
            W = T.vector('W')
            s = T.nnet.softmax(W).reshape((W.shape[0], 1, 1))
            weighted_avg_predictions = T.sum(X[:, idx, :] * s, axis=0)
            error = nn_plankton.log_loss(weighted_avg_predictions, t[idx])
            grad = T.grad(error, W)
            f = theano.function([W], error)
            g = theano.function([W], grad)

            # optimize weights of top n models
            w_init = (np.zeros(n_top_models)).astype(theano.config.floatX)
            out, _, _ = scipy.optimize.fmin_l_bfgs_b(f, w_init, fprime=g, pgtol=1e-09, epsilon=1e-08, maxfun=10000)

            # valid loss
            idx.set_value(idx_valid)
            valid_loss = f(out)
            valid_losses_kfolds.append(valid_loss)

        avg_loss_valid = np.mean(valid_losses_kfolds)
        avg_losses_valid.append(avg_loss_valid)
        print n_top_models, avg_loss_valid

    with open(target_path, 'w') as file:
        cPickle.dump({
            'ntop_models': ntop_range,
            'losses_valid': avg_losses_valid,
        }, file, cPickle.HIGHEST_PROTOCOL)
    print ' losses stored in %s' % target_path


def cv_count_top_models(metadata_path, max_ntop_models=26):
    """
    For each model count the number of times it appeared
    in top N across all 10 folds
    """
    with open(metadata_path) as f:
        d = cPickle.load(f)
    predictions_paths = d['predictions_paths']
    weights_kfolds = d['weights_kfolds']

    print "Loading validation set predictions"
    predictions_list = [np.load(path) for path in predictions_paths]
    predictions_stack = np.array(predictions_list).astype(theano.config.floatX)  # num_sources x num_datapoints x 121
    del predictions_list
    n_models = predictions_stack.shape[0]
    print 'number of models', n_models

    for n_top_models in xrange(1, min(n_models, max_ntop_models)):
        print 'top N = ', n_top_models
        counter_kfolds = np.zeros(n_models)
        avg_weights_kfolds = np.zeros(n_models)
        for weights in weights_kfolds:
            # find cut-off for top n models
            sorted_weights = np.sort(weights)[::-1]
            cut_off = sorted_weights[n_top_models]
            top_idx = np.array(np.where(weights > cut_off)[0])
            counter_kfolds[top_idx] += 1
            avg_weights_kfolds[top_idx] += weights[top_idx]

        nonzero_counts_idx = np.where(counter_kfolds > 0)[0]
        z = zip(avg_weights_kfolds[nonzero_counts_idx], counter_kfolds[nonzero_counts_idx],
                np.arange(n_models)[nonzero_counts_idx])
        zs = sorted(z, key=lambda x: x[0], reverse=True)
        for i, j, k in zs:
            print '%0.5f\t%2.0f\t%s' % (i / j, j, predictions_paths[k])
            # print '\'%s\',' % predictions_paths[k]
        print '----------------------------------------------------------------------------------'


def select_models_with_test_predictions(path):
    test_predictions_paths = glob.glob(path + '/test--*.npy')
    valid_predictions_paths = []
    print 'Validation predictions do not exist for these files: '
    for p in test_predictions_paths:
        valid_p = p.replace('test--', 'valid--', 1)
        if os.path.isfile(valid_p):
            valid_predictions_paths.append(valid_p)
        else:
            print p
    print
    return valid_predictions_paths


def filter_paths(predictions_paths, filters):
    filtered_prediction_paths = []
    for p in predictions_paths:
        b = [f in p for f in filters]
        if not any(b):
            filtered_prediction_paths.append(p)
    return filtered_prediction_paths


if __name__ == '__main__':
    metadata_path = utils.get_dir_path('ensembles')

    predictions_valid_pattern = "../plankton_predictions/*/valid--*.npy"
    predictions_paths = glob.glob(predictions_valid_pattern)

    cv_save_optimal_weights(predictions_paths, metadata_path)
    cv_uniform_blend_top(metadata_path, metadata_basename + '_uniform.pkl')
    cv_weighted_blend_top(metadata_path, metadata_basename + '_weighted.pkl')
    cv_count_top_models(metadata_path)
    create_final_weights(predictions_paths, metadata_path.replace('cv', ''))
