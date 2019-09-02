from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import json
import h5py


def find_accuracy(model_predictions, conf_thresh, actual_labels=None,
                  num_mon_sites=None, num_mon_inst_test=None,
                  num_unmon_sites_test=None, num_unmon_sites=None):
    """Compute TPR and FPR based on softmax output predictions."""

    # Calculates output classes (classes with the highest probability)
    actual_labels = np.argmax(actual_labels, axis=1)

    # Changes predictions according to confidence threshold
    thresh_model_labels = np.zeros(len(model_predictions))
    for inst_num, softmax in enumerate(model_predictions):
        predicted_class = np.argmax(softmax)
        if predicted_class < num_mon_sites and \
                softmax[predicted_class] < conf_thresh:
            thresh_model_labels[inst_num] = num_mon_sites
        else:
            thresh_model_labels[inst_num] = predicted_class

    # Computes TPR and FPR
    two_class_true_pos = 0  # Mon correctly classified as any mon site
    multi_class_true_pos = 0  # Mon correctly classified as specific mon site
    false_pos = 0  # Unmon incorrectly classified as mon site

    for inst_num, inst_label in enumerate(actual_labels):
        if inst_label == num_mon_sites:  # Supposed to be unmon site
            if thresh_model_labels[inst_num] < num_mon_sites:
                false_pos += 1
        else:  # Supposed to be mon site
            if thresh_model_labels[inst_num] < num_mon_sites:
                two_class_true_pos += 1
            if thresh_model_labels[inst_num] == inst_label:
                multi_class_true_pos += 1

    two_class_tpr = two_class_true_pos / \
                    (num_mon_sites * num_mon_inst_test) * 100
    two_class_tpr = '%.2f' % two_class_tpr + '%'
    multi_class_tpr = multi_class_true_pos / \
                      (num_mon_sites * num_mon_inst_test) * 100
    multi_class_tpr = '%.2f' % multi_class_tpr + '%'

    if num_unmon_sites == 0:  # closed-world
        fpr = '0.00%'
    else:
        fpr = false_pos / num_unmon_sites_test * 100
        fpr = '%.2f' % fpr + '%'

    return two_class_tpr, multi_class_tpr, fpr


def log_cw(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    two_class_tpr, multi_class_tpr, fpr = find_accuracy(
        softmax, 0., **parameters)
    print('\t accuracy: %s' % multi_class_tpr)
    results['%s_acc' % sub_model_name] = multi_class_tpr


def log_ow(results, sub_model_name, softmax, **parameters):
    print('%s model:' % sub_model_name)
    for conf_thresh in np.arange(0, 1.01, 0.1):
        two_class_tpr, multi_class_tpr, fpr = find_accuracy(
            softmax, conf_thresh, **parameters)
        print('\t conf: %f' % conf_thresh)
        print('\t \t two-class TPR: %s' % two_class_tpr)
        print('\t \t multi-class TPR: %s' % multi_class_tpr)
        print('\t \t FPR: %s' % fpr)

        prefix = '%s_%f' % (sub_model_name, conf_thresh)
        results['%s_two_TPR' % prefix] = two_class_tpr
        results['%s_multi_TPR' % prefix] = multi_class_tpr
        results['%s_FPR' % prefix] = fpr


def log_setting(setting, predictions, results, **parameters):
    print(setting + '-world results')
    for sub_model_name, softmax in predictions.items():
        if setting == 'closed':
            log_cw(results, sub_model_name, softmax, **parameters)
        elif setting == 'open':
            log_ow(results, sub_model_name, softmax, **parameters)


def main(config):
    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    data_dir = config['data_dir']
    predictions_dir = config['predictions_dir']
    mixture = config['mixture']

    with h5py.File('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                         num_mon_inst, num_unmon_sites_train,
                                         num_unmon_sites_test), 'r') as f:
        test_labels = f['test_data/labels'][:]

    # Aggregates predictions from mixture models
    predictions = {}
    ensemble_softmax = None
    for inner_comb in mixture:
        sub_model_name = '_'.join(inner_comb)
        softmax = np.load('%s%s_model.npy' % (predictions_dir, sub_model_name))
        if ensemble_softmax is None:
            ensemble_softmax = np.zeros_like(softmax)
        predictions[sub_model_name] = softmax

    parameters = {'actual_labels': test_labels,
                  'num_mon_sites': num_mon_sites,
                  'num_mon_inst_test': num_mon_inst_test,
                  'num_unmon_sites_test': num_unmon_sites_test,
                  'num_unmon_sites': num_unmon_sites}

    # Performs simple average to get ensemble predictions
    for softmax in predictions.values():
        ensemble_softmax += softmax
    ensemble_softmax /= len(predictions)
    if len(predictions) > 1:
        predictions['ensemble'] = ensemble_softmax

    results = {}
    if num_unmon_sites == 0:  # Closed-world
        log_setting('closed', predictions, results, **parameters)
    else:  # Open-world
        log_setting('open', predictions, results, **parameters)

    with open('job_result.json', 'w') as f:
        json.dump(results, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    main(config)
