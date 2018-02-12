from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

NUM_MON_SITES = 100
NUM_MON_INST_TEST = 30
NUM_MON_INST_TRAIN = 60
NUM_MON_INST = NUM_MON_INST_TEST + NUM_MON_INST_TRAIN
NUM_UNMON_SITES_TEST = 5500
NUM_UNMON_SITES_TRAIN = 3500
NUM_UNMON_SITES = NUM_UNMON_SITES_TEST + NUM_UNMON_SITES_TRAIN


def find_accuracy(predictions, actual, min_confidence):
    """Compute TPR and FPR based on softmax output predictions, one-hot encodings for the correct classes,
    and the minimum confidence threshold."""

    # calculate class with highest probability
    uncertain_predictions = np.argmax(predictions, axis=1)
    actual = np.argmax(actual, axis=1)

    # adjust predicted classes to reflect min_confidence
    certain_predictions = np.zeros(uncertain_predictions.shape)
    for i in range(0, len(certain_predictions)):
        # if classified as sens with not high-enough probability, re-classify as insens
        predicted_class = uncertain_predictions[i]
        if predicted_class < NUM_MON_SITES and predictions[i][predicted_class] < min_confidence:
            certain_predictions[i] = NUM_MON_SITES
        else:
            certain_predictions[i] = predicted_class

    # compute TPR and FPR
    sens_correct = 0
    insens_as_sens = 0
    for i in range(len(actual)):
        if actual[i] == NUM_MON_SITES:  # insens site
            if certain_predictions[i] < NUM_MON_SITES:  # but predicted as a sens site
                insens_as_sens += 1
        else:  # sens site
            if actual[i] == certain_predictions[i]:  # prediction matches up
                sens_correct += 1

    tpr = sens_correct / (NUM_MON_SITES * NUM_MON_INST_TEST) * 100
    if NUM_UNMON_SITES == 0:  # closed-world
        fpr = 0
    else:
        fpr = insens_as_sens / NUM_UNMON_SITES_TEST * 100

    return "TPR: %f, FPR: %f" % (tpr, fpr)


def main(num_mon_sites, num_mon_inst_test, num_mon_inst_train, num_unmon_sites_test, num_unmon_sites_train):
    global NUM_MON_SITES
    global NUM_MON_INST_TEST
    global NUM_MON_INST_TRAIN
    global NUM_MON_INST
    global NUM_UNMON_SITES_TEST
    global NUM_UNMON_SITES_TRAIN
    global NUM_UNMON_SITES

    NUM_MON_SITES = num_mon_sites
    NUM_MON_INST_TEST = num_mon_inst_test
    NUM_MON_INST_TRAIN = num_mon_inst_train
    NUM_MON_INST = num_mon_inst_test + num_mon_inst_train
    NUM_UNMON_SITES_TEST = num_unmon_sites_test
    NUM_UNMON_SITES_TRAIN = num_unmon_sites_train
    NUM_UNMON_SITES = num_unmon_sites_test + num_unmon_sites_train

    prediction_dir = "/home/primes/attack_scripts/open_world/predictions"
    data_dir = "/home/primes/attack_scripts/open_world/preprocess"

    # read in data from numpy files
    time_predictions = np.load(r"%s/time_model.npy" % prediction_dir)
    dir_predictions = np.load(r"%s/dir_model.npy" % prediction_dir)
    test_labels = np.load(r"%s/test_labels.npy" % data_dir)

    # Var-CNN ensemble predictions are just a simple average of Var-CNN time and direction softmax outputs
    ensemble_predictions = np.add(time_predictions, dir_predictions)
    ensemble_predictions = np.divide(ensemble_predictions, 2)

    if NUM_UNMON_SITES == 0:  # closed-world
        print("min_confidence=", 0.)
        print("time model results:", find_accuracy(time_predictions, test_labels, 0.))
        print("dir model results:", find_accuracy(dir_predictions, test_labels, 0.))
        print("ensemble results:", find_accuracy(ensemble_predictions, test_labels, 0.))
    else:
        for conf in range(0, 11):
            conf *= 0.1
            print("min_confidence=", conf)
            print("time model results:", find_accuracy(time_predictions, test_labels, conf))
            print("dir model results:", find_accuracy(dir_predictions, test_labels, conf))
            print("ensemble results:", find_accuracy(ensemble_predictions, test_labels, conf))


if __name__ == '__main__':
    main(100, 30, 60, 5500, 3500)
