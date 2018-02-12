from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

import numpy as np
import random as rn
import gc
import os
import shutil

NUM_MON_SITES = 100
NUM_MON_INST_TEST = 30
NUM_MON_INST_TRAIN = 60
NUM_MON_INST = NUM_MON_INST_TEST + NUM_MON_INST_TRAIN
NUM_UNMON_SITES_TEST = 5500
NUM_UNMON_SITES_TRAIN = 3500
NUM_UNMON_SITES = NUM_UNMON_SITES_TEST + NUM_UNMON_SITES_TRAIN

seq_length = 4096

data_loc = "/home/primes/datasets/knndata/batch"


def release_list(a_list):
    """Free a_list from memory"""
    del a_list[:]
    del a_list


def process():
    """Read in data, perform randomized split into train/test sets, calculate
    inter-packet timings and metadata, pad/truncate sequences, create one-hot encodings of labels,
    and save all this information to the preprocess folder."""

    train_seq_and_labels = []
    test_seq_and_labels = []

    print("reading data - sens")

    for site in range(0, NUM_MON_SITES):
        all_instances = []

        for sense in range(0, NUM_MON_INST):
            path = data_loc + "/%d-%d" % (site, sense)
            f = open(path)
            cell_time_and_dir = f.read().split()

            cell_time = []
            cell_dir = []

            # Metadata measurements
            total_time = float(cell_time_and_dir[len(cell_time_and_dir) - 2])
            total_outgoing = 0  # 1
            total_incoming = 0  # -1

            for i, value in enumerate(cell_time_and_dir):
                packet_num = i / 2
                if i % 2 == 0:
                    if packet_num < 4100:
                        cell_time.append(float(value))
                else:
                    if packet_num < 4100:
                        cell_dir.append(float(value))

                    if float(value) == 1.:
                        total_outgoing += 1
                    else:
                        total_incoming += 1

            total_packets = total_outgoing + total_incoming
            metadata = [total_time, total_packets / total_time,
                        total_packets, total_outgoing, total_incoming, total_outgoing / total_packets,
                        total_incoming / total_packets]

            all_instances.append([cell_time, cell_dir, metadata])

        # shuffling instances ensures no bias among instances in train and validation/test
        rn.shuffle(all_instances)

        # split instances into train and validation/test sets
        for sense in range(0, NUM_MON_INST):
            if sense < NUM_MON_INST_TRAIN:
                train_seq_and_labels.append([all_instances[sense][0], all_instances[sense][1],
                                             all_instances[sense][2], site])
            else:
                test_seq_and_labels.append([all_instances[sense][0], all_instances[sense][1],
                                            all_instances[sense][2], site])

    print("reading data - insens")

    all_insens = []
    for site in range(0, NUM_UNMON_SITES):
        path = data_loc + "/%d" % site
        f = open(path)
        cell_time_and_dir = f.read().split()

        cell_time = []
        cell_dir = []

        # Metadata measurements
        total_time = float(cell_time_and_dir[len(cell_time_and_dir) - 2])
        total_outgoing = 0  # 1
        total_incoming = 0  # -1

        for i, value in enumerate(cell_time_and_dir):
            packet_num = i / 2
            if i % 2 == 0:
                if packet_num < 4100:
                    cell_time.append(float(value))
            else:
                if packet_num < 4100:
                    cell_dir.append(float(value))

                if float(value) == 1.:
                    total_outgoing += 1
                else:
                    total_incoming += 1

        total_packets = total_outgoing + total_incoming
        metadata = [total_time, total_packets / total_time,
                    total_packets, total_outgoing, total_incoming, total_outgoing / total_packets,
                    total_incoming / total_packets]

        all_insens.append([cell_time, cell_dir, metadata])

    # shuffling instances ensures no bias among instances in train and validation/test
    rn.shuffle(all_insens)

    # split instances into train and validation/test set
    for insens in range(0, NUM_UNMON_SITES):
        if insens < NUM_UNMON_SITES_TRAIN:
            train_seq_and_labels.append([all_insens[insens][0], all_insens[insens][1],
                                         all_insens[insens][2], NUM_MON_SITES])
        else:
            test_seq_and_labels.append([all_insens[insens][0], all_insens[insens][1],
                                        all_insens[insens][2], NUM_MON_SITES])

    print("processing data")

    # currently lists have randomly-arranged instances in order of site. Need to take site dependency out
    rn.shuffle(train_seq_and_labels)
    rn.shuffle(test_seq_and_labels)

    train_time = []
    train_dir = []
    train_metadata = []
    train_labels = []

    test_time = []
    test_dir = []
    test_metadata = []
    test_labels = []

    # extract sequences from randomized train/test sets
    for time_seq, dir_seq, metadata, label in train_seq_and_labels:
        train_time.append(time_seq)
        train_dir.append(dir_seq)
        train_metadata.append(metadata)
        train_labels.append(label)
    for time_seq, dir_seq, metadata, label in test_seq_and_labels:
        test_time.append(time_seq)
        test_dir.append(dir_seq)
        test_metadata.append(metadata)
        test_labels.append(label)

    # free randomized sets from memory to conserve system RAM
    release_list(all_insens)
    release_list(train_seq_and_labels)
    release_list(test_seq_and_labels)
    gc.collect()

    train_metadata = np.array(train_metadata)
    test_metadata = np.array(test_metadata)

    # pad and truncate sequences to desired len
    train_time = pad_sequences(train_time, maxlen=seq_length, dtype='float32', padding='post', truncating='post')
    train_dir = pad_sequences(train_dir, maxlen=seq_length, dtype='float32', padding='post', truncating='post')
    test_time = pad_sequences(test_time, maxlen=seq_length, dtype='float32', padding='post', truncating='post')
    test_dir = pad_sequences(test_dir, maxlen=seq_length, dtype='float32', padding='post', truncating='post')

    # calculate inter-packet timings - time difference between consecutive packets
    train_time_dleft = np.zeros(train_time.shape)  # for current packet, time difference bw/ prev packet and current
    train_time_dright = np.zeros(train_time.shape)  # for current packet, time difference bw/ next packet and current
    for row in range(train_time.shape[0]):
        for col in range(1, train_time.shape[1]):
            train_time_dleft[row][col] = train_time[row][col] - train_time[row][col - 1]
        for col in range(0, train_time.shape[1] - 1):
            train_time_dright[row][col] = train_time[row][col + 1] - train_time[row][col]

    test_time_dleft = np.zeros(test_time.shape)
    test_time_dright = np.zeros(test_time.shape)
    for row in range(test_time.shape[0]):
        for col in range(1, test_time.shape[1]):
            test_time_dleft[row][col] = test_time[row][col] - test_time[row][col - 1]
        for col in range(0, test_time.shape[1] - 1):
            test_time_dright[row][col] = test_time[row][col + 1] - test_time[row][col]

    # stacking makes these sequences easier to save
    train_seq = np.stack((train_time, train_time_dleft, train_time_dright, train_dir), axis=-1)
    test_seq = np.stack((test_time, test_time_dleft, test_time_dright, test_dir), axis=-1)

    # one-hot encoding of labels
    if NUM_UNMON_SITES == 0:  # closed-world
        train_labels = to_categorical(train_labels, num_classes=NUM_MON_SITES)
        test_labels = to_categorical(test_labels, num_classes=NUM_MON_SITES)
    else:
        # add extra class for unmonitored sites
        train_labels = to_categorical(train_labels, num_classes=NUM_MON_SITES + 1)
        test_labels = to_categorical(test_labels, num_classes=NUM_MON_SITES + 1)

    print("training data stats: ")
    print(train_seq.shape)
    print(train_metadata.shape)
    print(train_labels.shape)
    
    print("testing data stats: ")
    print(test_seq.shape)
    print(test_metadata.shape)
    print(test_labels.shape)
    
    print("saving data")
    save_dir = "preprocess"
    shutil.rmtree(save_dir)  # delete save_dir so we don't overlap data
    os.mkdir(save_dir)

    np.save(file=r"%s/train_seq" % save_dir, arr=train_seq)
    np.save(file=r"%s/train_metadata" % save_dir, arr=train_metadata)
    np.save(file=r"%s/train_labels" % save_dir, arr=train_labels)

    np.save(file=r"%s/test_seq" % save_dir, arr=test_seq)
    np.save(file=r"%s/test_metadata" % save_dir, arr=test_metadata)
    np.save(file=r"%s/test_labels" % save_dir, arr=test_labels)


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

    process()


if __name__ == "__main__":
    main(100, 30, 60, 5500, 3500)
