from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import random
import h5py
import json
import os
from tqdm import tqdm
import wang_to_varcnn
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler


def main(config):
    """Preprocesses data from all_{}_world.npz and creates .h5 data files.

    Reads in data, performs randomized split into training/test sets,
    calculates inter-packet timings and metadata, pads/truncates sequences,
    creates one-hot encodings of labels, and saves all this information to the
    preprocess folder.
    """

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    inter_time = config['inter_time']
    scale_metadata = config['scale_metadata']
    data_dir = config['data_dir']
    mon_data_loc = data_dir + 'all_closed_world.npz'
    unmon_data_loc = data_dir + 'all_open_world.npz'
    if not os.path.exists(mon_data_loc) or not os.path.exists(unmon_data_loc):
        wang_to_varcnn.main(config)

    print('Starting %d_%d_%d_%d.h5' % (num_mon_sites, num_mon_inst,
                                       num_unmon_sites_train,
                                       num_unmon_sites_test))
    start = time.time()

    train_seq_and_labels = []
    test_seq_and_labels = []

    print('reading monitored data')
    mon_dataset = np.load(mon_data_loc)
    mon_dir_seq = mon_dataset['dir_seq']
    mon_time_seq = mon_dataset['time_seq']
    mon_metadata = mon_dataset['metadata']
    mon_labels = mon_dataset['labels']

    mon_site_data = {}
    mon_site_labels = {}
    print('getting enough monitored websites')
    for dir_seq, time_seq, metadata, site_name \
            in tqdm(zip(mon_dir_seq, mon_time_seq, mon_metadata, mon_labels)):
        if site_name not in mon_site_data:
            if len(mon_site_data) >= num_mon_sites:
                continue
            else:
                mon_site_data[site_name] = []
                mon_site_labels[site_name] = len(mon_site_labels)

        mon_site_data[site_name].append(
            [dir_seq, time_seq, metadata, mon_site_labels[site_name]])

    print('randomly choosing instances for training and test sets')
    assert len(mon_site_data) == num_mon_sites
    for instances in tqdm(mon_site_data.values()):
        random.shuffle(instances)
        assert len(instances) >= num_mon_inst
        for inst_num, all_data in enumerate(instances):
            if inst_num < num_mon_inst_train:
                train_seq_and_labels.append(all_data)
            elif inst_num < num_mon_inst:
                test_seq_and_labels.append(all_data)
            else:
                break

    del mon_dataset, mon_dir_seq, mon_time_seq, mon_metadata, \
        mon_labels, mon_site_data, mon_site_labels

    print('reading unmonitored data')

    unmon_dataset = np.load(unmon_data_loc)
    unmon_dir_seq = unmon_dataset['dir_seq']
    unmon_time_seq = unmon_dataset['time_seq']
    unmon_metadata = unmon_dataset['metadata']

    unmon_site_data = [[dir_seq, time_seq, metadata, num_mon_sites] for
                       dir_seq, time_seq, metadata in
                       zip(unmon_dir_seq, unmon_time_seq, unmon_metadata)]

    print('randomly choosing unmonitored instances for training and test sets')
    random.shuffle(unmon_site_data)
    assert len(unmon_site_data) >= num_unmon_sites
    for inst_num, all_data in tqdm(enumerate(unmon_site_data)):
        if inst_num < num_unmon_sites_train:
            train_seq_and_labels.append(all_data)
        elif inst_num < num_unmon_sites:
            test_seq_and_labels.append(all_data)
        else:
            break

    del unmon_dataset, unmon_dir_seq, unmon_time_seq, \
        unmon_metadata, unmon_site_data

    print('processing data')

    # Removes mon site ordering
    random.shuffle(train_seq_and_labels)
    random.shuffle(test_seq_and_labels)

    train_dir = []
    train_time = []
    train_metadata = []
    train_labels = []

    test_dir = []
    test_time = []
    test_metadata = []
    test_labels = []

    for dir_seq, time_seq, metadata, label in train_seq_and_labels:
        train_dir.append(dir_seq)
        train_time.append(time_seq)
        train_metadata.append(metadata)
        train_labels.append(label)
    for dir_seq, time_seq, metadata, label in test_seq_and_labels:
        test_dir.append(dir_seq)
        test_time.append(time_seq)
        test_metadata.append(metadata)
        test_labels.append(label)

    del train_seq_and_labels, test_seq_and_labels

    train_dir = np.array(train_dir)
    train_time = np.array(train_time)
    train_metadata = np.array(train_metadata)

    test_dir = np.array(test_dir)
    test_time = np.array(test_time)
    test_metadata = np.array(test_metadata)

    # Converts from absolute times to inter-packet times.
    # Each spot holds time diff between curr packet and prev packet
    if inter_time:
        inter_time_train = np.zeros_like(train_time)
        inter_time_train[:, 1:] = train_time[:, 1:] - train_time[:, :-1]
        train_time = inter_time_train

        inter_time_test = np.zeros_like(test_time)
        inter_time_test[:, 1:] = test_time[:, 1:] - test_time[:, :-1]
        test_time = inter_time_test

    # Reshape to add 3rd dim for CNN input
    train_dir = np.reshape(train_dir,
                           (train_dir.shape[0], train_dir.shape[1], 1))
    test_dir = np.reshape(test_dir, (test_dir.shape[0], test_dir.shape[1], 1))

    train_time = np.reshape(train_time,
                            (train_time.shape[0], train_time.shape[1], 1))
    test_time = np.reshape(test_time,
                           (test_time.shape[0], test_time.shape[1], 1))

    if scale_metadata:
        metadata_scaler = StandardScaler()
        train_metadata = metadata_scaler.fit_transform(train_metadata)
        test_metadata = metadata_scaler.transform(test_metadata)

    # One-hot encoding of labels, using one more class for
    # unmonitored sites if in open-world
    num_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    train_labels = to_categorical(train_labels, num_classes=num_classes)
    test_labels = to_categorical(test_labels, num_classes=num_classes)

    print('training data stats:')
    print(train_dir.shape)
    print(train_time.shape)
    print(train_metadata.shape)
    print(train_labels.shape)

    print('testing data stats:')
    print(test_dir.shape)
    print(test_time.shape)
    print(test_metadata.shape)
    print(test_labels.shape)

    print('saving data')
    with h5py.File('%s%d_%d_%d_%d.h5' %
                   (data_dir, num_mon_sites, num_mon_inst,
                    num_unmon_sites_train, num_unmon_sites_test), 'w') as f:
        f.create_group('training_data')
        f.create_group('validation_data')
        f.create_group('test_data')
        for ds_name, arr in [['dir_seq', train_dir],
                             ['time_seq', train_time],
                             ['metadata', train_metadata],
                             ['labels', train_labels]]:
            f.create_dataset('training_data/' + ds_name,
                             data=arr[:int(0.95 * len(arr))])
        for ds_name, arr in [['dir_seq', train_dir],
                             ['time_seq', train_time],
                             ['metadata', train_metadata],
                             ['labels', train_labels]]:
            f.create_dataset('validation_data/' + ds_name,
                             data=arr[int(0.95 * len(arr)):])
        for ds_name, arr in [['dir_seq', test_dir],
                             ['time_seq', test_time],
                             ['metadata', test_metadata],
                             ['labels', test_labels]]:
            f.create_dataset('test_data/' + ds_name,
                             data=arr)

    end = time.time()
    print('Finished %d_%d_%d_%d.h5 in %f seconds' %
          (num_mon_sites, num_mon_inst, num_unmon_sites_train,
           num_unmon_sites_test, end - start))


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    main(config)
