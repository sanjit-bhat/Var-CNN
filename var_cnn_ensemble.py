from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.callbacks import LearningRateScheduler
from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, Flatten, Dropout, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras import Input
from sklearn.preprocessing import StandardScaler

import numpy as np
import time

NUM_MON_SITES = 100
NUM_MON_INST_TEST = 30
NUM_MON_INST_TRAIN = 60
NUM_MON_INST = NUM_MON_INST_TEST + NUM_MON_INST_TRAIN
NUM_UNMON_SITES_TEST = 5500
NUM_UNMON_SITES_TRAIN = 3500
NUM_UNMON_SITES = NUM_UNMON_SITES_TEST + NUM_UNMON_SITES_TRAIN

seq_length = 4096


def time_conv_layer(model, nb_filters):
    """Conv-ReLU-Batch Normalization layer for timing. This does not use dilated convolutions."""

    model = Conv1D(filters=nb_filters, kernel_size=3, padding='causal', activation='relu')(model)
    model = BatchNormalization()(model)
    return model


def time_conv_block(model, nb_layers, nb_filters):
    """Creates multiple conv layers followed by max pooling and dropout layers."""

    for _ in range(nb_layers):
        model = time_conv_layer(model, nb_filters)
    model = MaxPooling1D()(model)
    model = Dropout(0.1)(model)
    return model


def dir_conv_layer(model, nb_filters, rate):
    """Conv-ReLU-Batch Normalization layer for direction. This uses dilated convolutions."""

    model = Conv1D(filters=nb_filters, kernel_size=3, padding='causal', dilation_rate=rate, activation='relu')(model)
    model = BatchNormalization()(model)

    # exponentially increase dilated convolution receptive field
    # receptive field size loops back around when rate = 16 to create [1...8] block
    rate *= 2
    if rate == 16:
        rate = 1
    return model, rate


def dir_conv_block(model, nb_layers, nb_filters, rate):
    """Creates multiple conv layers followed by max pooling and dropout layers."""

    for _ in range(nb_layers):
        model, rate = dir_conv_layer(model, nb_filters, rate)
    model = MaxPooling1D()(model)
    model = Dropout(0.1)(model)
    return model, rate


def dense_layer(model, units, drop_rate):
    """Fully-connected-ReLU-Batch Normalization-Dropout layer."""

    model = Dense(units=units, activation='relu')(model)
    model = BatchNormalization()(model)
    model = Dropout(drop_rate)(model)
    return model


def lr_scheduler(epochs):
    """Multiplies learning rate by 0.1 at 100 and 150 epochs, i.e.,
    new learning rate = old learning rate * 0.1"""

    switch_points = [0, 99, 149]
    for i in [2, 1, 0]:
        if epochs >= switch_points[i]:
            return 0.001 * pow(0.1, i)


# CNN function
def dir_cnn():
    """Load data, normalize metadata, and prepare only the packet direction information for input to Var-CNN.
    Then initialize the Var-CNN model, compute final softmax output, and return time taken to run."""

    data_dir = "/home/primes/attack_scripts/open_world/preprocess"

    # read in data from numpy files
    train_metadata = np.load(r"%s/train_metadata.npy" % data_dir)
    test_metadata = np.load(r"%s/test_metadata.npy" % data_dir)
    train_seq = np.load(r"%s/train_seq.npy" % data_dir)
    train_labels = np.load(r"%s/train_labels.npy" % data_dir)
    test_seq = np.load(r"%s/test_seq.npy" % data_dir)
    test_labels = np.load(r"%s/test_labels.npy" % data_dir)

    # apply normalization to metadata
    metadata_scaler = StandardScaler()
    train_metadata = metadata_scaler.fit_transform(train_metadata)
    test_metadata = metadata_scaler.transform(test_metadata)

    # extract sequences
    train_time, train_time_dleft, train_time_dright, train_dir = np.split(train_seq, 4, axis=2)
    test_time, test_time_dleft, test_time_dright, test_dir = np.split(test_seq, 4, axis=2)

    train_seq = train_dir
    test_seq = test_dir

    # construct CNN
    dilation_rate = 1
    cnn_input = Input(shape=(seq_length, 1,), name='cnn_input')
    cnn_model, dilation_rate = dir_conv_block(cnn_input, 2, 4, dilation_rate)
    cnn_model, dilation_rate = dir_conv_block(cnn_model, 2, 8, dilation_rate)
    cnn_model, dilation_rate = dir_conv_block(cnn_model, 2, 8, dilation_rate)
    cnn_model, dilation_rate = dir_conv_block(cnn_model, 3, 16, dilation_rate)
    cnn_model, dilation_rate = dir_conv_block(cnn_model, 3, 16, dilation_rate)
    cnn_output = Flatten()(cnn_model)
    cnn_output = dense_layer(cnn_output, 1024, 0.4)

    # construct MLP for metadata
    metadata_input = Input(shape=(7,), name='metadata_input')
    metadata_output = dense_layer(metadata_input, 32, 0.)  # consider this the embedding of all the metadata

    # concatenate before second dense layer
    combined = Concatenate()([cnn_output, metadata_output])
    combined = dense_layer(combined, 1024, 0.5)

    # add final softmax layer
    if NUM_UNMON_SITES == 0:  # closed-world
        combined_output = Dense(units=NUM_MON_SITES, activation='softmax', name='combined_output')(combined)
    else:
        # add extra class for unmonitored sites
        combined_output = Dense(units=NUM_MON_SITES + 1, activation='softmax', name='combined_output')(combined)

    model = Model(inputs=[cnn_input, metadata_input], outputs=[combined_output])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    training_data = ({'cnn_input': train_seq,
                      'metadata_input': train_metadata},
                     {'combined_output': train_labels})

    test_data = ({'cnn_input': test_seq,
                  'metadata_input': test_metadata},
                 {'combined_output': test_labels})

    lr_modifier = LearningRateScheduler(schedule=lr_scheduler)

    # train model
    train_time_start = time.time()
    model.fit(x=training_data[0],
              y=training_data[1],
              batch_size=50,
              epochs=200,
              verbose=0,
              callbacks=[lr_modifier])
    train_time_end = time.time()

    # compute final softmax predictions on test set and save predictions
    test_time_start = time.time()
    predictions = model.predict(test_data[0], batch_size=50, verbose=0)
    test_time_end = time.time()
    
    save_dir = "predictions"
    np.save(file=r"%s/dir_model" % save_dir, arr=predictions)
    
    return (train_time_end - train_time_start), (test_time_end - test_time_start)


# CNN function
def time_cnn():
    """Load data, normalize metadata, and prepare only the packet time information for input to Var-CNN.
        Then initialize the Var-CNN model, compute final softmax output, and return time taken to run."""

    data_dir = "/home/primes/attack_scripts/open_world/preprocess"

    # read in data from numpy files
    train_metadata = np.load(r"%s/train_metadata.npy" % data_dir)
    test_metadata = np.load(r"%s/test_metadata.npy" % data_dir)
    train_seq = np.load(r"%s/train_seq.npy" % data_dir)
    train_labels = np.load(r"%s/train_labels.npy" % data_dir)
    test_seq = np.load(r"%s/test_seq.npy" % data_dir)
    test_labels = np.load(r"%s/test_labels.npy" % data_dir)

    # apply normalization to metadata
    metadata_scaler = StandardScaler()
    train_metadata = metadata_scaler.fit_transform(train_metadata)
    test_metadata = metadata_scaler.transform(test_metadata)

    # extract sequences
    train_time, train_time_dleft, train_time_dright, train_dir = np.split(train_seq, 4, axis=2)
    test_time, test_time_dleft, test_time_dright, test_dir = np.split(test_seq, 4, axis=2)

    # reshape to be able to normalize
    train_time = np.reshape(train_time, (train_time.shape[0], train_time.shape[1]))
    test_time = np.reshape(test_time, (test_time.shape[0], test_time.shape[1]))
    train_time_dleft = np.reshape(train_time_dleft, (train_time_dleft.shape[0], train_time_dleft.shape[1]))
    test_time_dleft = np.reshape(test_time_dleft, (test_time_dleft.shape[0], test_time_dleft.shape[1]))
    train_time_dright = np.reshape(train_time_dright, (train_time_dright.shape[0], train_time_dright.shape[1]))
    test_time_dright = np.reshape(test_time_dright, (test_time_dright.shape[0], test_time_dright.shape[1]))

    # apply normalization to packet time data according to scaling computed on train timestamp data
    time_scaler = StandardScaler()
    train_time = time_scaler.fit_transform(train_time)
    test_time = time_scaler.transform(test_time)
    train_time_dleft = time_scaler.transform(train_time_dleft)
    test_time_dleft = time_scaler.transform(test_time_dleft)
    train_time_dright = time_scaler.transform(train_time_dright)
    test_time_dright = time_scaler.transform(test_time_dright)

    train_seq = np.stack((train_time, train_time_dleft, train_time_dright), axis=-1)
    test_seq = np.stack((test_time, test_time_dleft, test_time_dright), axis=-1)

    # construct CNN
    cnn_input = Input(shape=(seq_length, 3,), name='cnn_input')
    cnn_model = time_conv_block(cnn_input, 2, 4)
    cnn_model = time_conv_block(cnn_model, 2, 8)
    cnn_model = time_conv_block(cnn_model, 2, 8)
    cnn_model = time_conv_block(cnn_model, 3, 16)
    cnn_model = time_conv_block(cnn_model, 3, 16)
    cnn_output = Flatten()(cnn_model)
    cnn_output = dense_layer(cnn_output, 1024, 0.4)

    # construct MLP for metadata
    metadata_input = Input(shape=(7,), name='metadata_input')
    metadata_output = dense_layer(metadata_input, 32, 0.)  # consider this the embedding of all the metadata

    # concatenate before second dense layer
    combined = Concatenate()([cnn_output, metadata_output])
    combined = dense_layer(combined, 1024, 0.5)

    # add final softmax layer
    if NUM_UNMON_SITES == 0:  # closed-world
        combined_output = Dense(units=NUM_MON_SITES, activation='softmax', name='combined_output')(combined)
    else:
        # add extra class for unmonitored sites
        combined_output = Dense(units=NUM_MON_SITES + 1, activation='softmax', name='combined_output')(combined)

    model = Model(inputs=[cnn_input, metadata_input], outputs=[combined_output])
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(0.001),
                  metrics=['accuracy'])

    training_data = ({'cnn_input': train_seq,
                      'metadata_input': train_metadata},
                     {'combined_output': train_labels})

    test_data = ({'cnn_input': test_seq,
                  'metadata_input': test_metadata},
                 {'combined_output': test_labels})

    lr_modifier = LearningRateScheduler(schedule=lr_scheduler)

    # train model
    train_time_start = time.time()
    model.fit(x=training_data[0],
              y=training_data[1],
              batch_size=50,
              epochs=200,
              verbose=0,
              callbacks=[lr_modifier])
    train_time_end = time.time()

    # compute final softmax predictions on test set and save predictions
    test_time_start = time.time()
    predictions = model.predict(test_data[0], batch_size=50, verbose=0)
    test_time_end = time.time()
    
    save_dir = "predictions"
    np.save(file=r"%s/time_model" % save_dir, arr=predictions)
    
    return (train_time_end - train_time_start), (test_time_end - test_time_start)


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

    print("running dir model")
    time_train_dir_model, time_test_dir_model = dir_cnn()
    print("running time model")
    time_train_time_model, time_test_time_model = time_cnn()
    print("Total Train Time: %f" % (time_train_dir_model + time_train_time_model))
    print("Total Test Time: %f" % (time_test_dir_model + time_test_time_model))


if __name__ == '__main__':
    main(100, 30, 60, 5500, 3500)
