from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers import Dense, ELU, Activation, Dropout, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adamax
from keras import Input


def get_model(config):
    """Returns deep fingerprinting model to run_model.py

    Args:
        config (dict): Deserialized JSON config file (see config.json)
    """

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

    seq_length = config['seq_length']

    dir_input = Input(shape=(seq_length, 1,), name='dir_input')

    # Block 1
    x = Conv1D(32, 8, strides=1, padding='same')(dir_input)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = Conv1D(32, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = ELU(alpha=1.0)(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 2
    x = Conv1D(64, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(64, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 3
    x = Conv1D(128, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(128, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)

    # Block 4
    x = Conv1D(256, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(256, 8, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=8, strides=4)(x)
    x = Dropout(0.1)(x)
    x = Flatten()(x)

    # FC layers
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(512)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    # Add final softmax layer
    output_classes = num_mon_sites if num_unmon_sites == 0 else num_mon_sites + 1
    model_output = Dense(units=output_classes, activation='softmax',
                         name='model_output')(x)

    model = Model(inputs=dir_input, outputs=model_output)
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adamax(0.002),
                  metrics=['accuracy'])

    callbacks = []
    return model, callbacks
