from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import json
import os
import time
import numpy as np

import var_cnn
import df
import evaluate
import preprocess_data
import data_generator


def update_config(config, updates):
    """Updates config dict and config file with updates dict."""
    config.update(updates)
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)


def is_valid_mixture(mixture):
    """Check if mixture is a 2D array with strings representing the models."""
    assert type(mixture) == list and len(mixture) > 0
    for inner_comb in mixture:
        assert type(inner_comb) == list and len(inner_comb) > 0
        for model in inner_comb:
            assert model in ['dir', 'time', 'metadata']


def train_and_val(config, model, callbacks, mixture_num, sub_model_name):
    """Train and validate model."""
    print('training %s %s model' % (model_name, sub_model_name))

    train_size = int(
        (num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.95)
    train_steps = train_size // batch_size
    val_size = int(
        (num_mon_sites * num_mon_inst_train + num_unmon_sites_train) * 0.05)
    val_steps = val_size // batch_size

    train_time_start = time.time()
    model.fit_generator(
        data_generator.generate(config, 'training_data', mixture_num),
        steps_per_epoch=train_steps if train_size % batch_size == 0 else train_steps + 1,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
        validation_data=data_generator.generate(
            config, 'validation_data', mixture_num),
        validation_steps=val_steps if val_size % batch_size == 0 else val_steps + 1,
        shuffle=False)
    train_time_end = time.time()

    print('Total training time: %f' % (train_time_end - train_time_start))


def predict(config, model, mixture_num, sub_model_name):
    """Compute and save final predictions on test set."""
    print('generating predictions for %s %s model'
          % (model_name, sub_model_name))

    if model_name == 'var-cnn':
        model.load_weights('model_weights.h5')

    test_size = num_mon_sites * num_mon_inst_test + num_unmon_sites_test
    test_steps = test_size // batch_size

    test_time_start = time.time()
    predictions = model.predict_generator(
        data_generator.generate(config, 'test_data', mixture_num),
        steps=test_steps if test_size % batch_size == 0 else test_steps + 1,
        verbose=0)
    test_time_end = time.time()

    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    np.save(file='%s%s_model' % (predictions_dir, sub_model_name),
            arr=predictions)

    print('Total test time: %f' % (test_time_end - test_time_start))


with open('config.json') as config_file:
    config = json.load(config_file)
    if config['model_name'] == 'df':
        update_config(config, {'mixture': [['dir']], 'batch_size': 128})

num_mon_sites = config['num_mon_sites']
num_mon_inst_test = config['num_mon_inst_test']
num_mon_inst_train = config['num_mon_inst_train']
num_mon_inst = num_mon_inst_test + num_mon_inst_train
num_unmon_sites_test = config['num_unmon_sites_test']
num_unmon_sites_train = config['num_unmon_sites_train']
num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train

data_dir = config['data_dir']
model_name = config['model_name']
mixture = config['mixture']
batch_size = config['batch_size']
predictions_dir = config['predictions_dir']
epochs = config['var_cnn_max_epochs'] if model_name == 'var-cnn' \
    else config['df_epochs']
is_valid_mixture(mixture)

if not os.path.exists('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites,
                                            num_mon_inst,
                                            num_unmon_sites_train,
                                            num_unmon_sites_test)):
    preprocess_data.main(config)

for mixture_num, inner_comb in enumerate(mixture):
    model, callbacks = var_cnn.get_model(config, mixture_num) \
        if model_name == 'var-cnn' else df.get_model(config)

    sub_model_name = '_'.join(inner_comb)
    train_and_val(config, model, callbacks, mixture_num, sub_model_name)
    predict(config, model, mixture_num, sub_model_name)

print('evaluating mixture on test data...')
evaluate.main(config)
