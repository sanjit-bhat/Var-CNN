from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import random
import json

random.seed(4286794567481)


def process_trace(args):
    """Extract dir seq, time seq, metadata, and label for particular trace."""
    dir_name = args[0]
    trace_path = args[1]

    with open(os.path.join(dir_name, trace_path), 'r') as f:
        lines = f.readlines()

    dir_seq = np.zeros(5000, dtype=np.int8)
    time_seq = np.zeros(5000, dtype=np.float32)
    label = 0 if '-' not in trace_path else int(trace_path.split('-')[0]) + 1
    total_time = float(lines[-1].split('\t')[0])
    total_incoming = 0
    total_outgoing = 0

    for packet_num, line in enumerate(lines):
        line = line.split('\t')
        curr_time = float(line[0])
        curr_dir = np.sign(int(line[1]))

        if packet_num < 5000:
            dir_seq[packet_num] = curr_dir
            time_seq[packet_num] = curr_time

        if curr_dir == 1:
            total_outgoing += 1
        elif curr_dir == -1:
            total_incoming += 1

    total_packets = total_incoming + total_outgoing
    if total_packets == 0:
        metadata = np.zeros(7, dtype=np.float32)
    else:
        metadata = np.array([total_packets, total_incoming, total_outgoing,
                             total_incoming / total_packets,
                             total_outgoing / total_packets,
                             total_time, total_time / total_packets],
                            dtype=np.float32)
    return dir_seq, time_seq, metadata, label


def main(config):

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst = num_mon_inst_test + num_mon_inst_train
    num_unmon_sites_test = config['num_unmon_sites_test']
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites = num_unmon_sites_test + num_unmon_sites_train
    data_dir = config['data_dir']

    arg_list = []
    for trace_path in os.listdir(data_dir + 'batch_wang'):
        arg_list.append([data_dir + 'batch_wang', trace_path])

    # set up the output
    mon_idx = 0
    dir_seq_mon = [None] * (num_mon_sites * num_mon_inst)
    time_seq_mon = [None] * (num_mon_sites * num_mon_inst)
    metadata_mon = [None] * (num_mon_sites * num_mon_inst)
    labels_mon = [None] * (num_mon_sites * num_mon_inst)

    unmon_idx = 0
    dir_seq_unmon = [None] * num_unmon_sites
    time_seq_unmon = [None] * num_unmon_sites
    metadata_unmon = [None] * num_unmon_sites
    labels_unmon = [None] * num_unmon_sites

    print('size of total list: %d' % len(arg_list))
    for i in range(len(arg_list)):
        dir_seq, time_seq, metadata, label = process_trace(arg_list[i])
        if i % 5000 == 0:
            print("processed", i)

        if label == 0:  # unmon site
            dir_seq_unmon[unmon_idx] = dir_seq
            time_seq_unmon[unmon_idx] = time_seq
            metadata_unmon[unmon_idx] = metadata
            labels_unmon[unmon_idx] = label
            unmon_idx += 1
        else:
            dir_seq_mon[mon_idx] = dir_seq
            time_seq_mon[mon_idx] = time_seq
            metadata_mon[mon_idx] = metadata
            labels_mon[mon_idx] = label
            mon_idx += 1

    # save monitored traces
    dir_seq_mon = np.array(dir_seq_mon, dtype=np.int8)
    time_seq_mon = np.array(time_seq_mon, dtype=np.float32)
    metadata_mon = np.array(metadata_mon, dtype=np.float32)

    print('number of monitored traces: %d' % len(labels_mon))
    np.savez_compressed(data_dir + 'all_closed_world.npz', dir_seq=dir_seq_mon,
                        time_seq=time_seq_mon, metadata=metadata_mon,
                        labels=labels_mon)

    # save unmonitored traces
    dir_seq_unmon = np.array(dir_seq_unmon, dtype=np.int8)
    time_seq_unmon = np.array(time_seq_unmon, dtype=np.float32)
    metadata_unmon = np.array(metadata_unmon, dtype=np.float32)

    print('number of unmonitored traces: %d' % len(labels_unmon))
    np.savez_compressed(data_dir + 'all_open_world.npz', dir_seq=dir_seq_unmon,
                        time_seq=time_seq_unmon, metadata=metadata_unmon,
                        labels=labels_unmon)


if __name__ == '__main__':
    with open('config.json') as config_file:
        config = json.load(config_file)

    main(config)
