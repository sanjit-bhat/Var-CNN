import h5py
import threading


class ThreadSafeIter:
    """Takes an iterator/generator and makes it thread-safe.

    Does this by serializing call to the `next` method of given iterator/
    generator. See https://anandology.com/blog/using-iterators-and-generators/
    for more information.
    """

    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):  # Py3
        with self.lock:
            return next(self.it)

    def next(self):  # Py2
        with self.lock:
            return self.it.next()


def thread_safe_generator(f):
    """Decorator that takes a generator function and makes it thread-safe."""

    def g(*a, **kw):
        return ThreadSafeIter(f(*a, **kw))

    return g


@thread_safe_generator
def generate(config, data_type, mixture_num):
    """Yields batch of data with the correct content and formatting.

    Args:
        data_type (str): Either 'training_data', 'validation_data', or
            'test_data'
        config (dict): Deserialized JSON config file (see config.json)
    """

    num_mon_sites = config['num_mon_sites']
    num_mon_inst_train = config['num_mon_inst_train']
    num_mon_inst_test = config['num_mon_inst_test']
    num_mon_inst = num_mon_inst_train + num_mon_inst_test
    num_unmon_sites_train = config['num_unmon_sites_train']
    num_unmon_sites_test = config['num_unmon_sites_test']

    data_dir = config['data_dir']
    batch_size = config['batch_size']
    mixture = config['mixture']
    use_dir = 'dir' in mixture[mixture_num]
    use_time = 'time' in mixture[mixture_num]
    use_metadata = 'metadata' in mixture[mixture_num]

    with h5py.File('%s%d_%d_%d_%d.h5' % (data_dir, num_mon_sites, num_mon_inst,
                                         num_unmon_sites_train,
                                         num_unmon_sites_test), 'r') as f:
        # Stores a **reference** of the data, not the actual data, in memory
        dir_seq = f[data_type + '/dir_seq']
        time_seq = f[data_type + '/time_seq']
        metadata = f[data_type + '/metadata']
        labels = f[data_type + '/labels']

        batch_start = 0
        while True:
            if batch_start >= len(labels):
                batch_start = 0
            batch_end = batch_start + batch_size

            batch_data = ({},
                          {'model_output': labels[batch_start:batch_end]})

            # Accesses and stores relevant data slices
            if use_dir:
                batch_data[0]['dir_input'] = dir_seq[batch_start:batch_end]
            if use_time:
                batch_data[0]['time_input'] = time_seq[batch_start:batch_end]
            if use_metadata:
                batch_data[0]['metadata_input'] = metadata[
                                                  batch_start:batch_end]

            batch_start += batch_size
            # Test data does not use labels
            if data_type == 'test_data':
                yield batch_data[0]
            else:
                yield batch_data
