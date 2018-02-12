from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import preprocess_data
import var_cnn_ensemble
import evaluate_ensemble

orig_stdout = sys.stdout
f = open('run_open_closed_world_out.txt', 'a', 1)
sys.stdout = f

# closed-world
for i in range(1, 11):
    print("closed-world try = ", i)
    # calls are in the format (num_mon_sites, num_mon_inst_test, num_mon_inst_train,
    # num_unmon_sites_test, num_unmon_sites_train)
    preprocess_data.main(100, 30, 60, 0, 0)
    var_cnn_ensemble.main(100, 30, 60, 0, 0)
    evaluate_ensemble.main(100, 30, 60, 0, 0)

# open-world
for i in range(1, 11):
    print("open-world try = ", i)
    preprocess_data.main(100, 30, 60, 5500, 3500)
    var_cnn_ensemble.main(100, 30, 60, 5500, 3500)
    evaluate_ensemble.main(100, 30, 60, 5500, 3500)
    
sys.stdout = orig_stdout
f.close()
