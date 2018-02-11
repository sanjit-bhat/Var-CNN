from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import preprocess_script
import run_ensemble
import evaluate_ensemble_confidence

orig_stdout = sys.stdout
f = open('results.txt', 'a', 1)
sys.stdout = f

# closed-world
for i in range(1, 1):
    print("closed-world try = ", i)
    # calls are in the format (num_sens_sites, num_sens_inst_test, num_sens_inst_train, num_insens_sites_test, num_insens_sites_train)
    preprocess_script.main(100, 30, 60, 0, 0)
    run_ensemble.main(100, 30, 60, 0, 0)
    evaluate_ensemble_confidence.main(100, 30, 60, 0, 0)

# open-world
for i in range(6, 11):
    print("open-world try = ", i)
    preprocess_script.main(100, 30, 60, 5500, 3500)
    run_ensemble.main(100, 30, 60, 5500, 3500)
    evaluate_ensemble_confidence.main(100, 30, 60, 5500, 3500)
    
sys.stdout = orig_stdout
f.close()
