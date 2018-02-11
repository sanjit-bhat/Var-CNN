This is the code for the paper "Var-CNN and DynaFlow: Improved Attacks and Defenses for Website Fingerprinting".

To run the attack, clone preprocess_script, run_ensemble, evaluate_ensemble_confidence, and run_open_closed_world into the same folder.
In that folder, make directories called preprocess and predictions. These will be used to store the randomized train/test
sets for preprocess_script and the final softmax outputs of the packet time and packet direction models.

Make the following changes to the scripts:
1. For preprocess_script, change data_loc to point to the location of Wang et al.'s k-NN data set.
You can download Wang et al.'s dataset here: https://www.cse.ust.hk/~taow/wf/data/.

2. For run_ensemble, change the 2 instances of data_dir to point to the location of the preprocess folder.

3. For evaluate_ensemble_confidence, change prediction_dir and data_dir to point to the location of the prediction and preprocess folders.

By calling run_open_closed_world, it will re-create the open- and closed- world results of our paper, running each scenario
10 times and outputting open-world results for CNN-Var time, direction, and ensemble under all minimum confidence threshold levels.
