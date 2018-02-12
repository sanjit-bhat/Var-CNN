This is the code for the paper "Var-CNN and DynaFlow: Improved Attacks and Defenses for Website Fingerprinting".

Before you begin, first ensure that you have a functioning machine with an NVIDIA GPU inside it. Without a GPU, the model will take significantly longer to run on a CPU. To set up the relevent software stack, see this link: https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415 under the "Software Setup" section. For our experiments, we used Ubuntu 16.04 LTS, CUDA 8.0, CuDNN v6, and TensorFlow 1.3.0 as a backend for Keras 2.0.8.

To run the attack, make sure preprocess_data.py, var_cnn_ensemble.py, evaluate_ensemble.py, and run_open_closed_world.py are in the same folder.
In that folder, make directories called preprocess and predictions. These will be used to store the randomized train/test
sets and the final softmax outputs of the packet time and packet direction models.

Make the following changes to the scripts:
1. For preprocess_data.py, change data_loc to point to the location of Wang et al.'s k-NN data set.
You can download Wang et al.'s dataset here: https://www.cse.ust.hk/~taow/wf/data/.

2. For var_cnn_ensemble.py, change the 2 instances of data_dir to point to the location of the preprocess folder.

3. For evaluate_ensemble, change prediction_dir and data_dir to point to the location of the prediction and preprocess folders.

To re-create the open- and closed- world results of our paper, execute run_open_closed_world.py. This script will run open- closed- world scenarios 10 times and output the results for CNN-Var time, direction, and ensemble. In addition, by changing the .main calls to the 3 other scripts here, you can re-produce our experiment with the trained unmonitored sites.
