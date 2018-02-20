# Var-CNN and DynaFlow

This repository contains the code and dataset for the attack model and defense described in the following paper

[Var-CNN and DynaFlow: Improved Attacks and Defenses for Website Fingerprinting](https://arxiv.org)

Sanjit Bhat, David Lu, [Albert Kwon](http://www.albertkwon.com), and [Srinivas Devadas](https://people.csail.mit.edu/devadas/).

### Citation
If you find Var-CNN or DynaFlow useful in your research, please consider citing:

	@article{bhat18,
	  title={Var-CNN and DynaFlow: Improved Attacks and Defenses for Website Fingerprinting},
	  author={Bhat, Sanjit and Lu, David and Kwon, Albert and Devadas, Srinivas},
	  journal={arXiv},
	  year={2018}
	}

## Introduction
Most prior work on website fingerprinting attacks use manually extracted features, and thus are fragile to protocol changes or simple defenses. In contrast, Var-CNN uses model variations on convolutional neural networks with both packet sequence and packet timing data, In open-world settings, Var-CNN yields new state-of-the-art results of 90.9% TPR and 0.3% FPR on the Wang et al. data set. 

<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">
Figure 1: Var-CNN convolutional feature extractor.
<img src="https://cloud.githubusercontent.com/assets/8370623/17981494/f838717a-6ad1-11e6-9391-f0906c80bc1d.jpg" width="480">
Figure 2: Full Var-CNN architecture.

DynaFlow is a new countermeasure based on dynamically adjusting flows to protect against website fingerprinting attacks. DynaFlow provides a similar level of security as current state-of-the-art and defeats all attacks, including our own, while being over 40% more efﬁcient than existing defenses. 

## Dependencies
1. Ensure that you have a functioning machine with an NVIDIA GPU inside it. Without a GPU, the model will take significantly longer to run on a CPU. 
2. To set up the relevent software stack, see the instructions [here](https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415) under the "Software Setup" section. For our experiments, we used Ubuntu 16.04 LTS, CUDA 8.0, CuDNN v6, and TensorFlow 1.3.0 as a backend for Keras 2.0.8.

## Attack setup
1. Clone this repo: ```git clone https://github.com/sanjit-bhat/Var-CNN--DynaFlow```
2. In the same directory, make folders called preprocess and predictions. These will be used to store the randomized train/test
sets and the final softmax outputs of the packet time and packet direction models.
3. For preprocess_data.py, change data_loc to point to the location of Wang et al.'s k-NN data set.
You can download Wang et al.'s dataset [here](https://www.cse.ust.hk/~taow/wf/data/).
4. For var_cnn_ensemble.py, change the 2 instances of data_dir to point to the location of the preprocess folder.
5. For evaluate_ensemble, change prediction_dir and data_dir to point to the location of the prediction and preprocess folders.

## Attack usage
1. To re-create the open- and closed- world results of our paper, execute run_open_closed_world.py: ```python run_open_closed_world.py```. This script will run open- and closed- world scenarios 10 times and output the results for CNN-Var time, direction, and ensemble at varying minimum confidence levels. 
2. By changing the parameters of the .main calls inside run_open_closed_world.py, you can re-produce our experiment with the trained unmonitored sites.

## Contact
sanjit.bhat at gmail.com  
davidboxboro at gmail.com
kwonal at mid.edu
Any discussions, suggestions and questions are welcome!
