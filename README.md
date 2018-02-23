# Var-CNN and DynaFlow

This repository contains the code and data set for the attack model and defense described in the following paper

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

DynaFlow is a new countermeasure based on dynamically adjusting flows to protect against website fingerprinting attacks. DynaFlow provides a similar level of security as current state-of-the-art and defeats all attacks, including our own, while being over 40% more efﬁcient than existing defenses. 

<img src="https://user-images.githubusercontent.com/26041354/36411425-90260782-15e3-11e8-8022-997fb73707bb.png" width="480">

Figure 1: Var-CNN convolutional feature extractor.


<img src="https://user-images.githubusercontent.com/26041354/36411430-9613e8c6-15e3-11e8-9521-b5ce19a4ff80.png" width="480">

Figure 2: Full Var-CNN architecture.

## Dependencies
1. Ensure that you have a functioning machine with an NVIDIA GPU inside it. Without a GPU, the model will take significantly longer to run on a CPU. 
2. To set up the relevent software stack, see the instructions [here](https://blog.slavv.com/the-1700-great-deep-learning-box-assembly-setup-and-benchmarks-148c5ebe6415) under the "Software Setup" section. For our experiments, we used Ubuntu 16.04 LTS, CUDA 8.0, CuDNN v6, and TensorFlow 1.3.0 as a backend for Keras 2.0.8.

## Var-CNN
### Setup
1. Clone this repo: ```git clone https://github.com/sanjit-bhat/Var-CNN--DynaFlow```
2. In the same directory, make sub-directories called ```preprocess``` and ```predictions```. These will be used to store the randomized train/test
sets and the final softmax outputs of the packet time and packet direction models.
3. For ```preprocess_data.py```, change ```data_loc``` to point to the location of Wang et al.'s k-NN data set.
You can download Wang et al.'s data set [here](https://www.cse.ust.hk/~taow/wf/data/).
4. For ```var_cnn_ensemble.py```, change the 2 instances of ```data_dir``` to point to the location of the ```preprocess``` directory.
5. For ```evaluate_ensemble```, change `prediction_dir` and `data_dir` to point to the location of the ```prediction``` and ```preprocess``` directories.

### Usage
1. To re-create the open- and closed-world results of our paper, run ```python run_open_closed_world.py```. This script will run open- and closed-world scenarios 10 times and output the results for Var-CNN time, direction, and ensemble at varying minimum confidence levels. 
2. By changing the parameters of the .main calls inside ```run_open_closed_world.py```, you can re-produce our experiment with the trained unmonitored sites.

### Results on Wang et al. data set
Attack | Accuracy (Closed) | TPR (Open) | FPR (Open)
-------|:-------:|:--------:|:--------:|
*k*-NN | 91 ± 3 | 85 ± 4 | 0.6 ± 0.4
*k*-FP |91 ± 1 | 88 ± 1 | 0.5 ± 0.1
SDAE | 88 | 86 | 2
Var-CNN Ensemble (conf. threshold = 0.0) | 93.2 ± 0.5 | 93.0 ± 0.5 | 0.7 ± 0.1
Var-CNN Ensemble (conf. threshold = 0.5) | 93.2 ± 0.5| 90.9 ± 0.5 | 0.3 ± 0.1

## DynaFlow
### Setup  
1. Clone this repo: ```git clone https://github.com/sanjit-bhat/Var-CNN--DynaFlow```
2. Make a directory called ```choices```.
3. Make a directory called ```batches``` and put ```batch-primes```(found here) inside that directory.  

### Usage
1. To re-create the defense results of our paper, run ```python dynaflow.py```. This will run all the configurations of the defense in both the open- and closed-worlds, creating the defended traces. The condensed version of each defended trace will be saved to the ```choices``` folder. The defense results will be saved to ```dynaflow.results```.
2. Run ```python bounds_closed.py``` and ```python bounds_open.py``` to attain the metrics of the optimal attacker on each defended data set. 
3. To run Var-CNN on a DynaFlow-defended data set, change ```preprocess_data.py``` to point to the location of the defended data set. 
4. To run Wang et al.'s [k-NN](https://www.cse.ust.hk/~taow/wf/attacks/) and Hayes et al.'s [k-FP](https://github.com/jhayes14/k-FP), download their attacks and follow their documentation.  
5. To run other configurations of your choice, change the parameters at the bottom of ```dynaflow.py```. Make sure the paths at the bottom of ```bounds_closed.py``` and ```bounds_open.py``` correspond with those found in ```dynaflow.py```. 
 


## Contact
sanjit.bhat (at) gmail.com

davidboxboro (at) gmail.com

kwonal (at) mit.edu

Any discussions, suggestions, and questions are welcome!
