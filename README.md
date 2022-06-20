# RVAE-LAMOL
source code of IJCNN 2022 oral long paper: "[RVAE-LAMOL:  Residual Variational Autoencoder to Enhance Lifelong Language Learning](https://arxiv.org/pdf/2205.10857.pdf)"

Lifelong Language Learning (LLL) aims to train a neural network to learn a stream of NLP tasks while retaining knowledge from previous tasks. However, previous works which followed data-free constraint still suffer from catastrophic forgetting issue, where the model forgets what it just learned from previous tasks. In order to alleviate catastrophic forgetting, we propose the residual variational autoencoder (RVAE) to enhance LAMOL, a recent LLL model, by mapping different tasks into a limited unified semantic space. In this space, previous tasks are easy to be correct to their own distribution by pseudo samples. Furthermore, we propose an identity task to make the model is discriminative to recognize the sample belonging to which task. For training RVAE-LAMOL better, we propose a novel training scheme Alternate Lag Training. In the experiments, we test RVAE-LAMOL on permutations of three datasets from DecaNLP. The experimental results demonstrate that RVAE-LAMOL outperforms naive LAMOL on all permutations and generates more meaningful pseudo-samples. 

The main code is from original [LAMOL](https://github.com/chho33/LAMOL). Downloading the data is same as [LAMOL](https://github.com/chho33/LAMOL). 


The hyper-parameters have been set as default values in setting.py. The detail can follow our paper "Implementation Detail". The demo about how to run our code will be updated later.


