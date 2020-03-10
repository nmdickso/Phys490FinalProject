# Phys 490 Final Project: Discovering Physical Concepts with Neural Networks

# The Paper

The paper of interest is "Discovering physical concepts with neural networks", by R. Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner (arXiv:1807.10300v1 (2018)).

Humans tend to build new models around existing ones, which can be counterproductive, calling into question whether current models are truly the simplest way of explaining data. The paper questions how machine learning can combat our limitations and tell us whether we are right or wrong in our modelling. A network can begin with no assumptions, unlike a human mind, completely unbiased. While neural networks have been applied to a variety of problems in physics, most work to date has focused on their efficiency or quality of predictions, without understanding what it  has really learned.

The goal of this paper is to create a neural network which can learn physics. The paper introduces SciNet, a neural network architecture that models physical reasoning processes, which can be used to extract simple physical concepts from experimental data in an unbiased manner. The network compresses experimental data into a simple (latent) representation to answer questions about the system. 

![helio](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/helio.JPG)

Human learning, as visualized in figure (a) below, consists of first compressing experimental observations into a simple representation (*encoding*). At a later time, if asked a question about the physical system, one can produce an answer using the representation, withouth the original data (*decoding*). For example, the observation can be a few seconds of watching someone ski down a hill. The representation can be parameters: speed (*v*) and initial position (*x<sub>0</sub>*). The decoder implements the appropriate equation of motion using the information provided by the representation, to produce an answer.

A physicists reasoning can be broken into four parts, visualized in figure (a).

# SciNet

The neural network structure *SciNet* was recreated in PyTorch. As seen in figure (b) below, the observations are encoded as real parameters, and are fed to an encoder (a feed-forwward neural network). The encoder compresses the data into a latent representation. The question is encoded in N real parameters, which together with the latent representation are fed to the decoder network to generate an answer. 

This network combines both supervised and unsupervised learning. Although the network is trained using a supervised method, we’re interested in the latent representation itself which is trained in an unsupervised way.

![scinet](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/scinet.JPG)

## Recreating SciNet

SciNet is a simply-connected neural network with a bottleneck and an additional input on one of its hidden layers. The question neuron is done by concatenating the output of the layer that comes before the latent layer with the question neuron(s), which is then be passed to the first layer of the decoder (done in the forward function).

For the loss function and optimizer, a few choices were compared before settling with mean square loss and adam, respectively.


## Extending the Examples


# Requirements

- Python 3.x
- ``numpy``
- ``matplotlib``
- ``torch``
- ``tqdm``

## Running the Network

1. Clone the repository.
2. Import `from scinet import *`. This includes the shortcuts `nn` to the `model.py` code and `dl` to the `data_loader.py` code.
3. Import additional files (e.g. data generation scripts) using `import scinet.my_data_generator as my_data_gen_name`.

Generated data files are stored in the ``data`` directory. Saved models are stored in the ``save`` directory

