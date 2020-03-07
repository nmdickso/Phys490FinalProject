# Phys 490 Final Project: Discovering Physical Concepts with Neural Networks

# The Paper

The paper of interest is "Discovering physical concepts with neural networks", by R. Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner (arXiv:1807.10300v1 (2018)).

![helio](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/helio.JPG)

Human learning, as illustrated in figure a) below, consists of first compressing experimental observations into a simple representation (*encoding*). At a later time, if asked a question about the physical system, one can produce an answer using the representation, withouth the original data (*decoding*). For example, the observation can be a few seconds of watching someone ski down a hill. The representation can be parameters: speed (*v*) and initial position (*x<sub>0</sub>*). The decoder implements the appropriate equation of motion using the information provided by the representation, to produce an answer.

# SciNet

The neural network structure *SciNet* was recreated in PyTorch. As seen in image b) below, the observations are encoded as real parameters, and are fed to an encoder (a feed-forwward neural network). The encoder compresses the data into a latent representation. The question is encoded in N real parameters, which together with the latent representation are fed to the decoder network to generate an answer. 


![scinet](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/scinet.JPG)


# Requirements

- Python 3.x
- ``numpy``
- ``matplotlib``
- ``torch``
- ``tqdm``

# Running the Network

1. Clone the repository.
2. Import `from scinet import *`. This includes the shortcuts `nn` to the `model.py` code and `dl` to the `data_loader.py` code.
3. Import additional files (e.g. data generation scripts) using `import scinet.my_data_generator as my_data_gen_name`.

Generated data files are stored in the ``data`` directory. Saved models are stored in the ``save`` directory

