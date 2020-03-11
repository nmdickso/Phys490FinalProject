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


## Examples
### Damped Pendulum

The time evolution of the system is given by the differential equation in the top right of the slide. One that I am sure most of you are familiar with.
m is the mass of the pendulum, k is the spring constant and b is the damping factor. Only k and b matter as m can simply be absorbed into them.
![equation](https://latex.codecogs.com/gif.latex?m%5Cddot%7Bx%7D%20%3D%20-kx%20-b%20%5Cdot%7Bx%7D)
The authors created the scinet structure with 3 latent neurons.

The network was given position timeseries data of a damped pendulum as the training observations. The question posed to the network was “Where would the pendulum be at time t, outside the observation timeseries?”

As can be seen in the position graph, it could do so very well, with a RMS accuracy less than 2%

Examining the sensitivity of the latent neurons to k and b revealed that the system was correctly determining these physical constants as the only ones necessary to predict a solution.

The neuron’s linear activation wrt each constant are illustrated in the bottom plots.
There was no sensitivity of the third neuron to either of the parameters, meaning it gave no important information toward the final prediction, as one might expect.


#### Recreating the Dampled Pendulum


#### Extending the Dampled Pendulum

### Copernican Heliocentrism
In the 1500’s Copernicus observed the the complex motion of the planets in the sky and postulated a heliocentric model as the simplest representation of the solar system.
To investigate the same process, we pose the problem to SciNet of predicting, based on prior positions, the angles of the Sun and Mars, as seen from Earth, at some future time. That is, describing the time evolution of the planetary orbits.
In order to simulate time evolution, a small feed-forward network is introduced after the representation, before the decoder is applied, transforming SciNet into a recurrent neural network.
In the paper, it is shown that the information stored in the time-evolved latent representation actually corresponds to the angles of the planets as seen from the Sun, demonstrating that the simplest representation it can find is, as Copernicus, the heliocentric model.

#### Recreating the Dampled Pendulum
#### Extending the Dampled Pendulum

### Representation of Qubits
An interesting property of scinet is its ability to determine properties of physical systems with no prior theoretical structure. Take the case of determining the dimensionality of a Hilbert space

-For this, scinet is given the average measurement identical states psi with respect to basis states phi.It is then posed a question, “what will be the result of a measurement of psi with respect to a random state omega?”

-It is important to stress that omega, is parameterized in such a way as to not hint at any human invented structure

-For tomographically complete observations, scinet’s Error drops with the number of representation neurons until said number equals the degrees of freedom of the hilbert space, after which the predictions have near perfect accuracy 

-For tomographically incomplete data scinets Error plateaus before the number of representations neurons reaches the degrees of freedom of the hilbert space, with further representation neurons not decreasing Error

Rough notes for possible extension:
What would happen if you had psi be an entangled 2 qubit system, and we added a non fixed input (call it beta) which represented average result of a the partial measurement of the first system, we then had omega be the parameterization of a partial measurement on the second system. Would scinet respond similarly to decreasing the number latent neurons for both non-entangled and entangled partial measurements (my hypothesis would be yes).

#### Recreating the Dampled Pendulum
#### Extending the Dampled Pendulum


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

### Running Dampled Pendulum

### Running Heliocentrism

### Running Representation of Qubits