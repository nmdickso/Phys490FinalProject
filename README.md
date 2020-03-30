# Phys 490 Final Project: Discovering Physical Concepts with Neural Networks

# The Paper

The paper of interest is "Discovering physical concepts with neural networks", by R. Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner (arXiv:1807.10300v1 (2018)).

Humans tend to build new models around existing ones, which can be counterproductive, calling into question whether current models are truly the simplest way of explaining data. The paper questions how machine learning can combat our limitations and tell us whether we are right or wrong in our modelling. A network can begin with no assumptions, unlike a human mind, completely unbiased. While neural networks have been applied to a variety of problems in physics, most work to date has focused on their efficiency or quality of predictions, without understanding what it  has really learned.

The goal of this paper is to create a neural network which can learn physics. The paper introduces SciNet, a neural network architecture that models physical reasoning processes, which can be used to extract simple physical concepts from experimental data in an unbiased manner. The network compresses experimental data into a simple (latent) representation to answer questions about the system. 

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

The time evolution of the damped pendulum system is given by the following differential:
![equation](https://latex.codecogs.com/gif.latex?m%5Cddot%7Bx%7D%20%3D%20-kx%20-b%20%5Cdot%7Bx%7D)

The mass of the pendulum is denoted by *m*, the spring constant by *k*, and the damping factor by *b*. Only *k* and *b* are relevant here, as *m* can simply be absorbed into them. The authors of the paper created the SciNet structure with three latent neurons.

The network is given position timeseries data of a damped pendulum, as the training observations. The question posed to the network is: “Where would the pendulum be at time *t*, outside the observation timeseries?”

As can be seen in the position-time graph below, the network performs quite well, with a root mean squared accuracy of less than 2%.

![pendulum](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/pendulum.JPG)

Examining the sensitivity of the latent neurons to *k* and *b* revealed that the system was correctly determining these physical constants as the only ones required to predict a solution. The neuron’s linear activation with respect to each constant are illustrated in the plots below.
There is no sensitivity of the third neuron to either of the parameters, meaning it gives no important information towards the final prediction, as one might expect.

![pendulum_graph](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/pendulum_graph.JPG)

#### Extending the Dampled Pendulum Model

### Copernican Heliocentrism
In the 1500’s Copernicus observed the complex motion of the planets in the sky and postulated a heliocentric model as the simplest representation of the solar system. To investigate the same process, we posed the problem to SciNet of predicting, based on prior positions, the angles of the Sun and Mars as seen from Earth, at a future time. That is, describing the time evolution of the planetary orbits.

![helio](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/helio.JPG)

In order to simulate time evolution, a small feed-forward network is introduced after the representation, before the decoder is applied, transforming SciNet into a recurrent neural network. In the paper, it is shown that the information stored in the time-evolved latent representation corresponds to the angles of the planets as seen from the Sun, demonstrating that the simplest representation it can find is, as Copernicus discovered, the heliocentric model.

### Representation of Qubits
An interesting property of SciNet is its ability to determine properties of physical systems with no prior theoretical structure. In the case of determining the dimensionality of a Hilbert space, SciNet is given the average "measurement identical states" psi, with respect to "basis states" phi. The following question is then posed: “What is the result of a measurement of psi with respect to a random state omega?”

![Qubits](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/qubits.JPG)

Omega is parameterized in such a way as to not hint at any human invented structure. For tomographically complete observations, SciNet’s error drops with the number of representation neurons until it equals the degrees of freedom of the Hilbert Space, after which the predictions have near perfect accuracy. 

For tomographically *incomplete* data, SciNet's error plateaus before the number of representation neurons reaches the degrees of freedom of the Hilbert Space, with further representation neurons not decreasing in error.

#### Extending the Representation of Qubits Problem
To extend this problem, we can pose the question: "What would happen if psi consisted of an entangled 2 qubit system and a non-fixed input representing the average result of the partial measurement of the first system? Omega would be the parameterization of a partial measurement on the second system. Would SciNet respond in a similar manner to decreasing the number of latent neurons for both non-entangled and entangled partial measurements? 

# Technical Instructions

## Requirements

- Python 3.x
- ``numpy``
- ``matplotlib``
- ``mpl_toolkits``
- ``setuptools``
- ``itertools``
- ``argparse``
- ``datetime``
- ``random``
- ``torch``
- ``tqdm``
- ``io``
- ``os``
- ``sys``

## Installing SciNet
1. Clone the repository
2. Navigate to the `scinet` folder
3. Install SciNet by running `pip install -e .`

### Running the Dampled Pendulum Model
1. Navigate to the DampedPendulum folder
2. *First*, generate the data by including the following in the command line:
- `k`: Spring Constant
- `b`: Damping Constant
- The last flag should be the path and name of file to save output data.

Optional Arguments:
- `x_0`: Initial position of pendulum
- `v_0`: Initial velocity of pendulum
- `num_points`: Number of time points in series
- `domain`: Range of time values covered

```
python .\DampedOscillatorGeneration.py --k_range 5 10 100 --b_range 0.5 1 100 -d 0 30 -np 100 C:\Users\xxx\Desktop\Example_Folder
```

3. To run the example itself, enter the path to where the output data was saved.
```
python main.py --params path\to\params --outdir path\to\savePlots C:\Users\xxx\Desktop\Example_Folder 
```

### Running the Heliocentrism Model

To run the Copernican Heliocentrism example:

```
python examples/copernicus/copernicus.py [-h] [FLAGS] [OPTIONS]
```

Optional Arguments:
- `-M M`: Number of time-evolution steps [default=5]
- `-N N`: Training dataset size [default=15000]
- `--test-N TEST_N`: Testing dataset size [default=1000]
- `-t DEL_T, --del-t DEL_T`: Number of days per time-evolution step [default=7]

- `-a A`: Learning rate (α) [default=0.001]
- `-b B`: Training batch size [default=2000]
- `-E E`: Number of training epochs [default=25]

Optional Flags:
- `-B, --beta-anneal`: Utilize 'Beta-Annealing'
- `-v, --verbose`: Print out a progress bar and loss values during training
- `--plot-loss`: Show a plot of loss over training epochs

### Running the Representation of Qubits Model

To run the Representation of Qubits example:

```
python main.py --numQubits 1 --outputDir path\to\savePlots
```
- `numQubits` must be either the integer 1 or 2 (anything larger than this has an incredibly long runtime)