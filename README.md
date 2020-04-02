# Phys 490 Final Project: Discovering Physical Concepts with Neural Networks

# The Paper

The paper of interest is "Discovering physical concepts with neural networks", by R. Iten, T. Metger, H.Wilming, L. del Rio, and R. Renner (arXiv:1807.10300v1 (2018)).

Humans tend to build new models around existing ones, which can be counterproductive, calling into question whether current models are truly the simplest way of explaining data. The paper questions how machine learning can combat our limitations and tell us whether we are right or wrong in our modelling. A network can begin with no assumptions, unlike a human mind, completely unbiased. While neural networks have been applied to a variety of problems in physics, most work to date has focused on their efficiency or quality of predictions, without understanding what it  has really learned.

The goal of this paper is to create a neural network which can learn physics. The paper introduces SciNet, a neural network architecture that models physical reasoning processes, which can be used to extract simple physical concepts from experimental data in an unbiased manner. The network compresses experimental data into a simple (latent) representation to answer questions about the system. 

Human learning, as visualized in figure (a) below, consists of first compressing experimental observations into a simple representation (*encoding*). At a later time, if asked a question about the physical system, one can produce an answer using the representation, withouth the original data (*decoding*). For example, the observation can be a few seconds of watching someone ski down a hill. The representation can be parameters: speed (*v*) and initial position (*x<sub>0</sub>*). The decoder implements the appropriate equation of motion using the information provided by the representation, to produce an answer.

A physicists reasoning can be broken into four parts, visualized in figure (a).

# SciNet

The neural network structure *SciNet* was recreated in PyTorch. As seen in figure (b) below, the observations are encoded as real parameters, and are fed to an encoder (a feed-forward neural network). The encoder compresses the data into a latent representation. The question is encoded in N real parameters, which together with the latent representation are fed to the decoder network to generate an answer. 

This network combines both supervised and unsupervised learning. Although the network is trained using a supervised method, we’re interested in the latent representation itself which is trained in an unsupervised way.

![scinet](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/scinet.JPG)

## Recreating SciNet

SciNet is a fully connected Variational Auto Encoder (VAE), with a weighting term (β) in the KLD term in the loss function. There are two differences which make SciNet unique from a standard VAE:

1. β is annealed, meaning its value is increased over epochs.
2. There is an additional input into the decoder with the latent layer in the form of the question neurons.

The only complexity in recreating this network pertains to the question neuron. This is done by concatenating the output of the latent layer with the question neurons, all of which is then passed through the decoder.

## Examples
### Damped Pendulum

The time evolution of the damped pendulum system is given by the following differential:
![equation](https://latex.codecogs.com/gif.latex?m%5Cddot%7Bx%7D%20%3D%20-kx%20-b%20%5Cdot%7Bx%7D)

The mass of the pendulum is denoted by *m*, the spring constant by *k*, and the damping factor by *b*. Only *k* and *b* are relevant here, as *m* can simply be absorbed into them. The authors of the paper created the SciNet structure with three latent neurons.

The network is given position timeseries data of a damped pendulum, as the training observations. The question posed to the network is: “Where would the pendulum be at time *t*?”

As can be seen in the position-time graph below, the network performs quite well for the authors, with a root mean squared accuracy of less than 2%.

![pendulum](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/pendulum.JPG)

Examining the sensitivity of the latent neurons to *k* and *b* revealed that the system was correctly determining these physical constants as the only ones required to predict a solution. The neuron’s linear activation with respect to each constant are illustrated in the plots below.
There is no sensitivity of the third neuron to either of the parameters, meaning it gives no important information towards the final prediction, as one might expect.

![pendulum_graph](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/pendulum_graph.JPG)

#### Extending the Dampled Pendulum Model

The extension of the damped pendulum takes inspiration from the Lorentz Model Oscillator used in Optics. In this extension, a third term is present in the differential equation. This term is known as the driving force, or electric force, on the electron cloud. It is a wave (a harmonic wave in this case) which depends on some frequency w (omega) of the light causing the EM perturbation.

The extension introduces new physics to the constants, where the spring constant is the resonant frequency of the material, and the damping constant b becomes a dissipation term for energy lost (e.g. lost to heating from light).

### Copernican Heliocentrism

From the earth, other planetary bodies can be seen to take complex paths through the sky, exhibiting elements like retrograde motion.
In the 1500’s Copernicus observed the motion of the planets in the sky and postulated that these features could be explained in the simplest representation by a heliocentric solar system, with simple circular planetary orbits.

To examine this same process, we tasked Scinet with predicting, based on prior observations, the angles of the Sun and Mars as seen from Earth, as some future time. That is, describing the time evolution of the planetary orbits.
 
To facilitate the time-evolution, we modify Scinet by introducing a small recurrent neural network onto the latent layers.
The small feed-forward layers map a simple translation from r<sub>j</sub>(t<sub>i</sub>) → r<sub>j</sub>(t<sub>i</sub>) + b<sub>j</sub> for each timestep, before decoding back to the known Earth-angles.

![helio](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/helio.JPG)

In the paper, it is shown that the information stored in the time-evolved latent representation corresponds to a linear combination of the sun angles ϕ, demonstrating that the simplest representation it can find is, as Copernicus discovered, the heliocentric model.

### Representation of Qubits
An interesting property of SciNet is its ability to determine properties of physical systems with no prior theoretical structure. Take the case of determining the dimensionality of a Hilbert space

For this, SciNet is given the average measurement of arbitrary states psi with respect to some fixed basis states phi. These are our observations.

SciNet is then posed a question, “What will be the result of a measurement of psi with respect to a random state omega?”

It is important to stress that omega is parameterized in such a way as to not hint at any human invented structure (it is given as a set of binary projective measurements with respect to some other basis similar to the observations).

For tomographically complete observations, SciNet’s error drops with the number of representation neurons until said number equals the degrees of freedom of the Hilbert Space, after which the predictions have near perfect accuracy.

For tomographically incomplete data, SciNet's error plateaus before the number of representation neurons reaches the degrees of freedom of the Hilbert Space, with further representation neurons not decreasing error.


![Qubits](https://github.com/nmdickso/Phys490FinalProject/blob/veronica/images/qubits.JPG)


#### Extending the Representation of Qubits Problem
For SciNet to be a useful tool, it must be able to run in a more reasonable timeframe. To accomplish this, we used learning rate annealing to drastically cut down the number of epochs required to train SciNet (down to 13.3% of what is used in the paper for the single qubit example). In addition to this, the script will automatically detect if a CUDA device is avalible, to make use of GPU acceleration, which is what was used to get our results.

# Technical Instructions

## Requirements

- ``numpy``
- ``matplotlib``
- ``torch``
- ``tqdm``

## Installing SciNet
1. Clone the repository
2. Navigate to the cloned repository (`Phys490FinalProject`)
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
python examples/quantumTomography/main.py numQubits outputDir
```
- `numQubits` must be either the integer 1 or 2 (anything larger than this has an incredibly long runtime). The default is 1.
- `outputDir` name of a folder in quantumTomography is optional. The plots default to saving in the `Results` folder.
