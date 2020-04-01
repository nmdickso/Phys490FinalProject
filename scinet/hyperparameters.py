import torch.optim as optim
import torch.nn as nn


class Hyperparameters:
    '''Container class for network training parameters

    Configuration container used to store all necessary hyperparameters for
    the training of Scinet. Utilized in the initialization of the network.

    It is expected that the default attributes will be changed to suit specific
    needs before intialization of any networks

    Attributes
    ----------

    encoderNodes : listof int
        An ordered list of the number of neurons to make up the encoder layer of
        the network. The first value must correspond to the number of input
        (observation) neurons

    latentNodes : int
        The number of neurons to make up the latent layer of the network.

    questionNodes : int
        The number of neurons to concatenate to the latent representation before
        passing to the decoder

    decoderNodes : listof int
        An ordered list of the number of neurons to make up the decoder layer of
        the network. The last value must correspond to the number of output
        (answer) neurons

    learningRate : float
        Constant learning rate to be applied throughout network training

    optimizer : torch.optim.Optimizer
        Optimizing algorithm to be used in network training. Must be the
        uninitialized class, not an object instance.

    leadingLoss : torch.nn._Loss
        Cost function which is used as the leading term alongside KL-divergence
        in the VAE loss function. Must be the uninitialized class, not an
        object instance.

    annealEpoch : int or None
        Parameter or adjusting the β-annealed weight. If `None`, annealing will
        not be performed and β will be a constant (1). Otherwise, β is computed
        based on the epoch number stored in the network's `trainCounter`
        attribute

    '''

    def __init__(self):

        # Observation nodes and encoder
        self.encoderNodes = [5, 100, 100]
        self.encoderLayers = len(self.encoderNodes)

        # Latent nodes
        self.latentNodes = 2
        self.questionNodes = 1

        # Decoder and answer nodes
        self.decoderNodes = [100, 100, 5]
        self.decoderLayers = len(self.decoderNodes)

        # Learning rate
        self.learningRate = 0.001

        # Training functions
        self.optimizer = optim.Adam
        self.leadingLoss = nn.MSELoss

        # β-annealing
        self.annealEpoch = None
