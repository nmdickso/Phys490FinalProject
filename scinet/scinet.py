import torch
import torch.nn as nn
import torch.nn.functional as funct


class Scinet(nn.Module):
    '''Neural network designed to learn physical concepts

    Scinet is a fully connected β-VAE with a modified decoder accepting an
    extra "question" neuron, designed to learn physical concepts through
    training on basic physics observations.

    Parameters
    ----------
    hyp : hyperparameters.Hyperparameters
        Hyperparameters object guiding the setup of the network

    '''

    def __init__(self, hyp):
        super().__init__()

        # Populate encoder
        self.encoder = nn.ModuleList()
        for i in range(0, hyp.encoderLayers - 1):

            inputNodes = hyp.encoderNodes[i]
            outputNodes = hyp.encoderNodes[i + 1]

            fcLayer = nn.Linear(inputNodes, outputNodes)
            self.encoder.append(fcLayer)

        # create mu and sigma layer
        self.fc_mu = nn.Linear(hyp.encoderNodes[-1], hyp.latentNodes)
        self.fc_sig = nn.Linear(hyp.encoderNodes[-1], hyp.latentNodes)

        # Add latent layer
        inputNodes = hyp.encoderNodes[-1]
        outputNodes = hyp.latentNodes

        # Populate decoder
        # Decoder input is question nodes plus latent nodes
        self.decoder = nn.ModuleList()
        inputNodes = hyp.latentNodes + hyp.questionNodes
        outputNodes = hyp.decoderNodes[0]
        firstDecoderLayer = nn.Linear(inputNodes, outputNodes)
        self.decoder.append(firstDecoderLayer)

        for i in range(1, hyp.decoderLayers - 1):
            inputNodes = hyp.decoderNodes[i]
            outputNodes = hyp.decoderNodes[i + 1]
            fcLayer = nn.Linear(inputNodes, outputNodes)
            self.decoder.append(fcLayer)

        # Use only CPU due to small network size and occasional bugs
        self.device = torch.device("cpu")
        self.to(self.device)

        # Optimizer and loss functions
        self.optimizer = hyp.optimizer(self.parameters(), hyp.learningRate)

        self.leadingLoss = hyp.leadingLoss(reduction='sum')
        self.lossFunct = self._VAE_loss

        self.trainCounter = 0
        self.annealEpoch = hyp.annealEpoch

    @property
    def annealWeight(self):
        if self.annealEpoch is None:
            return 1
        return min(1, self.trainCounter / self.annealEpoch)

    def _VAE_loss(self, mu, sig, X_rec, X):
        '''VAE KL-divergence/entropy cost function'''

        # Basic nn cost function
        leading = self.leadingLoss(X_rec, X)

        # KL-divergence
        std = torch.exp(sig.mul_(0.5))
        D_KL = 0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))

        return leading - self.annealWeight * D_KL.sum()

    def encode(self, x):
        '''VAE encoder layer'''

        # Pass through encoder layers
        for layer in self.encoder:
            x = funct.relu(layer(x))

        # Create mu and sig
        mu = self.fc_mu(x)
        sig = self.fc_sig(x)

        return mu, sig

    def reparameterize(self, mu, sig):
        '''VAE reparameterization function'''

        # Sample from normal distribution
        gauss = torch.randn_like(sig)

        # convert variance to standard deviation
        std = torch.exp(sig.mul_(0.5))

        # Reparameterize the sampled data
        Z = mu + gauss * std

        return Z

    def decode(self, Z, question):
        '''VAE decoder layer'''

        # Concatenate Z with question neurons
        Z = torch.cat((question, Z), dim=-1)

        # Pass through decoder layers (without applying relu on final layer)
        lastDecoderLayer = len(self.decoder) - 1

        for i, layer in enumerate(self.decoder):
            Z = layer(Z)

            if i != lastDecoderLayer:
                Z = funct.relu(Z)

        return Z

    def forward(self, x, question):
        '''VAE forward function

        Variational auto-encoder based forward function, with encoding,
        reparamaterization and decoding.

        Parameters
        ----------
        x : torch.Tensor
            Input training dataset batch

        question : torch.Tensor
            "Question" neurons to be concatenated to the latent layer before
            decoding

        Returns
        -------
        mu : torch.Tensor
            Latent neuron reparameterization mean

        sig : torch.Tensor
            Latent neuron reparameterization standard deviation

        Z : torch.Tensor
            The reparameterized latent neurons

        Y : torch.Tensor
            The decoded final VAE neurons
        '''

        # pass through encoder
        mu, sig = self.encode(x)

        # reparamaterize
        Z = self.reparameterize(mu, sig)

        # Pass through decoder
        Y = self.decode(Z, question)

        return mu, sig, Z, Y

    def train(self, observations, questions, answers, batchSize, verbose=False):
        '''Network training function

        Batched training function, splitting up the input training dataset into
        batches before passing to the forward function and performing
        back-propogation.

        Parameters
        ----------
        observations : torch.Tensor
            Input training dataset

        questions : torch.Tensor
            Question neurons

        answers : torch.Tensor
            Target answer dataset

        batchSize : int
            Batch size

        Returns
        -------
        avgLoss : float
            The averaged value of the cost function over all batches
        '''

        avgLoss = 0
        trainSize = len(observations)

        # Split datset into batches
        for i in range(0, trainSize, batchSize):

            observationBatch = observations[i:i + batchSize].to(self.device)
            answersBatch = answers[i:i + batchSize].to(self.device)
            questionBatch = questions[i:i + batchSize].to(self.device)

            # Pass batch through network
            self.zero_grad()

            mu, sig, Z, outputs = self(observationBatch, questionBatch)

            # Loss function and back-propogation
            loss = self.lossFunct(mu, sig, outputs, answersBatch)
            loss.backward()

            self.optimizer.step()

            # Average losses
            avgLoss += loss.item() * len(observationBatch)

        avgLoss /= trainSize
        self.trainCounter += 1

        if verbose:
            print("Training loss:", avgLoss)

        return avgLoss

    def test(self, observations, questions, answers, verbose=False):
        '''Network testing function

        Test the model effectiveness by passing the input observations through
        the trained network, to observe the final accuracy and the activations
        of the final latent layers.

        Identical to `train`, with the exclusion of all back-propogation,
        gradient descent and batching.

        Parameters
        ----------
        observations : torch.Tensor
            Input testing dataset

        questions : torch.Tensor
            Question neurons

        answers : torch.Tensor
            Target answer dataset

        Returns
        -------
        avgLoss : float
            The averaged value of the cost function over all τ

        latent_activation : torch.Tensor
            Activation of trained network latent neurons
        '''

        avgLoss = 0

        with torch.no_grad():

            observations = observations.to(self.device)
            answers = answers.to(self.device)
            questions = questions.to(self.device)

            # Pass data through network
            mu, sig, latent_activation, outputs = self(observations, questions)

            # Loss calculation
            loss = self.lossFunct(mu, sig, outputs, answers)

            avgLoss = loss.item()

        if verbose:
            print("Testing Loss:", avgLoss)

        return avgLoss, latent_activation
