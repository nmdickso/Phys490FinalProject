from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as funct


class Scinet(nn.Module):

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

        latent = nn.Linear(hyp.encoderNodes[-1], hyp.latentNodes)
        self.latent = latent

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
        self.leadingLoss = hyp.leadingLoss
        self._defineLoss()

    def _defineLoss(self):

        if self.leadingLoss == 'BCE':
            leadingLoss = nn.BCELoss(reduction='sum')
        else:
            leadingLoss = nn.MSELoss(reduction='sum')

        def VAE_loss(mu, sig, X_rec, X):
            leading = leadingLoss(X_rec, X)
            std = torch.exp(sig.mul_(0.5))
            D_KL = 0.5 * (1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))
            return leading - D_KL.sum()
        
        self.lossFunct = VAE_loss

    def encode(self, x):
        # Pass through encoder layers
        for layer in self.encoder:
            x = funct.relu(layer(x))
        # Create mu and sig    
        mu = self.fc_mu(x)
        sig = self.fc_sig(x)
        # Return them
        return mu, sig

    def reparameterize(self, mu, sig):
        # generate normal dist
        gauss = torch.randn_like(sig)
        # convert sig to std
        std = torch.exp(sig.mul_(0.5))
        # Appy formula for Z
        Z = mu + gauss * std
        # Return them
        return Z

    def decode(self, Z, question):
        # Concatenate output of encoder with question
        Z = torch.cat((question, Z), dim=-1)

        # Pass through decoder layers (without applying relu on answer neuron)
        lastDecoderLayer = len(self.decoder) - 1

        for i, layer in enumerate(self.decoder):
            Z = layer(Z)

            if i != lastDecoderLayer:
                Z = funct.relu(Z)
        return Z

    def forward(self, x, question, verbose=False):

        # pass through encoder
        mu, sig = self.encode(x)

        # reparamaterize
        Z = self.reparameterize(mu, sig)

        # decode
        Y = self.decode(Z, question)

        return mu, sig, Z, Y

    def train(self, observations, questions, answers, batchSize, verbose=False):

        avgLoss = 0
        trainSize = len(observations)

        for i in range(0, trainSize, batchSize):

            observationBatch = observations[i:i + batchSize].to(self.device)
            answersBatch = answers[i:i + batchSize].to(self.device)
            questionBatch = questions[i:i + batchSize].to(self.device)

            self.zero_grad()
            mu, sig, Z, outputs = self(observationBatch, questionBatch, verbose=verbose)
            loss = self.lossFunct(mu, sig, outputs, answersBatch)
            loss.backward()
            self.optimizer.step()

            avgLoss += loss.item() * len(observationBatch)

        avgLoss /= trainSize

        if verbose:
            print("Training loss:", avgLoss)

        return (avgLoss)

    def test(self, observations, questions, answers, batchSize, verbose=False):

        avgLoss = 0
        testSize = len(observations)

        with torch.no_grad():
            # for i in tqdm(range(0, testSize, batchSize)):
            for i in range(0, testSize, batchSize):

                observationBatch = observations[i:i + batchSize].to(self.device)
                answersBatch = answers[i:i + batchSize].to(self.device)
                questionBatch = questions[i:i + batchSize].to(self.device)

                mu, sig, Z, outputs = self(observationBatch, questionBatch, verbose=verbose)
                loss = self.lossFunct(mu, sig, outputs, answersBatch)

                avgLoss += loss.item() * len(observationBatch)

        avgLoss /= testSize

        if verbose:
            print("Testing Loss:", avgLoss)

        return (avgLoss)
