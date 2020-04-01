import torch.optim as optim
import torch.nn as nn


class Hyperparameters:
    def __init__(self):

        # Observation nodes and encoder
        self.encoderNodes = [5, 100, 100]
        self.encoderLayers = len(self.encoderNodes)

        self.latentNodes = 2
        self.questionNodes = 1

        # Decoder and answer nodes
        self.decoderNodes = [100, 100, 5]
        self.decoderLayers = len(self.decoderNodes)
        self.answerNodes = 1

        self.learningRate = 0.001

        self.optimizer = optim.Adam
        self.leadingLoss = nn.MSELoss

        self.annealEpoch = None


