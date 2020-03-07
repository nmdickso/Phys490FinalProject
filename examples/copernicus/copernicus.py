import scinet.scinet
import scinet.hyperparameters as hyper

import torch
import torch.nn as nn
import torch.nn.functional as funct


class TimeEvolvedScinet(scinet.scinet.Scinet):

    def __init__(self, hyp):
        super().__init__(hyp)

        N = hyp.latentNodes

        self.evolver = nn.ModuleList((nn.Linear(N, N), nn.Linear(N, N)))

    def forward(self, x):
        # Dummy question neuron
        # question=torch.Tensor(x.size()[0]*[[0]]).to(self.device)

        # Pass through encoder layers
        for layer in self.encoder:
            x = funct.relu(layer(x))

        # Pass through latent layer
        x = funct.relu(self.latent(x))

        # pass through time evolution network
        for layer in self.evolver:
            x = funct.relu(layer(x))

        # Pass through decoder layers (without applying relu on answer neuron)
        lastDecoderLayer = len(self.decoder) - 1

        for i, layer in enumerate(self.decoder):
            x = layer(x)

            if i != lastDecoderLayer:
                x = funct.relu(x)

        return x


if __name__ == '__main__':
    hyp = hyper.Hyperparameters()

    model = TimeEvolvedScinet(hyp)
