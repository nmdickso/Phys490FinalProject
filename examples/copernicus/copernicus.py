import scinet
import data_gen

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as funct


class TimeEvolvedScinet(scinet.Scinet):

    def __init__(self, hyp):
        super().__init__(hyp)

        N = hyp.latentNodes

        self.evolver = nn.ModuleList((nn.Linear(N, N), nn.Linear(N, N)))

    def forward(self, x, question):
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

    hyp = scinet.Hyperparameters()
    hyp.encoderNodes[0] = 2
    hyp.latentNodes = 2
    hyp.decoderNodes[-1] = 2
    hyp.questionNodes = 0

    model = TimeEvolvedScinet(hyp)

    # TODO havent really figured out how to apply the time evolution multiple
    #   times, what do we train on?
    φ_e, φ_m, θ_e, θ_m = data_gen.generate_orbits(1000, 2, 7)

    for i in range(100):

        # Observation are the first elements of θ
        obs = torch.from_numpy(np.vstack((θ_e[:, 0], θ_m[:, 0])).T).float()

        # Answers are the proceeding elements of θ, one for each jump
        ans = torch.from_numpy(np.vstack((θ_e[:, -1], θ_m[:, -1])).T).float()

        loss = model.train(obs, torch.Tensor([]), ans, 100)
