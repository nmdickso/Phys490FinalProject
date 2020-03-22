import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss(num_epochs, losses):
    
    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.plot(range(num_epochs), losses, color='blue')
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Graph of Loss per Epoch")
    plt.savefig('Graph_Epochs_Loss.pdf')

    # plt.show()

def plot_latent(latent1, latent2, activation):
    
    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for i, ax in enumerate((ax1, ax2)):

        ac = self.act[:, i]

        ax.scatter(latent1, latent2, activation, c=ac)

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel('Latent_1')
        ax.set_ylabel('Latent_2')
        ax.set_zlabel('Activation')

        _set_pi_ticks((ax.xaxis, ax.yaxis))

    plt.show()
