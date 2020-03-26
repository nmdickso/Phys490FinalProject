import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss(losses, filename=None):

    fig, ax = plt.subplots()

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.plot(range(len(losses)), losses, color='blue')

    if filename:
        plt.savefig(filename)
    
    return fig, ax


def plot_latent(param1, param2, activation, filename=None):

    fig_height = 4
    fig_width = 4.5
    num_activations = activation.shape[-1]
    
    fig = plt.figure(figsize=(num_activations * fig_width, fig_height))

    for i in range(num_activations):
        ax = fig.add_subplot(1, num_activations, i + 1, projection='3d')

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel('Parameter 1')
        ax.set_ylabel('Parameter 2')
        ax.set_zlabel('Activation')

        act = activation[:, i]
        ax.scatter(param1, param2, act, c=act)

    if filename:
        plt.savefig(filename)
    
    return fig, ax
