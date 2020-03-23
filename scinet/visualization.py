import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss(losses, filename=None):

    fig, ax = plt.subplots()

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_aspect('equal')

    ax.plot(range(len(losses)), losses, color='blue')

    if filename:
        plt.savefig(filename)

    plt.show()


def plot_latent(latent1, latent2, activation, filename=None):

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for i, ax in enumerate((ax1, ax2)):

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel('Latent_1')
        ax.set_ylabel('Latent_2')
        ax.set_zlabel('Activation')

        act = activation[:, i]

        ax.scatter(latent1, latent2, act, c=act)

    if filename:
        plt.savefig(filename)

    plt.show()
