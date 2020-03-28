import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss(losses, filename=None):

    fig, ax = plt.subplots()

    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    ax.plot(range(len(losses)), losses, color='blue')

    if filename:
        fig.savefig(filename)

    return fig, ax


def plot_latent(param1, param2, activation, filename=None,
                method='scatter', axlabels=[], axis_formatter=None):

    fig_height = 4
    fig_width = 4.5
    num_activations = activation.shape[-1]

    fig = plt.figure(figsize=(num_activations * fig_width, fig_height))

    for i in range(num_activations):
        ax = fig.add_subplot(1, num_activations, i + 1, projection='3d')

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel(axlabels[0:1] or 'Parameter 1')
        ax.set_ylabel(axlabels[1:2] or 'Parameter 2')
        ax.set_zlabel(axlabels[2:3] or 'Activation')

        if axis_formatter:
            axis_formatter(ax.xaxis, ax.yaxis)

        act = activation[:, i]

        if method == 'scatter':
            ax.scatter(param1, param2, act, c=act)

        elif method == 'surface':
            ax.plot_trisurf(param1, param2, act, cmap='magma')

    if filename:
        fig.savefig(filename)

    return fig, fig.axes
