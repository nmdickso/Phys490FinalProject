import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_loss(losses, filename=None):
    '''Plot of network training losses over the epochs'''

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
    '''Plot of latent neuron activations

    Three-dimensional plots showcasing the activation of the latent neurons
    of a trained network with respect to some physical variables, in order
    to examine the relation of these neurons with certain physical concepts

    The number of plots in the figure is determined automatically based on the
    dimensions of `activation`.

    Parameters
    ----------
    param1 : numpy.ndarray
        x-axis array of physical parameter values

    param2 : numpy.ndarray
        y-axis array of physical parameter values

    activation : numpy.ndarray
        z-axis array of latent neuron activations. Second dimension size
        determines the number of plots to create

    filename : str, optional
        If given, will save a copy of the figure to `filename`

    method : {'scatter', 'surface'}, optional
        Whether to create scatter plots or surface plots. The 'surface' option
        will require more careful calibration of input data.

    axlabels : listof (str str str), optional
        Labels of parameters and activation for plots. In order of
        (param1, param2, activation)

    axis_formatter : callable
        Function which is applied on the x and y axis of each plot, meant for
        formatting features such as labels or ticks

    Returns
    -------
    fig : matplotlib.figure.Figure
        Overall plot figure

    axes : listof matplotlib.axes._subplots.AxesSubplot
        List of individual plot `axes`, identical to `fig.axes`

    '''

    fig_height = 4
    fig_width = 4.5
    num_activations = activation.shape[-1]

    fig = plt.figure(figsize=(num_activations * fig_width, fig_height))

    for i in range(num_activations):
        ax = fig.add_subplot(1, num_activations, i + 1, projection='3d')

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel((axlabels[0:1] or ['Parameter 1'])[0])
        ax.set_ylabel((axlabels[1:2] or ['Parameter 2'])[0])
        ax.set_zlabel((axlabels[2:3] or ['Activation'])[0])

        if axis_formatter:
            axis_formatter(ax.xaxis, ax.yaxis)

        act = activation[:, i]

        if method == 'scatter':
            ax.scatter(param1, param2, act, c=act, cmap='inferno')

        elif method == 'surface':
            act = act.reshape(param1.shape)

            ax.plot_surface(param1, param2, act, cmap='inferno')

    if filename:
        fig.savefig(filename)

    return fig, fig.axes
