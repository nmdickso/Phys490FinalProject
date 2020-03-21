import scinet
import data_gen

import tqdm
import torch
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, MultipleLocator


def _set_pi_ticks(axis_list):

    for axis in axis_list:

        axis.set_major_formatter(FuncFormatter(
            lambda val, pos:
            '{:.0g}$\pi$'.format(val / np.pi) if val else '0'
        ))
        axis.set_major_locator(MultipleLocator(base=np.pi))


def visualize_latent(φ_e, φ_m, activation):
    '''activation M=2'''

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    for i, ax in enumerate((ax1, ax2)):
        act = activation[:, i]

        ax.scatter(φ_e, φ_m, act, c=act)

        ax.set_title(f'Latent neuron #{i+1}')
        ax.set_xlabel('φ_e')
        ax.set_ylabel('φ_m')
        ax.set_zlabel('activation')

        _set_pi_ticks((ax.xaxis, ax.yaxis))

    plt.show()


def visualize_sample(obs_θ, obs_φ, model_θ, model_φ):

    import matplotlib.animation as anim

    def _update_plot(n):

        lines[0].set_data(obs_θ[:n, 0], R[:n])
        lines[1].set_data(model_θ[:n, 0], R[:n])

        lines[2].set_data(obs_θ[:n, 1], R[:n])
        lines[3].set_data(model_θ[:n, 1], R[:n])

        lines[4].set_data(x[:n], obs_θ[:n, 0])
        lines[5].set_data(x[:n], model_θ[:n, 0])
        lines[6].set_data(x[:n], diff_θ[:n, 0])

        lines[7].set_data(x[:n], obs_θ[:n, 1])
        lines[8].set_data(x[:n], model_θ[:n, 1])
        lines[9].set_data(x[:n], diff_θ[:n, 1])

        return lines

    # Select a random orbit to visualize
    # get rid of first obs point, which can't be compared with model
    rand_ind = np.random.randint(obs_θ.shape[0])

    obs_θ = obs_θ[rand_ind, 1:]
    obs_φ = obs_φ[rand_ind, 1:]
    model_θ = model_θ[rand_ind, :]
    model_φ = model_φ[rand_ind, :]

    diff_θ = obs_θ - model_θ

    R = np.ones_like(obs_θ[:, 0])
    x = range(R.size)

    # Plot
    fig = plt.figure()
    fig.suptitle(f'Sample orbit {rand_ind}')

    ax1, ax2 = fig.add_subplot(221, polar=1), fig.add_subplot(222, polar=1)
    ax3, ax4 = fig.add_subplot(223), fig.add_subplot(224)

    ax1.set_title('Sun Angle (θ_e)')
    ax2.set_title('Mars Angle (θ_m)')

    ax1.plot(0, 0, 'go')
    ax2.plot(0, 0, 'go')

    [line1_o] = ax1.plot(obs_θ[:, 0], R, '-go')  # , markevery=[-1])
    [line1_m] = ax1.plot(model_θ[:, 0], R, '-ro')  # , markevery=[-1])

    [line2_o] = ax2.plot(obs_θ[:, 1], R, '-go')  # , markevery=[-1])
    [line2_m] = ax2.plot(model_θ[:, 1], R, '-ro')  # , markevery=[-1])

    [line3_o] = ax3.plot(x, obs_θ[:, 0], '-go')
    [line3_m] = ax3.plot(x, model_θ[:, 0], '-ro')
    [line3_d] = ax3.plot(x, diff_θ[:, 0], '-bo')

    [line4_o] = ax4.plot(x, obs_θ[:, 1], '-go')
    [line4_m] = ax4.plot(x, model_θ[:, 1], '-ro')
    [line4_d] = ax4.plot(x, diff_θ[:, 1], '-bo')

    lines = [line1_o, line1_m, line2_o, line2_m,
             line3_o, line3_m, line3_d, line4_o, line4_m, line4_d]

    fig.legend(lines, ['Observations', 'Model'])

    _set_pi_ticks((ax3.yaxis, ax4.yaxis))

    anim.FuncAnimation(fig, _update_plot, model_θ.shape[0], blit=True)

    plt.show()


class TimeEvolvedScinet(scinet.Scinet):

    def __init__(self, hyp):
        super().__init__(hyp)

        N = hyp.latentNodes

        self.τ_bias = torch.nn.Parameter(torch.zeros(N, requires_grad=True))
        self.τ_layers = 3

        # Optimizer and loss functions
        self.optimizer = hyp.optimizer(self.parameters(), hyp.learningRate)
        self.lossFunct = hyp.lossFunct()

    def forward(self, x, first_pass=False):

        # Pass through encoder layers, if the input is not a latent layer
        if first_pass:

            # Pass through encoder
            for layer in self.encoder:
                x = torch.tanh(layer(x))

            # Pass through latent layer
            x = torch.tanh(self.latent(x))

        # Pass through the time evolution (τ) network
        for _ in range(self.τ_layers):
            x = x + self.τ_bias

        evolved_latent = x

        # Pass through decoder layers (without applying relu on answer neuron)
        for layer in self.decoder[:-1]:
            x = torch.tanh(layer(x))

        # Answer neurons
        answer = self.decoder[-1](x)

        return answer, evolved_latent

    def train(self, observations, batch_N):

        avgLoss = 0
        N = observations.shape[0]

        for i in range(0, N, batch_N):

            obs_batch = observations[i:i + batch_N].to(self.device)

            # initial model input is given by first observation
            obs = obs_batch[:, 0, :]

            for te_step in range(observations.shape[1] - 1):

                # Target θ given by proceeding observation
                ans = obs_batch[:, te_step + 1, :]

                self.zero_grad()

                # Future model input given by evolved latent layers
                outputs, obs = self(obs, not te_step)

                # Loss function and propogation
                loss = self.lossFunct(outputs, ans)
                loss.backward(retain_graph=True)

                self.optimizer.step()

                avgLoss += loss.item()

        # TODO figure out best returned loss calc for new RNN
        avgLoss /= (N // batch_N)

        return avgLoss

    def test(self, observations):

        avgLoss = 0

        observations = observations.to(self.device)
        model_θ = np.empty_like(observations)
        model_φ = np.empty_like(observations)

        with torch.no_grad():

            # initial forward function input is given by observation
            obs = observations[:, 0, :]

            for te_step in range(observations.shape[1] - 1):

                # Target θ given by proceeding observation
                ans = observations[:, te_step + 1, :]

                # future forward function input is given by te'd latent layers
                model_ans, obs = self(obs, not te_step)

                model_θ[:, te_step + 1, :] = model_ans
                model_φ[:, te_step + 1, :] = obs

                loss = self.lossFunct(model_ans, ans)

                avgLoss += loss.item()

            avgLoss /= observations.shape[1]

        # f
        return avgLoss, model_θ[:, 1:, :], model_φ[:, 1:, :]


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

        loss = model.train(obs, torch.Tensor([]), ans, 1000)

    visualize_latent(φ_e[:, 0], φ_m[:, 0], model.latent_out)
