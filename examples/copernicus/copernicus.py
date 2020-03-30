import scinet
import data_gen

import argparse

import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, MultipleLocator


def _set_pi_ticks(*axis_list):

    for axis in axis_list:

        axis.set_major_formatter(FuncFormatter(
            lambda val, pos:
            '{:.0g}$\pi$'.format(val / np.pi) if val else '0'
        ))
        axis.set_major_locator(MultipleLocator(base=np.pi))


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

    _set_pi_ticks(ax3.yaxis, ax4.yaxis)

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

    def forward(self, x, first_pass=False):

        μ, σ = None, None

        # Pass through encoder layers, if the input is not a latent layer
        if first_pass:

            # Pass through encoder
            μ, σ = self.encode(x)

            # reparamaterize
            x = self.reparameterize(μ, σ)

        # Pass through the time evolution (τ) network
        for _ in range(self.τ_layers):
            x = x + self.τ_bias

        evolved_latent = x

        # Pass through decoder layers (without applying relu on answer neuron)
        answer = self.decode(x, torch.tensor([]))

        return answer, evolved_latent, μ, σ

    def train(self, observations, batch_N):

        avgLoss = 0
        N = observations.shape[0]

        for i in range(0, N, batch_N):

            obs_batch = observations[i:i + batch_N].to(self.device)

            # initial model input is given by first observation
            obs = obs_batch[:, 0, :]

            for te_step in range(observations.shape[1] - 1):
                first_pass = not te_step

                # Target θ given by proceeding observation
                ans = obs_batch[:, te_step + 1, :]

                self.zero_grad()

                # Future model input given by evolved latent layers
                outputs, obs, μ, σ = self(obs, first_pass)

                # Loss function and propogation
                if first_pass:
                    loss = self.lossFunct(μ, σ, outputs, ans)
                else:
                    loss = self.leadingLoss(outputs, ans)

                loss.backward(retain_graph=True)

                self.optimizer.step()

                avgLoss += loss.item() / observations.shape[1]

            self.trainCounter += 1

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
                first_pass = not te_step

                # Target θ given by proceeding observation
                ans = observations[:, te_step + 1, :]

                # future forward function input is given by te'd latent layers
                model_ans, obs, μ, σ = self(obs, first_pass)

                model_θ[:, te_step + 1, :] = model_ans
                model_φ[:, te_step + 1, :] = obs

                # Loss function and propogation
                if first_pass:
                    loss = self.lossFunct(μ, σ, model_ans, ans)
                else:
                    loss = self.leadingLoss(model_ans, ans)

                avgLoss += loss.item()

            avgLoss /= observations.shape[1]

        # The first elem of each row is empty as we don't decode before the RNN
        return avgLoss, model_θ[:, 1:, :], model_φ[:, 1:, :]


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-M', type=int, default=5,
                        help="Number of time-evolution steps")
    parser.add_argument('-N', type=int, default=15000,
                        help="Training dataset size")
    parser.add_argument('--test-N', type=int, default=1000,
                        help="Testing dataset size")
    parser.add_argument('-t', '--del-t', type=int, default=7,
                        help="Number of days per time-evolution step")

    parser.add_argument('-a', type=float, default=0.001,
                        help="Learning rate (α)")
    parser.add_argument('-b', type=int, default=2000,
                        help="Training batch size")
    parser.add_argument('-E', type=int, default=25,
                        help="Number of training epochs")
    parser.add_argument('-B', '--beta-anneal', action='store_true',
                        help="Whether to use 'Beta-Annealing'")

    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--plot-loss', action='store_true',
                        help='Show a plot of loss over training epochs')

    args = parser.parse_args()

    hyp = scinet.Hyperparameters()
    hyp.learningRate = args.a
    hyp.encoderNodes[0] = 2
    hyp.latentNodes = 2
    hyp.decoderNodes[-1] = 2
    hyp.questionNodes = 0
    hyp.leadingLoss = torch.nn.MSELoss
    if args.beta_anneal:
        hyp.annealEpoch = 1e8

    N = args.N
    M = args.M
    del_t = args.del_t
    test_N = args.test_N

    batch_size = args.b
    epochs = tqdm.tqdm(range(args.E), disable=(not args.verbose),
                       bar_format="{l_bar}{bar}| Epoch {n_fmt}/{total_fmt} ["
                                  "{elapsed}<{remaining}, {rate_fmt}{postfix}]")

    model = TimeEvolvedScinet(hyp)

    φ_e, φ_m, θ_e, θ_m = data_gen.generate_orbits(N, M, del_t)

    # data_gen.anim_orbit(φ_e, φ_m, θ_e, θ_m)

    loss = []
    for _ in epochs:

        # Observation are all the elements of θ
        obs = torch.from_numpy(np.concatenate(
            (θ_e[..., np.newaxis], θ_m[..., np.newaxis]), axis=-1
        )).float()

        loss.append(model.train(obs, batch_size))

        if args.verbose:
            epochs.set_description_str(f"loss: {loss[-1]:.2f}")

    if args.plot_loss:
        scinet.plot_loss(loss)

    # Test
    φ_et, φ_mt, θ_et, θ_mt = data_gen.generate_orbits(test_N, M, del_t, True)

    test_θ = torch.from_numpy(np.concatenate(
        (θ_et[..., np.newaxis], θ_mt[..., np.newaxis]), axis=-1
    )).float()

    test_φ = torch.from_numpy(np.concatenate(
        (φ_et[..., np.newaxis], φ_mt[..., np.newaxis]), axis=-1
    )).float()

    loss, out_θ, out_φ = model.test(test_θ)

    # visualize_sample(test_θ, test_φ, out_θ, out_φ)

    plot_kwargs = {"method": 'surface', "axlabels": ['φ_e', 'φ_m'],
                   "axis_formatter": _set_pi_ticks}

    space = np.linspace(0, 2 * np.pi, num=50)
    mesh_e, mesh_m = np.meshgrid(space, space)

    scinet.plot_latent(mesh_e, mesh_m, out_φ[:, -1, :], **plot_kwargs)

    plt.show()