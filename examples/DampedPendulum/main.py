# Standard library imports
import argparse
import json
import datetime as dt

# Additional dependancies
import matplotlib.pyplot as plt
import numpy as np

# Custom libraries
import scinet
import utils as u

DEFAULT_PARAMS = "./params/params.json"


class DampedPendulumSolver:
    '''Main container for damped pendulum examples

    Parameters
    ----------
    input_file : str
        Path to the input data file

    learning_rate : float
        Constant learning rate to be applied throughout network training
    '''

    def __init__(self, input_file, learning_rate):
        self.learning_rate = learning_rate
        self.training_data = input_file
        self._load_training()

        self._set_hyperparams()

        # initialize scinet
        self.model = scinet.Scinet(self.hyp)
        self.model.float()
        self.losses = []
        return

    def _load_training(self):
        '''load training data from input file `training_data`'''

        # self.k, self.b, self.t, self.O = u.load_data(self.training_data)
        # self.n_obs, self.n_t = self.O.shape[0], self.t.size

        self.train_k, self.test_k, self.train_b, self.test_b, \
            self.train_O, self.test_O, self.train_Q, self.test_Q, \
            self.train_A, self.test_A, self.t = u.load_data(self.training_data)

        self.n_train = self.train_O.shape[0]
        self.n_t = self.t.size

        return

    def _set_hyperparams(self):

        # initialize object from package
        params = scinet.Hyperparameters()
        # Set encoder layers
        params.encoderNodes = [self.n_t, 100, 100]
        params.encoderLayers = len(params.encoderNodes)
        # Set Latent nodes
        params.latentNodes = 3
        # Set decoder layers
        params.decoderNodes = [100, 100, 1]
        params.decoderLayers = len(params.decoderNodes)
        # Set learning rate
        params.learningRate = self.learning_rate
        params.annealEpoch = 1e30
        self.hyp = params
        return

    def train(self, num_epochs, display_epoch, batch_size):
        '''Network training loop

        Batched training of the network over `num_epochs`

        Parameters
        ----------
        num_epochs : int
            Number of training epochs

        display_epoch : int
            Multiple of epochs at which to write out the current loss values

        batch_size : int
            Batch size

        Returns
        -------
        losses : listof float
            List of cost function values over all training. Also saved to
            `DampedPendulumSolver.losses`
        '''

        print("\nTraining Model...")
        start = dt.datetime.now()

        batch_length = int(np.ceil(batch_size * self.n_train))
        train_inds = np.array(list(range(self.n_train)))

        batched_epochs = int(np.ceil(num_epochs / batch_size))

        for epoch in range(batched_epochs):

            tmp_train_inds = list(
                np.random.choice(
                    train_inds,
                    size=(batch_length),
                    replace=False
                )
            )
            tmp_train_O = self.train_O[tmp_train_inds, :]
            tmp_train_Q = self.train_Q[tmp_train_inds, :]
            tmp_train_A = self.train_A[tmp_train_inds, :]

            loss = self.model.train(tmp_train_O, tmp_train_Q, tmp_train_A,
                                    batch_length)
            self.losses.append(loss)

            epoch = epoch * batch_size
            if (not (epoch) % display_epoch):
                print(f"EPOCH: {epoch:02.0f} of {num_epochs}.\tLOSS: {loss}")

        end = dt.datetime.now()
        time_taken = end - start
        print(f"FINAL TRAINING LOSS:\t{self.losses[-1]}")
        print('Time: ', time_taken)

        return self.losses

    def test(self):
        '''Return losses and latent activations from network testing function'''
        print("\nTesting Model...")
        test_losses, activation = self.model.test(self.test_O,
                                                  self.test_Q,
                                                  self.test_A)
        return test_losses, activation

    def visualize(self, outdir='.'):
        '''Final network results plots

        Plots of position comparisons, losses, latent activations and time
        series comparisons given by the testing of the trained network.

        Parameters
        ----------
        outdir : str, optional
            Directory for saving of all plots
        '''
        # run test
        avgLoss, activation = self.test()
        model_A = self.model.forward(self.test_O, self.test_Q)[-1].detach().numpy().ravel()
        RMSError_all = np.sqrt(((self.test_A - model_A)**2).mean()) * 100

        # ------------------------------------------------------------------
        # General position comparison for all test
        # ------------------------------------------------------------------

        one_one_fig, one_one_ax = plt.subplots()

        one_one_ax.plot([-1, 1], [-1, 1], 'k-')
        test_A = self.test_A.numpy().ravel()
        one_one_ax.plot(test_A, model_A, 'bo',
                        label=f"RMS Error: {RMSError_all:.5f}%")
        one_one_ax.set_xlabel("Simulated Position")
        one_one_ax.set_ylabel("SciNet Position")
        one_one_ax.set_aspect('equal')
        one_one_ax.legend()
        one_one_fig.savefig(f"{outdir}/one_one.png")

        # ------------------------------------------------------------------
        # Loss plot
        # ------------------------------------------------------------------

        lossfig, lossax = scinet.plot_loss(
            self.losses, filename=f"{outdir}/Loss.png")

        # ------------------------------------------------------------------
        # Activation Plot
        # ------------------------------------------------------------------

        test_k_len = np.unique(self.test_k.numpy()).size
        test_b_len = np.unique(self.test_b.numpy()).size
        test_k = self.test_k.numpy().reshape(test_k_len, test_b_len)
        test_b = self.test_b.numpy().reshape(test_k_len, test_b_len)

        latentfig, latentax = scinet.plot_latent(
            test_k, test_b, activation.numpy(), method='surface',
            filename=f"{outdir}/Activations.png", axlabels=["k", "b"]
        )

        # ------------------------------------------------------------------
        # Timeseries comparison
        # ------------------------------------------------------------------

        n_test = self.test_O.shape[0]
        test_ind = np.random.randint(0, n_test)
        true_x = self.test_O[test_ind].numpy()
        predicted_x = u.predict_timeseries(self.model, true_x, self.t)

        posfig, posax = plt.subplots()
        posax.plot(self.t, true_x, 'ko', label="Simulated Position")
        posax.plot(self.t, predicted_x, 'bo', label="Predicted Position")
        posax.set_xlabel("Time (s)")
        posax.set_ylabel("Vertical Position (m)")
        posax.set_title(f"RMS Error: {np.sqrt(((predicted_x - true_x)**2).mean())}")
        posax.grid()
        posax.legend()
        posfig.savefig(f"{outdir}/PositonTimeseries.png")

        return


def main(input_file, params=DEFAULT_PARAMS, outdir='.'):
    '''Main example driver'''

    params = json.load(open(params, 'r'))
    nn = DampedPendulumSolver(input_file, params['learning_rate'])
    nn.train(params['num_epochs'], params['display_epoch'], params['batch_size'])
    nn.visualize(outdir=outdir)

    plt.show()

    return


if __name__ == "__main__":

    # ----------------------------------------------------------------------
    # Parse command line arguments
    # ----------------------------------------------------------------------

    parser = argparse.ArgumentParser(
        description="""
            Program to Apply SciNet to a series of Damped Pendulum Data
            And monitor what it learns
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Input training/testing set (specially formatted)"
    )
    parser.add_argument(
        '--params',
        type=str,
        help="Path to input parameters file",
        required=False,
        default=DEFAULT_PARAMS
    )
    parser.add_argument(
        '--outdir',
        type=str,
        help="Path to save output plots",
        required=False,
        default='.'
    )
    args = parser.parse_args()

    # ----------------------------------------------------------------------
    # Execute main stack
    # ----------------------------------------------------------------------

    main(args.input_file, params=args.params, outdir=args.outdir)
