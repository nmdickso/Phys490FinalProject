# Standard library imports
import argparse
import datetime as dt

# Additional dependancies
import matplotlib.pyplot as plt
import numpy as np
import torch

# Custom libraries
import scinet
import utils as u

num_epochs = 1000
display_epoch = 20
learning_rate = 1e-2
batch_size = 0.01


class DampedPendulumSolver:
    def __init__(self, input_file, learning_rate):
        self.num_epochs = num_epochs
        self.display_epoch = display_epoch
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.training_data = input_file
        self._load_training()

        self._set_hyperparams()

        # initialize scinet
        self.model = scinet.Scinet(self.hyp)
        self.model.float()
        self.losses = []
        return

    def _load_training(self):
        self.w, self.k, self.b, self.t, self.O = u.load_data(self.training_data)

        self.n_obs, self.n_t = self.O.shape[0], self.t.size

        self.train_w, self.test_w, self.train_k, self.test_k, \
        self.train_b, self.test_b, self.train_O, self.test_O, \
        self.train_Q, self.test_Q, self.train_A, \
        self.test_A = u.split_and_format_data(self.w,
                                              self.k,
                                              self.b,
                                              self.O)
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
        params.annealEpoch = 1e8
        self.hyp = params
        return

    def train(self, num_epochs, display_epoch, batch_size):
        print("\nTraining Model...")
        start = dt.datetime.now()

        batch_length = int(np.ceil(batch_size * self.n_obs))
        train_inds = np.array(list(range(self.train_O.shape[0])))

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

            loss = self.model.train(tmp_train_O, tmp_train_Q, tmp_train_A, batch_length)
            self.losses.append(loss)

            epoch = epoch * batch_size
            if (not (epoch) % display_epoch):
                print(f"EPOCH: {epoch:02.0f} of {num_epochs}.\tLOSS: {loss}")
        
        end = dt.datetime.now()
        time_taken = end - start
        print(f"FINAL TRAINING LOSS:\t{self.losses[-1]}")
        print('Time: ',time_taken)

        return self.losses
    
    def test(self):
        print("\nTesting Model...")
        test_losses, activation = self.model.test(self.test_O,
                                                  self.test_Q,
                                                  self.test_A)
        return test_losses, activation

    def visualize(self, outdir='.'):
        # run test
        avgLoss, activation = self.test()
        model_A = self.model.forward(self.test_O, self.test_Q)[-1].detach().numpy().ravel()
        RMSError_all = ((self.test_A - model_A)**2).mean()

        # --------------------------------------------------------------
        # General position comparison for all test
        # --------------------------------------------------------------
        one_one_fig, one_one_ax = plt.subplots()

        one_one_ax.plot([-1, 1], [-1, 1], 'k-')
        test_A = self.test_A.numpy().ravel()
        one_one_ax.plot(test_A, model_A, 'bo', label=f"RMS Error: {RMSError_all:.5f}")
        one_one_ax.set_xlabel("Simulated Position")
        one_one_ax.set_ylabel("SciNet Position")
        one_one_ax.set_aspect('equal')
        one_one_ax.legend()
        one_one_fig.savefig("one_one.png")

        # --------------------------------------------------------------
        # Loss plot
        # --------------------------------------------------------------
        lossfig, lossax = scinet.plot_loss(
            self.losses, filename=f"{outdir}/Loss.png")

        # --------------------------------------------------------------
        # Activation Plot
        # --------------------------------------------------------------
        latentkbfig, latentkbax = scinet.plot_latent(
            self.test_k, self.test_b, activation, filename=f"{outdir}/Activations_kb.png")
        latentkwfig, latentkwax = scinet.plot_latent(
            self.test_k, self.test_w, activation, filename=f"{outdir}/Activations_kw.png")
        latentbwfig, latentbwax = scinet.plot_latent(
            self.test_b, self.test_w, activation, filename=f"{outdir}/Activations_bw.png")
        
        # --------------------------------------------------------------
        # Timeseries comparison
        # --------------------------------------------------------------
        n_test = self.test_O.shape[0]
        test_ind = np.random.randint(0, n_test)
        true_x = self.test_O[test_ind].numpy()
        predicted_x = u.predict_timeseries(self.model, true_x, self.t)

        posfig, posax = plt.subplots()
        posax.plot(self.t, true_x, 'ko', label="Simulated Position")
        posax.plot(self.t, predicted_x, 'bo', label="Predicted Position")
        posax.set_xlabel("Time (s)")
        posax.set_ylabel("Vertical Position (m)")
        posax.set_title(f"RMS Error: {((predicted_x - true_x)**2).mean()}")
        posax.grid()
        posax.legend()
        posfig.savefig("PositonTimeseries.png")

        return


def main(input_file):

    nn = DampedPendulumSolver(input_file, learning_rate)
    nn.train(num_epochs, display_epoch, batch_size)
    nn.visualize()

    plt.show()

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
            DESCRIPTION
        """
    )
    parser.add_argument(
        'input_file',
        type=str,
        help="Input training/testing set (specially formatted)"
    )
    args = parser.parse_args()

    main(args.input_file)
