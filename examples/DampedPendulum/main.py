# Standard library imports
import argparse
import datetime as dt

# Additional dependancies
import matplotlib.pyplot as plt
import numpy as np
import torch

# Custom libraries
import scinet

num_epochs = 150
display_epoch = 20
learning_rate = 1e-2
batch_size = 0.02


def load_data(input_file):
    t = None
    position = []
    spring_consts = []
    damp_consts = []

    with open(input_file, 'r') as f:
        for ind, line in enumerate(list(f.readlines())):
            line = line[:-2].split(" ")  # drop the newline
            line = [float(i) for i in line]
            if ind == 0:
                t = np.array(line)
            else:
                spring_consts.append(line[0])
                damp_consts.append(line[1])
                position.append(np.array(line[2:]))

    n_data = len(position)
    n_points = len(t)
    X = np.empty((n_data, n_points, 2))

    for ind, tmppos in enumerate(position):
        observation = np.dstack((t, tmppos))[0]
        X[ind, :, :] = observation

    spring_consts = np.array(spring_consts)
    damp_consts = np.array(damp_consts)

    return spring_consts, damp_consts, t, X


def split_and_format_data(spring_consts, damp_consts, O):

    n_observations, n_points, _ = O.shape

    # create questions and answers
    Q_inds = np.random.randint(0, n_points, size=(n_observations,))
    # Q_inds = np.full((n_observations,), -1)
    QA = np.array([O[i, j, :] for i, j in zip(range(n_observations), Q_inds)])
    Q = np.array([[i] for i in QA[:, 0]])
    A = np.array([[i] for i in QA[:, 1]])
    O = O[:, :, 1]

    # now we have the necessary Observations, Questions and Answers
    # for the training. But we need a training and testing set
    all_inds = list(range(n_observations))
    num_training = int(round(0.9 * n_observations))
    train_inds = np.random.choice(all_inds, size=(num_training,), replace=False)
    train_test_bool = np.array([i in train_inds for i in all_inds])

    train_O = torch.from_numpy(O[train_test_bool]).float()
    test_O = torch.from_numpy(O[~train_test_bool]).float()

    train_Q = torch.from_numpy(Q[train_test_bool]).float()
    test_Q = torch.from_numpy(Q[~train_test_bool]).float()

    train_A = torch.from_numpy(A[train_test_bool]).float()
    test_A = torch.from_numpy(A[~train_test_bool]).float()

    train_spring = spring_consts[train_test_bool]
    test_spring = spring_consts[~train_test_bool]

    train_damp = damp_consts[train_test_bool]
    test_damp = damp_consts[~train_test_bool]

    return train_spring, test_spring, train_damp, test_damp, train_O, test_O, train_Q, test_Q, train_A, test_A


def load_hyperparams(n_points):

    # initialize object from package
    params = scinet.Hyperparameters()

    # Set encoder layers
    params.encoderNodes = [n_points, 64, 64]
    params.encoderLayers = len(params.encoderNodes)

    # Set Latent nodes
    params.latentNodes = 3

    # Set decoder layers
    params.decoderNodes = [64, 64, 1]
    params.decoderLayers = len(params.decoderNodes)

    # Set learning rate
    params.learningRate = learning_rate

    params.annealEpoch = 1e8

    return params


def train_SciNet(model, train_O, train_Q, train_A):
    print("\nTraining Model...")
    start = dt.datetime.now()

    n_observations = train_O.shape[0]
    batch_length = int(np.ceil(batch_size * n_observations))
    train_inds = np.array(list(range(n_observations)))

    batched_epochs = int(np.ceil(num_epochs / batch_size))

    losses = []
    for epoch in range(batched_epochs):

        tmp_train_inds = list(
            np.random.choice(
                train_inds,
                size=(batch_length),
                replace=False
            )
        )
        tmp_train_O = train_O[tmp_train_inds, :]
        tmp_train_Q = train_Q[tmp_train_inds, :]
        tmp_train_A = train_A[tmp_train_inds, :]

        loss = model.train(tmp_train_O, tmp_train_Q, tmp_train_A, batch_length)
        losses.append(loss)

        epoch = epoch * batch_size
        if (not (epoch) % display_epoch):
            print(f"EPOCH: {epoch:02.0f} of {num_epochs}.\tLOSS: {loss}")

    end = dt.datetime.now()
    time_taken = end - start
    print(f"FINAL TRAINING LOSS:\t{losses[-1]}")
    print('Time: ',time_taken)

    return losses


def test_SciNet(model, test_O, test_Q, test_A):
    print("\nTesting Model...")

    losses, activation = model.test(test_O, test_Q, test_A)

    return losses, activation


def predict_timeseries(model, x, t):

    n_t = t.size
    t = t[:, np.newaxis]

    x_repeated = np.empty((n_t, x.size))
    for i in range(n_t):
        x_repeated[i, :] = x

    # Conver to tensor object
    test_x = torch.from_numpy(x_repeated).float()
    test_times = torch.from_numpy(t).float()

    # Get the predicted positions
    _, _, _, predicted_x = model(test_x, test_times)

    predicted_x = predicted_x.detach().numpy().ravel()

    return predicted_x


def visualize_results(model, losses, test_spring, test_damp, timevals, test_O, test_Q, test_A, outdir='.'):
    # run test
    avgLoss, activation = test_SciNet(model, test_O, test_Q, test_A)
    model_A = model.forward(test_O, test_Q)[-1].detach().numpy().ravel()
    RMSError_all = ((test_A - model_A)**2).mean()

    # --------------------------------------------------------------
    # General position comparison for all test
    # --------------------------------------------------------------
    one_one_fig, one_one_ax = plt.subplots()

    one_one_ax.plot([-1, 1], [-1, 1], 'k-')
    test_A = test_A.numpy().ravel()
    one_one_ax.plot(test_A, model_A, 'bo', label=f"RMS Error: {RMSError_all:.2f}")
    one_one_ax.set_xlabel("Simulated Position")
    one_one_ax.set_ylabel("SciNet Position")
    one_one_ax.set_aspect('equal')
    one_one_fig.savefig("one_one.png")

    # --------------------------------------------------------------
    # Loss plot
    # --------------------------------------------------------------
    lossfig, lossax = scinet.plot_loss(
        losses, filename=f"{outdir}/Loss.png")

    # --------------------------------------------------------------
    # Activation Plot
    # --------------------------------------------------------------
    latentfig, latentax = scinet.plot_latent(
        test_spring, test_damp, activation, filename=f"{outdir}/Activations.png")
    
    # --------------------------------------------------------------
    # Timeseries comparison
    # --------------------------------------------------------------
    n_test = test_O.shape[0]
    test_ind = np.random.randint(0, n_test)
    true_x = test_O[test_ind].numpy()
    predicted_x = predict_timeseries(model, true_x, timevals)

    posfig, posax = plt.subplots()
    posax.plot(timevals, true_x, label="Simulated Position")
    posax.plot(timevals, predicted_x, label="Predicted Position")
    posax.set_xlabel("Time (s)")
    posax.set_ylabel("Vertical Position (m)")
    posax.set_title(f"RMS Error: {((predicted_x - true_x)**2).mean()}")
    posax.grid()
    posax.legend()
    posfig.savefig("PositonTimeseries.png")
    

    return


def main(input_file):
    # ==============================================================
    # Load the data from the generated file
    # ==============================================================
    spring_consts, damp_consts, timevals, O = load_data(input_file)

    train_spring, test_spring, train_damp, test_damp, train_O, test_O, train_Q, test_Q, train_A, test_A = split_and_format_data(spring_consts, damp_consts, O)
    n_points = train_O.shape[1]

    # ==============================================================
    # Create scinet object
    # ==============================================================

    # initialize parameters
    params = load_hyperparams(n_points)

    # initialize scinet
    model = scinet.Scinet(params)
    model.float()

    # ==============================================================
    # Train SciNet
    # ==============================================================
    losses = train_SciNet(model, train_O, train_Q, train_A)

    # ==============================================================
    # Visualization!
    # ==============================================================
    visualize_results(model, losses, test_spring, test_damp, timevals, test_O, test_Q, test_A)

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
