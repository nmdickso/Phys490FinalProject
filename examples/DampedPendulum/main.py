# Standard library imports
import argparse

# Additional dependancies
import matplotlib.pyplot as plt
import numpy as np
import torch

# Custom libraries
import scinet

num_epochs = 100
display_epoch = 100
learning_rate = 1e-3
batch_size = 0.05


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

    return spring_consts, damp_consts, X


def split_and_format_data(O):

    n_observations, n_points, _ = O.shape

    # create questions and answers
    Q_inds = np.random.randint(0, n_points, size=(n_observations,))
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

    return train_O, test_O, train_Q, test_Q, train_A, test_A


def load_hyperparams(n_points):

    # initialize object from package
    params = scinet.Hyperparameters()

    # Set encoder layers
    params.encoderNodes = [n_points, 300, 300]
    params.encoderLayers = len(params.encoderNodes)

    # Set Latent nodes
    params.latentNodes = 3

    # Set decoder layers
    params.decoderNodes = [300, 300, 1]
    params.decoderLayers = len(params.decoderNodes)

    # Set learning rate
    params.learningRate = learning_rate

    return params


def train_SciNet(model, train_O, train_Q, train_A):
    print("\nTraining Model...")

    n_observations = train_O.shape[0]
    batch_length = int(np.ceil(batch_size * n_observations))
    train_inds = np.array(list(range(n_observations)))

    losses = []
    for epoch in range(num_epochs):

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

        if (not (epoch) % display_epoch):
            print(f"EPOCH: {epoch:02d} of {num_epochs}.\tLOSS: {loss}")

    return losses


def test_SciNet(model, test_O, test_Q, test_A):
    print("\nTesting Model...")

    losses, activation = model.test(test_O, test_Q, test_A)

    return losses, activation


def main(input_file):
    # ==============================================================
    # Load the data from the generated file
    # ==============================================================
    spring_consts, damp_consts, O = load_data(input_file)

    train_O, test_O, train_Q, test_Q, train_A, test_A = split_and_format_data(O)
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
    # Test time!
    # ==============================================================
    model_A = model.forward(test_O, test_Q)[-1].detach().numpy().ravel()
    testy_A = test_A.numpy().ravel()

    fig, ax = plt.subplots()

    ax.set_aspect('equal')
    ax.plot([-1, 1], [-1, 1], 'k-')
    ax.plot(testy_A, model_A, 'bo')
    ax.set_xlabel("Simulated Position")
    ax.set_ylabel("SciNet Position")

    # plt.show()

    avgLoss, activation = test_SciNet(model, test_O, test_Q, test_A)

    # ==============================================================
    # Visualization!
    # ==============================================================

    plot_loss = scinet.plot_loss(num_epochs, losses) 
    plot_latent = scinet.plot_latent(param1, param2, activation)

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
