# Standard library imports
import argparse

# Additional dependancied
import numpy as np
import torch

# Custom libraries
import scinet

num_epochs = 100
display_epoch = 5


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


def main(input_file):
    # ==============================================================
    # Load the data from the generated file
    # ==============================================================

    # Load data
    spring_consts, damp_consts, O = load_data(input_file)
    n_observations = len(spring_consts)
    n_points = len(O[0])

    # create questions and answers
    questions_inds = np.random.randint(0, n_points, size=(n_observations,))
    QA = np.array([O[i, j, :] for i, j in zip(range(n_observations), questions_inds)])
    Q = QA[:, 0]
    A = QA[:, 1]
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

    # ==============================================================
    # Load Hyper Parameters object to initialise the network
    # ==============================================================

    # Initialize object
    params = scinet.Hyperparameters()
    params.latentNodes = 3
    params.encoderNodes = [n_points, 100, 100]
    params.encoderLayers = len(params.encoderNodes)

    params.decoderNodes = [100, 100, 1]
    params.decoderLayers = len(params.decoderNodes)

    # ==============================================================
    # Create scinet object
    # ==============================================================

    # initialize object
    model = scinet.Scinet(params)
    model.float()

    # begin training
    print("\nTraining Model...")

    losses = []
    for epoch in range(num_epochs):
        tmploss = model.train(train_O, train_Q, train_A, 5)
        losses.append(tmploss)

        if (not (epoch) % display_epoch):
            print(f"EPOCH: {epoch:02d} of {num_epochs}.\tLOSS: {tmploss}")

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