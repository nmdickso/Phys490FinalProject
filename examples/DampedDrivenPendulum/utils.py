import numpy as np
import torch


def load_data(input_file):
    t = None
    position = []
    spring_consts = []
    damp_consts = []
    driving_freqs = []

    with open(input_file, 'r') as f:
        for ind, line in enumerate(list(f.readlines())):
            line = line[:-2].split(" ")  # drop the newline
            line = [float(i) for i in line]
            if ind == 0:
                t = np.array(line)
            else:
                driving_freqs.append(line[0])
                spring_consts.append(line[1])
                damp_consts.append(line[2])
                position.append(np.array(line[3:]))

    n_data = len(position)
    n_points = len(t)
    X = np.empty((n_data, n_points, 2))

    for ind, tmppos in enumerate(position):
        observation = np.dstack((t, tmppos))[0]
        X[ind, :, :] = observation

    driving_freqs = np.array(driving_freqs)
    spring_consts = np.array(spring_consts)
    damp_consts = np.array(damp_consts)

    return driving_freqs, spring_consts, damp_consts, t, X


def split_and_format_data(driving_freqs, spring_consts, damp_consts, O):

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

    train_spring = spring_consts[train_test_bool]
    test_spring = spring_consts[~train_test_bool]

    train_damp = damp_consts[train_test_bool]
    test_damp = damp_consts[~train_test_bool]

    train_freqs = driving_freqs[train_test_bool]
    test_freqs = driving_freqs[~train_test_bool]

    return train_freqs, test_freqs, train_spring, test_spring, train_damp, \
        test_damp, train_O, test_O, train_Q, test_Q, train_A, test_A


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