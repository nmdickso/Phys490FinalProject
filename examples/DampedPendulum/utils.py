import numpy as np
import torch


def load_data(input_file):
    t = None
    train_O = []
    train_k = []
    train_b = []
    test_O = []
    test_k = []
    test_b = []
    with open(input_file, 'r') as f:

        # these will point to test once the line is reached
        O = train_O
        k = train_k
        b = train_b

        for ind, line in enumerate(list(f.readlines())):

            if line == "TEST\n":
                O = test_O
                k = test_k
                b = test_b
                continue

            line = line[:-2].split(" ")  # drop the newline
            line = [float(i) for i in line]

            if ind == 0:
                t = np.array(line)
                continue

            k.append(line[0])
            b.append(line[1])
            O.append(np.array(line[2:]))

    n_train = len(train_O)
    n_test = len(test_O)
    n_points = len(t)

    train_O = np.array(train_O)
    test_O = np.array(test_O)
    train_k = np.array(train_k)
    test_k = np.array(test_k)
    train_b = np.array(train_b)
    test_b = np.array(test_b)

    train_Q = np.empty((n_train, 1))
    train_A = np.empty((n_train, 1))
    for ind, obs in enumerate(train_O):
        Q_ind = np.random.randint(0, n_points)
        train_Q[ind, 0] = t[Q_ind]
        train_A[ind, 0] = train_O[ind, Q_ind]

    test_Q = np.empty((n_test, 1))
    test_A = np.empty((n_test, 1))
    for ind, obs in enumerate(test_O):
        Q_ind = np.random.randint(0, n_points)
        test_Q[ind, 0] = t[Q_ind]
        test_A[ind, 0] = test_O[ind, Q_ind]

    train_k = torch.from_numpy(train_k).float()
    test_k = torch.from_numpy(test_k).float()
    train_b = torch.from_numpy(train_b).float()
    test_b = torch.from_numpy(test_b).float()
    train_O = torch.from_numpy(train_O).float()
    test_O = torch.from_numpy(test_O).float()
    train_Q = torch.from_numpy(train_Q).float()
    test_Q = torch.from_numpy(test_Q).float()
    train_A = torch.from_numpy(train_A).float()
    test_A = torch.from_numpy(test_A).float()

    return train_k, test_k, train_b, test_b, train_O, test_O, train_Q, test_Q, \
        train_A, test_A, t


def split_and_format_data(spring_consts, damp_consts, O):

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

    return train_spring, test_spring, train_damp, test_damp, train_O, test_O, \
        train_Q, test_Q, train_A, test_A


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
