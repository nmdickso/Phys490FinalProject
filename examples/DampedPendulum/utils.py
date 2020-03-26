import numpy as np


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
