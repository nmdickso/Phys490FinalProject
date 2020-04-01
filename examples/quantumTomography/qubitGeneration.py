import numpy as np
import os
from random import uniform
from config import Config


def measureState(psi, phi):
    return abs(np.inner(psi, phi))**2


def generateRandomQubit(dim):
    qubit = np.zeros(dim, dtype=np.complex)
    singular = True
    while singular:
        for i in range(0, dim):
            qubit[i] = uniform(0, 1) + uniform(0, 1) * 1j
        norm = np.linalg.norm(qubit)
        singular = norm < 0.0000001

    return qubit / norm


def generateRandomSet(dim, numStates):
    states = []
    for i in range(0, numStates):
        states.append(generateRandomQubit(dim))
    states = np.array(states)
    return(states)


def parameterizeQuestion(omega, vecSet):
    omega = [measureState(omega, vec) for vec in vecSet]
    return omega


class DataGen:
    def __init__(self):
        self.psis = []
        self.basis = []
        self.observations = []
        self.questions = []
        self.answers = []

        # used to parameterize omega
        self.omegaBasis = []

    def generateDataSet(self, cfg, dataSetLen, fileName,
                        dimHilbert, obsSize, questionSize):
        print("\n>>Generating New Dataset")

        # generates psi
        for i in range(0, dataSetLen):
            phi = generateRandomQubit(dimHilbert)
            self.psis.append(phi)

        # generates phis (observationn basis)
        self.basis = generateRandomSet(dimHilbert, obsSize)

        # generates observations
        for psi in self.psis:
            obseravtion = []
            for phi in self.basis:
                measurement = measureState(psi, phi)
                obseravtion.append(measurement)
            obseravtion = np.array(obseravtion)
            self.observations.append(obseravtion)

        # generates omegas
        for i in range(0, dataSetLen):
            # generates omegas
            self.questions.append(generateRandomQubit(dimHilbert))

        # generates answers
        for i in range(0, dataSetLen):
            measurement = measureState(self.psis[i], self.questions[i])
            self.answers.append(np.array(measurement))
        self.answers = np.array(self.answers)

        # maps omegas onto parameterized omegas (done after finding soltuions to
        #   simplify math)
        self.omegaBasis = generateRandomSet(dimHilbert, questionSize)
        for i, omega in enumerate(self.questions):
            self.questions[i] = parameterizeQuestion(omega, self.omegaBasis)
        self.questions = np.array(self.questions, dtype=float)

        path = self.writeToFolder(cfg, dataSetLen, fileName)
        return(path)

    def writeToFolder(self, cfg, dataSetLen, label):
        sets = []
        for i in range(0, dataSetLen):
            dataSet = np.array(
                (self.observations[i], self.questions[i], self.answers[i])
            )
            sets.append(dataSet)
        np.array(sets)

        # makes results folder if it does not exist
        if not os.path.exists(cfg.dataPath[0:-1]):
            os.mkdir(cfg.dataPath[0:-1])

        i = 1
        while True:
            path = cfg.dataPath + label + '_' + str(i)
            i += 1
            try:
                open(path + ".npy")
            except:
                np.save(path, sets)
                break

            if i == 100:
                print("Unable to Save Data")
                break

        print("Data Written to {}".format(path))
        return(path + '.npy')


if __name__ == "__main__":
    gen = DataGen()
    cfg = Config()
    gen.generateDataSet(cfg)
