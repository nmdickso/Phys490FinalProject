import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as funct


class Trainer:
    def __init__(self, trainingData, testingData):
        self.trainingLoss = []

        self.testingLoss = []

        self.trainingObs = torch.from_numpy(np.array(
            [row[0] for row in trainingData]
        )).float()
        self.trainQuestions = torch.from_numpy(np.array(
            [row[1] for row in trainingData]
        )).float()
        self.trainingAns = torch.from_numpy(np.array(
            [[row[2]] for row in trainingData]
        )).float()

        self.testingObs = torch.from_numpy(np.array(
            [row[0] for row in testingData]
        )).float()
        self.testQuestions = torch.from_numpy(np.array(
            [row[1] for row in testingData]
        )).float()
        self.testingAns = torch.from_numpy(np.array(
            [[row[2]] for row in testingData]
        )).float()

    def train(self, neuralNet, hyp):
        for i in tqdm(range(0, hyp.epochs)):
            trainLoss = neuralNet.train(self.trainingObs, self.trainQuestions,
                                        self.trainingAns, hyp.trainBatchSize)
            self.trainingLoss.append(trainLoss)
            neuralNet.scheduler.step()

    def getMSE(self, net):
        with torch.no_grad():
            observations = self.testingObs.to(net.device)
            answers = self.testingAns.to(net.device)
            questions = self.testQuestions.to(net.device)

            mu, sig, latent_activation, outputs = net(observations, questions)

            reconMSE = funct.mse_loss(outputs, answers).item()

        return reconMSE
