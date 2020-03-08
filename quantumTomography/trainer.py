import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self,trainingData,testingData):
        self.trainingLoss=[]

        self.testingLoss=[]

        self.trainingObs=torch.from_numpy(np.array([row[0] for row in trainingData])).float()
        self.trainQuestions=torch.from_numpy(np.array([row[1] for row in trainingData])).float()
        self.trainingAns=torch.from_numpy(np.array([[row[2]] for row in trainingData])).float()

        self.testingObs=torch.from_numpy(np.array([row[0] for row in testingData])).float()
        self.testQuestions=torch.from_numpy(np.array([row[1] for row in testingData])).float()
        self.testingAns=torch.from_numpy(np.array([[row[2]] for row in testingData])).float()


    def trainAndTest(self,neuralNet,hyp):
        for i in range(0,hyp.epochs):
            print("Training Epoch {}:".format(i+1))
            trainLoss=neuralNet.train(self.trainingObs,self.trainQuestions,self.trainingAns,hyp.trainBatchSize)
            self.trainingLoss.append(trainLoss)
            
            print("Testing Epoch {}:".format(i+1))
            testLoss=neuralNet.test(self.testingObs,self.testQuestions,self.testingAns,hyp.testBatchSize)
            self.testingLoss.append(testLoss)
