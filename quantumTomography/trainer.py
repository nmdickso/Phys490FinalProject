import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self,trainingData,testingData):
        self.trainingLoss=[]

        self.testingLoss=[]
        print(trainingData[1][0])
        self.trainingObs=torch.from_numpy(np.array([row[0] for row in trainingData])).float()
        self.trainQuestions=torch.from_numpy(np.array([row[1] for row in trainingData])).float()
        self.trainingAns=torch.from_numpy(np.array([[row[2]] for row in trainingData])).float()

        self.testingObs=torch.from_numpy(np.array([row[0] for row in testingData])).float()
        self.testQuestions=torch.from_numpy(np.array([row[1] for row in testingData])).float()
        self.testingAns=torch.from_numpy(np.array([[row[2]] for row in testingData])).float()
        print(self.trainingObs[1])

    def trainAndTest(self,neuralNet,hyp):
        for i in range(0,hyp.epochs):
            

            #records weights to see if they change after training
            weights1=[]
            for param in neuralNet.parameters():
                weights1.append(param.clone())
                
            print("Training Epoch {}:".format(i+1))
            trainLoss=neuralNet.train(self.trainingObs,self.trainQuestions,self.trainingAns,hyp.trainBatchSize)
            self.trainingLoss.append(trainLoss)
            

            #prints if weights are unchanged
            weights2=[]
            unchangedWeights=0
            for param in neuralNet.parameters():
                weights2.append(param.clone())
            for j in zip(weights1, weights2):
                if torch.equal(j[0], j[1]):
                    unchangedWeights+=1
        if unchangedWeights>0:
            print(unchangedWeights,"of",len(weights1),"Weights Unchanged")
            print("Testing Epoch {}:".format(i+1))
            testLoss=neuralNet.test(self.testingObs,self.testQuestions,self.testingAns,hyp.testBatchSize)
            self.testingLoss.append(testLoss)
