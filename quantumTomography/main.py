import numpy as np
import sys
import torch
from random import random
from config import Config
from trainer import Trainer
from scinet import Hyperparameters,Scinet




def getDataArray(path,hyp,shuffle=True):
    print("Loading Data")
    data=np.load(path,allow_pickle=True)
    if shuffle:
        print("Shuffling Data")
        np.random.shuffle(data)

    trainingData=data[0:-hyp.testSize]
    testingData=data[:hyp.testSize]

    return(trainingData,testingData)


def editHyp(hyp):
        hyp.encoderNodes=[10,100,100]
        hyp.encoderLayers=len(hyp.encoderNodes)
        hyp.questionNodes=hyp.encoderNodes[0]
        hyp.decoderNodes=[100,100,1]
        hyp.decoderLayers=len(hyp.decoderNodes)

        hyp.testSize=1
        hyp.trainBatchSize=512
        hyp.epochs=100

        hyp.latentNodes =0


if __name__ == "__main__":
    
    cfg=Config()

    hyp=Hyperparameters()
    editHyp(hyp)
    path="quantumTomography\data\dataset_Complete_1.npy"
    trainingData,testingData=getDataArray(path,hyp)
    net=Scinet(hyp)
    net.device=torch.device("cuda:0")
    net.to(net.device)


    trainer=Trainer(trainingData,testingData)
    

    trainer.trainAndTest(net,hyp)

    