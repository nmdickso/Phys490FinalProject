import numpy as np
import sys
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
        hyp.encoderNodes=[6,100,100]
        hyp.encoderLayers=len(hyp.encoderNodes)
        hyp.questionNodes=hyp.encoderNodes[0]
        hyp.decoderNodes=[100,100,1]
        hyp.decoderLayers=len(hyp.decoderNodes)

        hyp.testSize=1000
        hyp.trainBatchSize=10
        hyp.testBatchSize=10
        hyp.epochs=5


if __name__ == "__main__":
    
    cfg=Config()

    hyp=Hyperparameters()
    editHyp(hyp)
    print(hyp.encoderNodes)

    trainingData,testingData=getDataArray(cfg.dataPath+"1.npy",hyp)
    net=Scinet(hyp)
    trainer=Trainer(trainingData,testingData)
    

    trainer.trainAndTest(net,hyp)

    