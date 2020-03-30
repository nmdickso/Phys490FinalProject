import numpy as np
import sys
import torch
import torch.optim as optim
from random import random
from config import Config
from trainer import Trainer
from scinet import Hyperparameters,Scinet
from qubitGeneration import DataGen

def createDataLabel(numQubits,tomComplete):
    if tomComplete:
        label="{}_Qubit_TomComplete_Data".format(numQubits)
    else:
        label="{}_Qubit_TomIncomplete_Data".format(numQubits)
    return label

def getDataArray(path,hyp,shuffle=True):
    print("Loading Data")
    data=np.load(path,allow_pickle=True)
    if shuffle:
        print("Shuffling Data")
        np.random.shuffle(data)

    trainingData=data[0:-hyp.testSize]
    testingData=data[:hyp.testSize]

    return(trainingData,testingData)


class App:
    def __init__(self,hyp,cfg):
        #stores list of mse as a funciton of latent nodes, key is number of qubits
        self.QubitComp={1: [], 2: []}
        self.QubitIncomp={1: [], 2: []}

        #number of projective measurment states for questions and observations based on number of qubits
        self.observationDim={1: 10, 2: 30}
        self.questionDim={1: 10, 2: 30}

        self.hyp=hyp
        self.cfg=cfg
    
    def editHyp(self,obsSize,questionSize,latentNodes):
        self.hyp.encoderNodes=[obsSize,100,100]
        self.hyp.encoderLayers=len(hyp.encoderNodes)
        self.hyp.questionNodes=questionSize
        self.hyp.decoderNodes=[100,100,1]
        self.hyp.decoderLayers=len(hyp.decoderNodes)

        self.hyp.testSize=1000
        self.hyp.trainBatchSize=512
        self.hyp.epochs=50
        self.hyp.annealEpoch=self.hyp.epochs

        self.hyp.learningRate=1e-2
        self.hyp.finalLr=1e-5

        self.hyp.latentNodes =latentNodes

#generates dataset, creates network, trains network, returns MSE on test data
    def getError(self,numQubits,latentNodes,tomComplete=True):
        print(">>Finding Mean Square Error for {} Qubits and {} Latent Nodes".format(numQubits,latentNodes))
        dimHilbert=2*(2**numQubits)-2
        obsSize=self.observationDim[numQubits]
        questionSize=self.questionDim[numQubits]

        dataLabel=createDataLabel(numQubits,tomComplete)

        dataGen=DataGen()

        self.editHyp(obsSize,questionSize,latentNodes)
        path=dataGen.generateDataSet(cfg,dataLabel,dimHilbert,obsSize,questionSize)
        trainingData,testingData=getDataArray(path,self.hyp)

        net=Scinet(hyp)
        if torch.cuda.is_available():
            net.device=torch.device("cuda:0")
        else:
            net.device=torch.device("cpu")
        net.to(net.device)

        net.scheduler=optim.lr_scheduler.CosineAnnealingLR(net.optimizer,self.hyp.epochs,self.hyp.finalLr)

        trainer=Trainer(trainingData,testingData)
        trainer.train(net,hyp)
        error=trainer.getMSE(net)

        return error

    def main(self):
        for numQubits in range(1,2):
            for numLatent in range(0,6):
                error=self.getError(numQubits,numLatent)
                print('error:',error)
                self.QubitComp[numQubits].append(error)

        print(self.QubitComp[1])



if __name__ == "__main__":
    hyp=Hyperparameters()
    cfg=Config()
    app=App(hyp,cfg)
    app.main()