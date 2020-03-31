import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from random import random
from config import Config
from trainer import Trainer
from scinet import Hyperparameters,Scinet
from qubitGeneration import DataGen

def createPdfLabel(numQubits):
    return "{} _Qubit_Bar_Graph".format(numQubits)

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

        self.numEpochs={1: 40, 2:80}
        self.encoderSize={1: 100, 2: 300}
        self.dataSetLen={1:100000, 2:300000}
        self.trainBatchSize={1:512, 2:1024}

        self.hyp=hyp
        self.cfg=cfg
    
    def editHyp(self,numQubits,obsSize,questionSize,latentNodes):
        self.hyp.encoderNodes=[obsSize,self.encoderSize[numQubits],100]
        self.hyp.encoderLayers=len(hyp.encoderNodes)
        self.hyp.questionNodes=questionSize
        self.hyp.decoderNodes=[100,100,1]
        self.hyp.decoderLayers=len(hyp.decoderNodes)

        self.hyp.testSize=1000
        self.hyp.trainBatchSize=self.trainBatchSize[numQubits]
        self.hyp.epochs=self.numEpochs[numQubits]
        self.hyp.annealEpoch=self.hyp.epochs

        self.hyp.learningRate=1e-2
        self.hyp.finalLr=1e-5

        self.hyp.latentNodes =latentNodes

        self.hyp.dataSetLen=self.dataSetLen[numQubits]

#trains scinet, runs test to get MSE which is returned
    def getError(self,trainingData,testingData):
        #network setup
        net=Scinet(hyp)
        net.lossFunct=lambda  mu, sig, X_rec, X: net.leadingLoss(X_rec,X)
        if torch.cuda.is_available():
            net.device=torch.device("cuda:0")
        else:
            net.device=torch.device("cpu")
        net.to(net.device)

        net.scheduler=optim.lr_scheduler.CosineAnnealingLR(net.optimizer,self.hyp.epochs,self.hyp.finalLr)

        #training and testing for MSE
        trainer=Trainer(trainingData,testingData)
        trainer.train(net,hyp)
        error=trainer.getMSE(net)
        return(error)

#generates dataset, creates network, trains network, gets MSE on test data
#subspacedim is only relevent for tomographically incomplete data, defualt is dummy
    def runQubitExample(self,numQubits,latentNodeList,tomComplete=True,subspaceDim=-1):
        if tomComplete:
            tomCompleteMessage="Tom. Complete"
        else:
            tomCompleteMessage="Tom. Incomplete"


        dimHilbert=2*(2**numQubits)-2
        
        #observation binary projective measurement set is restricted to a subspace
        #to generate tomographically incomplete data
        if tomComplete:
            obsSize=self.observationDim[numQubits]
        else:
            obsSize=subspaceDim
        
        questionSize=self.questionDim[numQubits]

        errorList=len(latentNodeList)*[0]
        
        for i in range(0,self.cfg.averaging):
            #data generation
            dataLabel=createDataLabel(numQubits,tomComplete)
            dataGen=DataGen()
            self.editHyp(numQubits,obsSize,questionSize,0)
            path=dataGen.generateDataSet(cfg,self.hyp.dataSetLen,dataLabel,dimHilbert,obsSize,questionSize)
            trainingData,testingData=getDataArray(path,self.hyp)

            for j,latentNodes in enumerate(latentNodeList):
                print(">>Finding Mean Square Error for {} Qubits and {} Latent Nodes ({}), Averaging Run {}".format(numQubits,latentNodes,tomCompleteMessage,i+1))
                self.editHyp(numQubits,obsSize,questionSize,latentNodes)
                errorList[j]+=self.getError(trainingData,testingData)
            
        errorList=[error/cfg.averaging for error in errorList]
        return errorList
    
    def plotErrors(self,numQubits):
        fig,ax=plt.subplots()
        xs=np.arange(len(self.QubitComp[numQubits]))
        barWidth=0.35
        completeBars=ax.bar(xs-barWidth/2,self.QubitComp[numQubits],width=barWidth,color="blue",label="Tom. Complete")
        IncompleteBars=ax.bar(xs+barWidth/2,self.QubitIncomp[numQubits],width=barWidth,color="orange",label="Tom. Incomplete")
        ax.set_xlabel("Number of Latent Neurons")
        ax.set_ylabel("Prediction Mean Square Error")
        ax.set_title("{} Qubit Example".format(numQubits))
        ax.set_xticks(xs)
        ax.legend()
        
        #auto increments file name
        i=1
        label=createPdfLabel(numQubits)
        while True:
            path=self.cfg.pdfSavePath+label+'_'+str(i)+'.pdf'
            i+=1
            try:
                f=open(path)
            except:
                plt.savefig(path)
                break
            
            if i==100:
                print("Unable to Save Figure, Folder Too Full")
                break

        plt.show()
    
    def main(self,numQubits):
        maxLatent=2*(2**numQubits)+2
        latentList=range(0,1)

        completeError=self.runQubitExample(numQubits,latentList)
        self.QubitComp[numQubits]=completeError

        incompleteError=self.runQubitExample(numQubits,latentList,False,2)
        self.QubitIncomp[numQubits]=incompleteError

        self.plotErrors(numQubits)



        



if __name__ == "__main__":
    #system args handling
    try:
        numQubits=int(sys.argv[1])
    except:
        print("Defaulting to 1 Qubit")
        numQubits=1
    else:
        if numQubits<1:
            print("Invalid Number of Qubits (must be positive int)")
            print("Defaulting to 1 Qubit")
            numQubits=1    

    cfg=Config()
    try:
        outputDir=str(sys.argv[2])
    except:
        print("Will Output to Defualt Path {}".format(cfg.pdfSavePath))
    else:
        cfg.pdfSavePath='quantumTomography/{}/'.format(outputDir)


    hyp=Hyperparameters()
    app=App(hyp,cfg)
    app.main(numQubits)