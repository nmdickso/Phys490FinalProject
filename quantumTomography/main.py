import numpy as np
import sys
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from random import random
from config import Config
from trainer import Trainer
from scinet import Hyperparameters,Scinet
from qubitGeneration import DataGen
def createPdfLabel(numQubits):
    return "{} _Qubit_Bar_Graph.pdf".format(numQubits)

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
        self.qbits=[1,2]
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
        self.hyp.epochs=40
        self.hyp.annealEpoch=self.hyp.epochs

        self.hyp.learningRate=1e-2
        self.hyp.finalLr=1e-5

        self.hyp.latentNodes =latentNodes

    def getError(self,trainingData,testingData):
        #network setup
        net=Scinet(hyp)
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

#generates dataset, creates network, trains network, returns MSE on test data
#subspacedim is only relevent for tomographically incomplete data
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

            path=dataGen.generateDataSet(cfg,dataLabel,dimHilbert,obsSize,questionSize)
            trainingData,testingData=getDataArray(path,self.hyp)

            for j,latentNodes in enumerate(latentNodeList):
                print(">>Finding Mean Square Error for {} Qubits and {} Latent Nodes ({}), Averaging Run {}".format(numQubits,latentNodes,tomCompleteMessage,i))
                self.editHyp(obsSize,questionSize,latentNodes)
                errorList[j]+=self.getError(trainingData,testingData)
            
        errorList=[error/cfg.averaging for error in errorList]
        return errorList
    
    def plotErrors(self):
        for numQubits in self.qbits:
            fig,ax=plt.subplots()
            xs=np.arange(len(self.QubitComp[numQubits]))
            barWidth=0.35
            completeBars=ax.bar(xs-barWidth/2,self.QubitComp[numQubits],width=barWidth,color="blue",label="Tom. Complete")
            IncompleteBars=ax.bar(xs+barWidth/2,self.QubitIncomp[numQubits],width=barWidth,color="orange",label="Tom. Incomplete")
            ax.set_xlabel("Number of Latent Neurons")
            ax.set_ylabel("Prediction Mean Square Error")
            ax.set_title("One Qubit Example")
            ax.set_xticks(xs)
            ax.legend()
            
            plt.savefig(self.cfg.pdfSavePath+createPdfLabel(numQubits))
            plt.show()
    
    def main(self):
        for numQubits in self.qbits:
            latentList=range(0,2*(2**numQubits)+2)

            completeError=self.runQubitExample(numQubits,latentList)
            self.QubitComp[numQubits]=completeError

            incompleteError=self.runQubitExample(numQubits,latentList,False,2)
            self.QubitIncomp[numQubits].append(incompleteError)

        self.plotErrors()



        



if __name__ == "__main__":
    hyp=Hyperparameters()
    cfg=Config()
    app=App(hyp,cfg)
    app.main()