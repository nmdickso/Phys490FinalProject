import numpy as np
from random import uniform
from config import Config
from scinet import Hyperparameters
from main import editHyp




def measureState(psi,phi):
    return abs(np.inner(psi,phi))**2

def generateRandomQubit(dim):
    qubit=np.zeros(dim,dtype=np.complex)
    singular=True
    while singular:
        for i in range(0,dim):
            qubit[i]=uniform(0,1)+uniform(0,1)*1j
        norm=np.linalg.norm(qubit)
        singular=norm<0.0000001
    
    return qubit/norm

def generateRandomSet(dim,numStates):
    states=[]
    for i in range(0,numStates):
        states.append(generateRandomQubit(dim))
    states=np.array(states)
    return(states)

def parameterizeQuestion(omega,vecSet):
    omega=[measureState(omega,vec) for vec in vecSet]
    return omega
    




class DataGen:
    def __init__(self):
        self.psi=0
        self.basis=[]
        self.observations=[]
        self.questions=[]
        self.answers=[]

        #used to parameterize omega
        self.omegaBasis=[]


    def generateDataSet(self,hyp,cfg):
        dim=hyp.encoderNodes[0]-2
        #generates psi
        print("Generating Psi")
        self.psi=generateRandomQubit(dim)

        #generates phis
        print("Generating Phis")
        self.basis=generateRandomSet(dim,dim+2)

        #generates observations
        print("Generating Observations")
        for phi in self.basis:
            measurement=measureState(self.psi,phi)
            self.observations.append(measurement)
        self.observations=np.array(self.observations)
        
        #generates omegas
        print("Generating Questions")
        for i in range(0,cfg.dataSetLen):
            #generates omegas
            self.questions.append(generateRandomQubit(dim))


        #generates answers
        print("Generating Answers")
        for question in self.questions:
            measurement=measureState(self.psi,question)
            self.answers.append(np.array(measurement))
        self.answers=np.array(self.answers)


        #maps omegas onto parameterized omegas (done after finding soltuions to simplify math)
        print("Parameterizing Questions")
        self.omegaBasis=generateRandomSet(dim,dim+2)
        for i,omega in enumerate(self.questions):
            self.questions[i]=parameterizeQuestion(omega,self.omegaBasis)
        self.questions=np.array(self.questions,dtype=float)
        
        self.writeToFolder(cfg)

    def writeToFolder(self,cfg):
        print("Writing to File")
        sets=[]
        for i in range(0,cfg.dataSetLen):
            dataSet=np.array((self.observations,self.questions[i],self.answers[i]))
            sets.append(dataSet)
        np.array(sets)

        i=1
        while True:
            path=cfg.dataPath+str(i)
            print("Trying to Write",path)
            i+=1
            try:
                f=np.load(path+".npy",allow_pickle=True)
            except:
                np.save(path,sets)
                break
            
            if i==100:
                print("Unable to Save Data")
                break

            
                
        print("Data Written to {}".format(path))

if __name__ == "__main__":
    gen=DataGen()
    cfg=Config()
    hyp=Hyperparameters()
    editHyp(hyp)
    gen.generateDataSet(hyp,cfg)
    