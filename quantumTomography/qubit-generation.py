import numpy as np
from random import uniform
from config import Config




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
        self.psis=[]
        self.basis=[]
        self.observations=[]
        self.questions=[]
        self.answers=[]

        #used to parameterize omega
        self.omegaBasis=[]

    def generateDataSet(self,cfg):
        #generates psi
        print("Generating Psis")
        for i in range(0,cfg.dataSetLen):
            phi=generateRandomQubit(cfg.dimHilbert)
            self.psis.append(phi)

        #generates phis (observationn basis)
        print("Generating Phis")
        self.basis=generateRandomSet(cfg.dimHilbert,cfg.observationBasisSize)

        #generates observations
        print("Generating Observations")
        for psi in self.psis:
            obseravtion=[]
            for phi in self.basis:
                measurement=measureState(psi,phi)
                obseravtion.append(measurement)
            obseravtion=np.array(obseravtion)
            self.observations.append(obseravtion)
        
        #generates omegas
        print("Generating Questions")
        for i in range(0,cfg.dataSetLen):
            #generates omegas
            self.questions.append(generateRandomQubit(cfg.dimHilbert))


        #generates answers
        print("Generating Answers")
        for i in range(0,cfg.dataSetLen):
            measurement=measureState(self.psis[i],self.questions[i])
            self.answers.append(np.array(measurement))
        self.answers=np.array(self.answers)


        #maps omegas onto parameterized omegas (done after finding soltuions to simplify math)
        print("Parameterizing Questions")
        self.omegaBasis=generateRandomSet(cfg.dimHilbert,cfg.quetionBasisSize)
        for i,omega in enumerate(self.questions):
            self.questions[i]=parameterizeQuestion(omega,self.omegaBasis)
        self.questions=np.array(self.questions,dtype=float)
        
        self.writeToFolder(cfg,"Complete")

    def writeToFolder(self,cfg,label):
        sets=[]
        for i in range(0,cfg.dataSetLen):
            dataSet=np.array((self.observations[i],self.questions[i],self.answers[i]))
            sets.append(dataSet)
        np.array(sets)

        i=1
        while True:
            path=cfg.dataPath+'_'+label+'_'+str(i)
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
    gen.generateDataSet(cfg)
    