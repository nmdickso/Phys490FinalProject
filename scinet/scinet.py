import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
from tqdm import tqdm


class Scinet(nn.Module):
    def __init__(self,hyp):
        super().__init__()
        #populates encoder
        self.encoder=nn.ModuleList()
        for i in range(0,hyp.encoderLayers-1):
            
            inputNodes=hyp.encoderNodes[i]
            outputNodes=hyp.encoderNodes[i+1]

            fcLayer=nn.Linear(inputNodes,outputNodes)
            self.encoder.append(fcLayer)
        
        #adds latent layer
        inputNodes=hyp.encoderNodes[-1]
        outputNodes=hyp.latentNodes


        latent=nn.Linear(hyp.encoderNodes[-1],hyp.latentNodes)
        self.latent=latent


        #populates decoder
        #input to decoder is number of quesiton nodes plus number of latent nodes
        self.decoder=nn.ModuleList()
        inputNodes=hyp.latentNodes+hyp.questionNodes
        outputNodes=hyp.decoderNodes[0]
        firstDecoderLayer=nn.Linear(inputNodes,outputNodes)
        self.decoder.append(firstDecoderLayer)

        for i in range(1,hyp.decoderLayers-1):
            inputNodes=hyp.decoderNodes[i]
            outputNodes=hyp.decoderNodes[i+1]
            fcLayer=nn.Linear(inputNodes,outputNodes)
            self.decoder.append(fcLayer)


        #selects device (gpu isnt used because of small network size and occasional bad behavior)
        self.device=torch.device("cpu")
        self.to(self.device)

        #optimizer and loss 
        self.optimizer=hyp.optimizer(self.parameters(),hyp.learningRate)
        self.lossFunct=hyp.lossFunct()

    def forward(self,x,question):
        #dummy question neuron
        #question=torch.Tensor(x.size()[0]*[[0]]).to(self.device)

        #pass through encoder layers
        for layer in self.encoder:
            x=funct.relu(layer(x))
        
        #pass through latent layer
        x=funct.relu(self.latent(x))


        #concatinate output of encoder with question
        x=torch.cat((question,x),dim=-1)

        #pass through decoder layers (dont apply relu on answer neuron)
        lastDecoderLayer=len(self.decoder)-1

        for i,layer in enumerate(self.decoder):
            x=layer(x)

            if i!=lastDecoderLayer:
                x=funct.relu(x)
        
        return x
    
    def train(self,observations,answers,batchSize):
        trainSize=len(observations)
        avgLoss=0
        for i in tqdm(range(0,trainSize,batchSize)):
            observationBatch=observations[i:i+batchSize].to(self.device)
            answersBatch=answers[i:i+batchSize].to(self.device)
            
            self.zero_grad()
            outputs=self(observationBatch)
            loss=self.lossFunct(outputs,answersBatch)
            loss.backward()
            self.optimizer.step()

            avgLoss+=loss.item()*len(observationBatch)

        avgLoss/=trainSize
        print("Training loss:",avgLoss)
        return (avgLoss)
    
    def test(self,observations,answers,batchSize):
        avgLoss=0
        testSize=len(observations)
        with torch.no_grad():
            for i in tqdm(range(0,testSize,batchSize)):
                observationBatch=observations[i:i+batchSize].to(self.device)
                answersBatch=answers[i:i+batchSize].to(self.device)
                outputs=self(observationBatch)
                loss=self.lossFunct(outputs,answersBatch)

                avgLoss+=loss.item()*len(observationBatch)


        avgLoss/=testSize
        print("Testing Loss:",avgLoss)
        return (avgLoss)
