import torch.optim as optim
import torch.nn as nn

class Hyperparameters:
    def __init__(self):
        #first is for observation nodes
        self.encoderNodes=[5,100,100]
        self.encoderLayers=len(self.encoderNodes)

        self.latentNodes=2
        self.questionNodes=1

        #last is for answer neuron
        self.decoderNodes=[100,100,5]
        self.decoderLayers=len(self.decoderNodes)
        self.answerNodes=1
        
        #note learning rate should not be set too high because 
        #of dead relu problems this is also effected by higher batch sizes
        self.learningRate=0.001

        self.optimizer=optim.Adam
        self.lossFunct=nn.MSELoss

        
        