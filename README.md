# Phys 490: Final Project
# Discovering Physical Concepts with Neural Networks

# The Paper


# SciNet

The neural network structure *SciNet* was recreated in PyTorch. The observations are encoded as real parameters, and are fed to an encoder (a feed-forwward neural network). The encoder compresses the data into a latent representation. The question is encoded in N real parameters, which together with the latent representation are fed to the decodr network to generate an answer. 

# Running the Network

To run the network, enter the following into the command line:

``` python main.py --param files/param.json ... etc```

