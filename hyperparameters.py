import torch.optim as optim
import torch as torch
import torch.nn.functional as funct
import torch.nn as nn
from math import sqrt
class convLayer:
    def __init__(self,inChannel,outChannel,kernal,stride,maxPool):
        self.conv=nn.Conv2d(inChannel,outChannel,kernal,stride)
        self.convInv=nn.ConvTranspose2d(outChannel,inChannel,kernal,stride)

        self.maxPool=nn.MaxPool2d(maxPool)
        self.maxPoolInv=nn.MaxUnpool2d(maxPool)

class flattener:
    def __init__(self,flattenDim,channels):
        self.flattenDim=flattenDim
        self.channels=channels
    def flatten(x):
        x.view(-1,self.flattenDim)
        return x
    def unflatten(x):
        x.view(-1,channels,sqrt(self.flattenDim/self.channels),sqrt(self.flattenDim/self.channels))
        return x

class linearLayer:
    def __init__(self,inNodes,outNodes):
        self.lin=nn.Linear(inNodes,outNodes)
        self.linInv=nn.Linear(outNodes,inNodes)

class Hyperparameters:
    def __init__(self):
        self.imageRes=(14,14)
        self.latentNodes = 2

        # Note learning rate and batch size must not be too high due to
        # dead relu problems
        self.learningRate = 0.001

        self.optimizer = optim.Adam
        self.trainBatchSize=50
        self.testBatchSize=1
        self.testSize=10
        self.epochs=20
        
    def makeLayers(self):
        self.layers=[
            convLayer(1,16,3,2,(2,2)),
            convLayer(16,32,5,2,(1,1))
        ]
        flattenDim=self.determineFlatten(self.layers)
        self.layers.extend([
            flattener(flattenDim,32),
            linearLayer(flattenDim,100)
        ])
    def determineFlatten(self,layers):
        x=torch.randn(1,1,self.imageRes[0],self.imageRes[1]).view(-1,1,self.imageRes[0],self.imageRes[1])
        for i,layer in enumerate(layers):
            layer.maxPool(funct.relu(layer.conv(x)))
        return x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

    


    def betaFunct(self,epoch):
        return 0.05*(epoch/self.epochs)