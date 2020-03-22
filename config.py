from math import sin,cos,pi
import numpy as np

class Config:
    def __init__(self):
        self.netSaveLocation="savedNetworks\savedNet"
        self.saveNet=False
        self.generateNewNet=False

        self.inputPhotoDim=(14,14)
        #note order in this list will be used to determine which one hot vector corrisponds
        #to which label 
        self.allLabels=[0,2,4,6,8]
        self.scatterColors=["red","green","blue","yellow","black"]

        self.dataPath="even_mnist.csv"
        self.numLabels=len(self.allLabels)

        self.inputLen=self.inputPhotoDim[0]*self.inputPhotoDim[1]

        self.framerate=30
        self.totalFrames=160
        self.tVals=np.linspace(0,2*pi,self.totalFrames)
        
    def parametricSeedPoint(self,t):
        c=np.array([cos(t),cos(t)*sin(t)],dtype=np.float32)
        return c