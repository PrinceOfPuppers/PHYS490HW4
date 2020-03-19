from math import sin,cos,pi
import numpy as np

class Config:
    def __init__(self):
        self.inputPhotoDim=(14,14)
        #note order in this list will be used to determine which one hot vector corrisponds
        #to which label 
        self.allLabels=[0,2,4,6,8]

        self.dataPath="even_mnist.csv"
        self.numLabels=len(self.allLabels)

        self.inputLen=self.inputPhotoDim[0]*self.inputPhotoDim[1]

        self.framerate=30
        self.totalFrames=160
        self.tVals=np.linspace(0,2*pi,self.totalFrames)
        
    def parametricSeedPoint(self,t):
        c=np.array([1*(sin(t)+1),1*(cos(t)+1)],dtype=np.float32)
        return c