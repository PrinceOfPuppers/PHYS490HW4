import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self,cfg,trainingData):
        self.trainingLoss=[]

        self.trainingImgs=torch.from_numpy(np.stack(trainingData[:,0]))
        self.trainingLabels=torch.from_numpy(np.stack(trainingData[:,1]))

    
    def train(self,neuralNet,hyp):
        for i in range(0,hyp.epochs):
            print("Training Epoch {}:".format(i+1))
            trainLoss=neuralNet.train(self.trainingImgs,hyp.trainBatchSize,i)
            self.trainingLoss.append(trainLoss)