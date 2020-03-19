import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

class Trainer:
    def __init__(self,cfg,trainingData,testingData):
        self.trainingAccuracy=[]
        self.trainingLoss=[]

        self.testingAccuracy=[]
        self.testingLoss=[]

        self.trainingImgs=torch.from_numpy(np.stack(trainingData[:,0]))

        self.testingImgs=torch.from_numpy(np.stack(testingData[:,0]))
    
    def trainAndTest(self,neuralNet,hyp):
        for i in range(0,hyp.epochs):
            print("Training Epoch {}:".format(i+1))
            trainLoss=neuralNet.train(self.trainingImgs,hyp.trainBatchSize,i)
            self.trainingLoss.append(trainLoss)
            
            print("Testing Epoch {}:".format(i+1))
            testLoss=neuralNet.test(self.testingImgs,hyp.testBatchSize,i)
            self.testingLoss.append(testLoss)
