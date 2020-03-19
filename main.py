import numpy as np
import sys
import json
from random import random
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from config import Config
from hyperparameters import Hyperparameters
from nn import NeralNet
from trainer import Trainer

#function for debugging data loading, along with checking balance 
#last 2 lines are commented out to remove matplotlib dependancy

def dataInfo(cfg,data,balanceTracker):
    print("data Loaded with Following Balance")
    for i in range(0,cfg.numLabels):
        print("label:{}   number of data points:{}".format(cfg.allLabels[i],balanceTracker[i]))

    exampleNum=int(len(data)*random())
    dataPoint=data[exampleNum][0]
    dataPointType=cfg.allLabels[np.argmax(data[exampleNum][1])]
    print("Displaying example datapoint number:{}, type:{}".format(exampleNum,dataPointType))

    #unflattens the data to be displayed by matplotlib
    img=np.zeros(cfg.inputPhotoDim)
    for i in range(0,len(dataPoint)):
        indices=(i//cfg.inputPhotoDim[0],i%cfg.inputPhotoDim[1])
        img[indices[0],indices[1]]=dataPoint[i]
    #plt.imshow(img)
    #plt.show()

def sampleLatentSpace(cfg,nn,point):
    data=nn.sampleLatent(point)
    #unflattens the data to be displayed by matplotlib
    frame=np.zeros(cfg.inputPhotoDim)
    for i in range(0,len(data)):
        indices=(i//cfg.inputPhotoDim[0],i%cfg.inputPhotoDim[1])
        frame[indices[0],indices[1]]=data[i]
    return frame

def makeAnimation(cfg,nn):
    frameList=[]
    fig=plt.figure()
    for frameNumber,t in enumerate(cfg.tVals):
        print("rendering frame",frameNumber+1)
        point=cfg.parametricSeedPoint(t)
        imageArray=sampleLatentSpace(cfg,nn,point)
        frame=plt.imshow(imageArray,cmap='Greys')

        frameList.append([frame])
    animation=ani.ArtistAnimation(fig,frameList,interval=(1000/cfg.framerate))
    plt.show()


def getDataArray(path,cfg):
    print("Loading Data")
    rawData=open(path)
    data=[]
    balanceTracker=[0]*cfg.numLabels

    for line in rawData:
        line=line.strip()
        label=float(line[-1])
        labelNum=cfg.allLabels.index(label)

        balanceTracker[labelNum]+=1

        #creates one hot vector for label 
        labelVec=np.eye(cfg.numLabels,dtype=np.float32)[labelNum]

        imgVec=tuple(map(float,line.split()[0:-1]))
        imgVec=np.array([i/255 for i in imgVec],dtype=np.float32)

        #image and label get put into an array togeater so they remain togeather 
        #through shuffling
        dataPair=np.array((imgVec,labelVec))
        data.append(dataPair)
    data=np.array(data)
    return(data,balanceTracker)

#shuffles data and splits it into training and testing sets
def partitionAndShuffleData(hyp,data):
    #Shuffle Data
    print("Shuffling Data")
    np.random.shuffle(data)
    trainingData=data[0:-hyp.testSize]
    testingData=data[:hyp.testSize]

    return(trainingData,testingData)



if __name__ == "__main__":
    cfg=Config()
    hyp=Hyperparameters()

    data,balanceTracker=getDataArray(cfg.dataPath,cfg)
    trainingData,testingData=partitionAndShuffleData(hyp,data)

    net=NeralNet(hyp)
    trainer=Trainer(cfg,trainingData,testingData)

    trainer.trainAndTest(net,hyp)

    makeAnimation(cfg,net)
    