import numpy as np
from math import sin,cos,pi
import getopt,sys
from random import random,shuffle
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from tqdm import tqdm
from config import Config
from hyperparameters import Hyperparameters
from nn import NeralNet,saveNet,loadNet
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
    print("Rendering Animation")
    for t in tqdm(cfg.tVals):
        point=cfg.parametricSeedPoint(t)
        imageArray=sampleLatentSpace(cfg,nn,point)
        frame=plt.imshow(imageArray,cmap='Greys')

        frameList.append([frame])
    animation=ani.ArtistAnimation(fig,frameList,interval=(1000/cfg.framerate))
    plt.show()

def makeLatentSpaceMap(cfg,nn,trainer):
    pointDict={label: [] for label in cfg.allLabels}
    numSamples=int(len(trainer.trainingImgs)/50)
    print("Sampling Encoder with {} Data Points".format(numSamples))
    for i in tqdm(range(0,numSamples)):
        img=trainer.trainingImgs[i]
        label=cfg.allLabels[int(trainer.trainingLabels[i].argmax())]

        point=nn.sampleEncoder(img).tolist()[0]
        pointDict[label].append(point)
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
    for i,label in tqdm(enumerate(pointDict)):
        pointsX=[point[0] for point in pointDict[label]]
        pointsY=[point[1] for point in pointDict[label]]
        
        ax.scatter(pointsX,pointsY,color=cfg.scatterColors[i])
    
    ax.set_title("Map of Latent Space")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    #ax.legend()
    plt.show()

    

def getDataArray(path,cfg):
    print("Loading Data")
    rawData=open(path)
    data=[]
    balanceTracker=[0]*cfg.numLabels

    for line in tqdm(rawData):
        labelIndex=-1
        line=line.strip()
        label=float(line[labelIndex])
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

def shuffleData(data):
    np.random.shuffle(data)
    return(data)

def generatePDFs(nn,cfg,hyp,trainer):
    print("Saving Results")
    #plots losses
    epochs=[i for i in range(0,hyp.epochs)]
    plt.plot(epochs,trainer.recLoss,label="Reconstruction Loss",color='green')
    plt.plot(epochs,trainer.kLD,label="KLDivergence", color='blue')
    plt.title("Reconstruction Loss and KLD over Epochs")
    plt.xlabel("Epochs")
    plt.legend()
    plt.savefig(cfg.savePDFLocation+'\{}'.format("Loss"))
    plt.clf()

    #plots samples
    rs=[2*j/cfg.numPDFs for j in range(0,cfg.numPDFs)]
    thetas=[2*pi*j/cfg.numPDFs for j in range(0,cfg.numPDFs)]
    shuffle(rs)
    shuffle(thetas)
    for i in range(0,cfg.numPDFs):
        r=rs[i]
        theta=thetas[i]
        x=r*cos(theta)
        y=r*sin(theta)
        point=np.array([x,y],dtype=np.float32)
        sample=sampleLatentSpace(cfg,nn,point)
        plt.imshow(sample,cmap='Greys')
        plt.title("figure {}; Latent Space Vector ({},{})".format(i,round(x,3),round(y,3)))
        plt.savefig(cfg.savePDFLocation+'\{}'.format(i))
        plt.clf()
    
    print("Results saved in {}".format(cfg.savePDFLocation))




if __name__ == "__main__":
    cfg=Config()
    cfg.getArgs()


    hyp=Hyperparameters()

    data,balanceTracker=getDataArray(cfg.dataPath,cfg)
    trainingData=shuffleData(data)

    net=NeralNet(hyp)
    trainer=Trainer(cfg,trainingData)

    trainer.train(net,hyp)
    if trainer.trainingLoss[-1]>20300:
        print(">>Network Stuck in Local Minimum, Please Re-run to get Proper Results")


    generatePDFs(net,cfg,hyp,trainer)

