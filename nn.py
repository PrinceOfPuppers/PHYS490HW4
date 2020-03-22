import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
from math import sqrt
from tqdm import tqdm

def saveNet(net,cfg):
    print("Saving Network")
    torch.save(net,cfg.netSaveLocation)

def loadNet(cfg):
    print("Loading Network")
    net=torch.load(cfg.netSaveLocation)
    return net

class NeralNet(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 10,3),
            nn.ReLU(),
        )
        self.flattenDim=self.determineFlatten()
        self.alsoEncoder=nn.Sequential(
            nn.Linear(self.flattenDim, 100),
            nn.ReLU(),
            nn.Linear(100,50),
            nn.ReLU()
        )
        self.logSigma=nn.Linear(50, 2)
        self.mu=nn.Linear(50, 2)

        self.decoder=nn.Sequential(
            nn.Linear(2,25),
            nn.ReLU(),            
            nn.Linear(25,75),
            nn.ReLU(),
            nn.Linear(75,14*14),
            nn.ReLU(),
            nn.Linear(14*14,14*14),
            nn.Sigmoid()
        )

        if torch.cuda.is_available():
            print("Using Cuda")
            self.device=torch.device("cuda:0")
        else:
            print("Cuda not avalible, using CPU")
            self.device=torch.device("cpu")
        self.to(self.device)

        self.betaFunct=hyp.betaFunct
        self.optimizer=optim.Adam(self.parameters(),lr=hyp.initalLr)
        self.scheduler=optim.lr_scheduler.CosineAnnealingLR(self.optimizer,hyp.epochs,hyp.finalLr)

    def determineFlatten(self):
        x=torch.randn(1,1,14,14).view(-1,1,14,14)
        x=self.encoder(x)
        return x[0].shape[0]*x[0].shape[1]*x[0].shape[2]

    def lossFunct(self,output,target,mu,logSigma,epoch):
       sigma=torch.exp(logSigma)

       klDivergence=0.5*torch.sum(sigma+torch.mul(mu,mu)-logSigma-1)
       recLoss=funct.binary_cross_entropy(output,target,reduction="sum")
       loss=recLoss+self.betaFunct(epoch)*klDivergence
       return loss,recLoss,klDivergence
        

    def forward(self,x):
        x=x.view(-1,1,14,14)
        x=self.encoder(x)
        x=x.view(-1,self.flattenDim)
        x=self.alsoEncoder(x)
        mu=self.mu(x)
        logSigma=self.logSigma(x)

        sigma = torch.exp(logSigma)
        normal = torch.randn_like(sigma)
        x = mu + normal * sigma

        x=self.decoder(x)

        return x,mu,logSigma

    def sampleLatent(self,x):
        x=torch.from_numpy(x)
        x=x.to(self.device)

        x=self.decoder(x)

        x=x.view(14*14)
        return x


    def sampleEncoder(self,x):
        x=x.view(-1,1,14,14)
        #x=x.view(-1,1,self.inputRes[0],self.inputRes[1])
        x=x.to(self.device)
        #pass through enocder
        x=self.encoder(x)
        x=x.view(-1,self.flattenDim)
        x=self.alsoEncoder(x)

        # Pass through latent layer
        mu=self.mu(x)
        logSigma=self.logSigma(x)
        sigma = torch.exp(logSigma)
        normal = torch.randn_like(sigma)
        x = mu + normal * sigma
        return(x)
        
    def train(self,trainData,batchSize,epoch):

        trainSize=len(trainData)
        avgLoss=0
        avgRecLoss=0
        avgKLD=0
        for i in range(0,trainSize,batchSize):

            imgBatch=trainData[i:i+batchSize].to(self.device)
            self.zero_grad()
            outputs,mu,logSigma=self(imgBatch)
            loss,recLoss,kLD=self.lossFunct(outputs,imgBatch,mu,logSigma,epoch)
            loss.backward()
            self.optimizer.step()

            avgLoss+=loss.item()*len(imgBatch)
            avgRecLoss+=recLoss.item()*len(imgBatch)
            avgKLD+=kLD.item()*len(imgBatch)

        avgLoss=avgLoss/trainSize
        avgRecLoss=avgRecLoss/trainSize
        avgKLD=avgKLD/trainSize
        print("Training loss:",avgLoss)
        self.scheduler.step()
        return (avgLoss,avgRecLoss,avgKLD)
    