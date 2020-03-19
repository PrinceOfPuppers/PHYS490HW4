import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
from math import sqrt
from tqdm import tqdm

class NeralNet(nn.Module):
    def __init__(self, hyp):
        super().__init__()
        self.inputRes=hyp.imageRes
        self.imagChannels=1
        self.maxPooling=[(2,2),(1,1)]
        # Populate encoder
        self.encoder=nn.ModuleList()
        
        convLayer=nn.Conv2d(self.imagChannels,32,3,1)
        self.encoder.append(convLayer)
        convLayer=nn.Conv2d(32,64,3,1)
        self.encoder.append(convLayer)
        
        self.numConv=len(self.encoder)
        self.flattenDim=self.determineFlatten()

        fcLayer = nn.Linear(self.flattenDim, 32)
        self.encoder.append(fcLayer)

        # Add latent layer
        self.latent=nn.ModuleList()

        self.mu=nn.Linear(32, hyp.latentNodes)
        self.sigma=nn.Linear(32, hyp.latentNodes)
        
        self.latent.append(self.mu)
        self.latent.append(self.sigma)

        # Populate decoder
        self.decoder = nn.ModuleList()

        fcLayer = nn.Linear(hyp.latentNodes, 32)
        self.decoder.append(fcLayer)
        fcLayer = nn.Linear(32,100)
        self.decoder.append(fcLayer)
        fcLayer = nn.Linear(100,14*14)
        self.decoder.append(fcLayer)

        # Use only CPU due to small network size and occasional bugs
        self.device = torch.device("cuda:0")
        self.to(self.device)

        # Optimizer and loss functions
        self.optimizer = hyp.optimizer(self.parameters(), hyp.learningRate)
        self.betaFunct=hyp.betaFunct
    

    def determineFlatten(self):
        x=torch.randn(1,1,self.inputRes[0],self.inputRes[1]).view(-1,1,self.inputRes[0],self.inputRes[1])
        for i,conv in enumerate(self.encoder):
            if i==self.numConv:
                break
            x=funct.max_pool2d(funct.relu(conv(x)),self.maxPooling[i])
        return x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
        
    def lossFunct(self,output,target,mu,sigma,epoch):
       sigmaSqrd=torch.mul(sigma,sigma)
       klDivergence=torch.sum(sigmaSqrd+torch.mul(mu,mu)-torch.log(sigmaSqrd)-torch.ones(mu.size()).to(self.device))
       loss=funct.mse_loss(output,target)
       return loss+self.betaFunct(epoch)*klDivergence
    
    def forward(self, x):
        x=x.view(-1,1,self.inputRes[0],self.inputRes[1])
        #pass through convolution
        for i,layer in enumerate(self.encoder):
            if i<self.numConv:
                x=funct.max_pool2d(funct.relu(layer(x)),self.maxPooling[i])
            elif i==self.numConv:
                x=x.view(-1,self.flattenDim)
                x=funct.relu(layer(x))
            else:
                x = funct.relu(layer(x))

        # Pass through latent layer
        mu=self.mu(x)
        sigma=self.sigma(x)
        #sample
        x=torch.distributions.Normal(mu,sigma).sample()
        # Pass through decoder layers (without applying relu on answer neuron)

        for i, layer in enumerate(self.decoder):
            if i==len(self.decoder)-1:
                x=layer(x)
            else:
                x = funct.relu(layer(x))

        return x,mu,sigma




    def sampleLatent(self,x):
        x=torch.from_numpy(x)
        x=x.to(self.device)

        for i, layer in enumerate(self.decoder):
            if i==len(self.decoder)-1:
                x=layer(x)
            else:
                x = funct.relu(layer(x))

        return x



    def train(self,trainData,batchSize,epoch):
        trainSize=len(trainData)
        avgLoss=0
        for i in tqdm(range(0,trainSize,batchSize)):
            imgBatch=trainData[i:i+batchSize].to(self.device)
            self.zero_grad()
            outputs,mu,sigma=self(imgBatch)
            loss=self.lossFunct(outputs,imgBatch,mu,sigma,epoch)
            loss.backward()
            self.optimizer.step()

            avgLoss+=loss.item()*len(imgBatch)

        avgLoss=avgLoss/trainSize
        print("Training loss:",avgLoss)
        return (avgLoss)
    
    def test(self,testData,batchSize,epoch):
        avgLoss=0
        testSize=len(testData)
        with torch.no_grad():
            for i in tqdm(range(0,testSize,batchSize)):
                imgBatch=testData[i:i+batchSize].to(self.device)
                outputs,mu,sigma=self(imgBatch)
                loss=self.lossFunct(outputs,imgBatch,mu,sigma,epoch)

                avgLoss+=loss.item()*len(imgBatch)

        avgLoss=avgLoss/testSize
        print("Testing Loss:",avgLoss)
        return (avgLoss)