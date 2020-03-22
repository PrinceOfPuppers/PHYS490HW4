from math import sin,cos,pi
import numpy as np
import sys,getopt
import os

class Config:
    def __init__(self):
        self.netSaveLocation="savedNetworks\savedNet"
        self.saveNet=True
        self.generateNewNet=True

        self.inputPhotoDim=(14,14)
        #note order in this list will be used to determine which one hot vector corrisponds
        #to which label 
        self.allLabels=[0,2,4,6,8]
        self.scatterColors=["red","green","blue","yellow","black"]

        self.dataPath="data\even_mnist.csv"
        self.numLabels=len(self.allLabels)

        self.inputLen=self.inputPhotoDim[0]*self.inputPhotoDim[1]

        self.framerate=30
        self.totalFrames=160
        self.tVals=np.linspace(0,2*pi,self.totalFrames)
    
    def getArgs(self):
        opts,args=getopt.getopt(sys.argv[1:],'o:n:')
        for opt,arg in opts:
            if opt=='-o':
                arg=arg.strip()
                outputFile=arg
            elif opt=='-n':
                arg=int(arg.strip())
                number=arg
        print("\n")
        try:
            number
        except:
            print("Please input -n argument (number of ouput PDFs)")
            exit()
        try:
            outputFile
        except:
            print("Please input -o argument (output dir name)")
            exit()

        self.numPDFs=number
        self.savePDFLocation=outputFile

        if not os.path.exists(self.savePDFLocation):
            os.mkdir(self.savePDFLocation)

    def parametricSeedPoint(self,t):
        amp=2
        c=np.array([amp*cos(t),amp*cos(t)*sin(t)],dtype=np.float32)
        return c