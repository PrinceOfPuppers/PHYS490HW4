class Hyperparameters:
    def __init__(self):
        self.initalLr=0.01
        self.finalLr = 0.0005
        self.trainBatchSize=500
        self.epochs=40

    def betaFunct(self,epoch):
        return 0.5*(epoch/(self.epochs-1))
    