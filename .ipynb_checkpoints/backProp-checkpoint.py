import numpy as np



class BackPropagationNetwork:
    layercount = 0
    shape = None
    weights = []

    def __init__(self , layersize):
        self.layercount = len(layersize)-1
        self.shape = layersize


        #input outpuit from last run
        self._layerInput = []
        self._layeroutput  = []

        #create weighht arrays
        for (l1 ,l2) in zip(layersize[:-1] , layersize[1:]) :
            self.weights.append(np.random.normal(scale = 0.01 , size=(l2 , l1-1)))





if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,5,1))
    print(bpn.shape)
    print(bpn.weights)








