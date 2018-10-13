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
            self.weights.append(np.random.normal(scale = 0.1 , size=(l2 , l1+1)))


    def Run(self , input):
        inCases = input.shape[0]
        #clear out the previous value lists
        self._layerInput=[]
        self._layeroutput=[]

        #Run it

        for index in range(self.layercount):
            if index == 0:
                layerinput = self.weights[0].dot(np.vstack([input.T ,np.ones([1 , inCases])]))
            else:
                layerinput = self.weights[index].dot(np.vstack([self._layeroutput[-1] , np.ones([1, inCases])]))

            self._layerInput.append(layerinput)
            self._layeroutput.append(self.sgm(layerinput))

        return (self._layeroutput[-1].T)

    def train_epoch(self, input: object, target: object, trainingRate: object = 0.2, layeroutput: object = None) -> object:
        """this method trains the network for 1 epoch"""
        delta =[]
        inCases = input.shape[0]

        #run the network
        self.Run(input)

        #calulate deltas
        for index  in reversed(range(self.layercount)):
            if index == self.layercount-1:
                #compare targert values
                output_delta = self._layeroutput - target.T
                error= np.sum(output_delta**2)
                delta.append(output_delta * self.sgm(self._layerInput[index] , True))

            else:
                #compairng to following layers delta value
                delta_pullback = self.weights[index-1].T.dot(delta[-1])
                delta.append(output_delta * self.sgm(self._layerInput[index] ,True))

            #compute weight delta
        for index in range(self.layercount):
            delta_index = self.layercount-1-index

            if index ==0:
                layeroutput = np.vstack([self._layeroutput[index-1] , np.ones([1,inCases])])
            else:
                layeroutput = np.vstack([self._layeroutput[index-1], np.ones([1 , self._layeroutput[index-1]])])

            weightDelta = np.sum(layeroutput[None , :, :].transpose(2,0,1) * delta[delta_index][None , : , :].transpose(2,1,0) , axis=0)
            self.weights[index] -= trainingRate * weightDelta
        return error



    def sgm(self , x , Derivative=False):
        if not Derivative:
            return 1/(1+np.exp(-x))
        else:
            out = self.sgm(x)
            return out*(1-out)





if __name__ == "__main__":
    bpn = BackPropagationNetwork((2,2,1))
    print(bpn.shape)
    print(bpn.weights)


    ivinput = np.array([[0,0] , [1,1] , [0 ,1] , [1,0]])
    ivtarget = np.array([[0.05] , [0.05] , [0.95] , [0.95]])

    inMax = 100000
    InErr = 1e-5
    for i in range(inMax+1):
        err = bpn.train_epoch(ivinput , ivtarget)
        if  i %10000 ==0:
            print("Itertaion {0}\t Error:{0.6f}".format(i ,err))

        if err<=InErr:
            print("Minimum error reaches at iteration {0}".format(i))
            break

    ivoutput = bpn.Run(ivinput)

    print("input {0} output {1}".format(ivinput , ivoutput))






