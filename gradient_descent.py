import numpy as np
alphas =[0.001 , 0.01 , 0.1 ,1,10 ,100 ,1000]
hidden_size =32

#compute sigmoid non linearity
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return  output

#convert to derivative
def sig_to_derivative(output):
    return output*(1-output)

x = np.array([[0,0,1] , [0,1,1] , [1,0,1] , [1,1,1]])
y =np.array([[0] , [1] , [1] , [0]])
for alpha in alphas:
    print("\n Training WIth alpha:"+str(alpha))
    np.random.seed(1)


    #initialise the weights
    synapse_0 = 2*np.random.random((3 , hidden_size))-1
    synapse_1 =2*np.random.random((hidden_size , 1))-1

    for j in range(60000):
        layer_0 = x
        layer_1 = sigmoid(np.dot(layer_0 , synapse_0))
        layer_2 = sigmoid(np.dot(layer_1 , synapse_1))

        #how much didi ii miss?

        layer_2_error = layer_2 -y

        if(j%10000)==0:
            print("Error sfter "+ str(j)+" iterations: "+str(np.mean(np.abs(layer_2_error))))
            layer_2_delta  = layer_2_error*sig_to_derivative(layer_2)


            #how much did l1 contribuute ti l2 error
            layer_1_error = layer_2_delta.dot(synapse_1.T)

            layer_1_delta = layer_1_error * sig_to_derivative(layer_1)

            synapse_1 -= alpha * (layer_1.T.dot(layer_2_delta))
            synapse_0 -= alpha * (layer_0.T.dot(layer_1_delta))
