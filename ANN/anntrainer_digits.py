__author__ = 'piyush'
import numpy as np
import time

class BackPropagationNetwork:

    layerCount = 0;
    shape  = None;
    weights = [];

    def __init__(self,layerSize):

        self.layerCount = len(layerSize) - 1;
        self.shape = layerSize

        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        for (l1,l2) in zip(layerSize[:-1],layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.1,size=(l2,l1+1)))
            self._previousWeightDelta.append(np.zeros((l2,l1+1)))

    def forwardProc(self,input):

        InCases = input.shape[0]

        self._layerInput = []
        self._layerOutput = []

        for index in range(self.layerCount):
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T,np.ones([1,InCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1],np.ones([1,InCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.sgm(layerInput))

        return self._layerOutput[-1].T

    def train(self,input,target, trainingRate = 0.005, momentum = 0.5):

        delta = []
        InCases = input.shape[0]

        self.forwardProc(input)

        for index in reversed(range(self.layerCount)):

            if index == self.layerCount - 1 :

                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta**2)
                delta.append(output_delta * self.sgm(self._layerInput[index],True))

            else:

                delta_pullback = self.weights[index+1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1,:] * self.sgm(self._layerInput[index],True))

        for index in range(self.layerCount):
            delta_index  = self.layerCount - 1 - index

            if index == 0:
                layerOutput  = np.vstack([input.T,np.ones([1,InCases])])
            else:
                layerOutput = np.vstack([self._layerOutput[index-1],np.ones([1,self._layerOutput[index-1].shape[1]])])

            currWeightDelta = np.sum(layerOutput[None,:,:].transpose(2,0,1) * delta[delta_index][None,:,:].transpose(2,1,0),axis = 0)

            weightDelta = trainingRate * currWeightDelta + momentum * self._previousWeightDelta[index]

            self.weights[index] -= weightDelta

            self._previousWeightDelta[index] = weightDelta

        return error

    def sgm(self,x,Derivative=False):
        if not Derivative:
            return 1/ (1+np.exp(-x))
        else:
            out = self.sgm(x)
            return out*(1-out)


if __name__ == "__main__":
    bpn = BackPropagationNetwork((260,25,25,10))

    mode = "temp"

    f1 = open("mfccData/" + mode + "/1_mfcc.npy")
    f2 = open("mfccData/" + mode + "/2_mfcc.npy")
    f3 = open("mfccData/" + mode + "/3_mfcc.npy")
    f4 = open("mfccData/" + mode + "/4_mfcc.npy")
    f5 = open("mfccData/" + mode + "/5_mfcc.npy")
    f6 = open("mfccData/" + mode + "/6_mfcc.npy")
    f7 = open("mfccData/" + mode + "/7_mfcc.npy")
    f8 = open("mfccData/" + mode + "/8_mfcc.npy")
    f9 = open("mfccData/" + mode + "/9_mfcc.npy")
    f10 = open("mfccData/" + mode + "/10_mfcc.npy")

    inputArray1  = np.load(f1)
    inputArray2  = np.load(f2)
    inputArray3  = np.load(f3)
    inputArray4  = np.load(f4)
    inputArray5  = np.load(f5)
    inputArray6  = np.load(f6)
    inputArray7  = np.load(f7)
    inputArray8  = np.load(f8)
    inputArray9  = np.load(f9)
    inputArray10  = np.load(f10)

    inputArray = np.concatenate((inputArray1,inputArray2,inputArray3,inputArray4,inputArray5,inputArray6,inputArray7,inputArray8,inputArray9,inputArray10))

    print inputArray.shape

    t1 = np.array([[1,0,0,0,0,0,0,0,0,0] for _ in range(len(inputArray1))])
    t2 = np.array([[0,1,0,0,0,0,0,0,0,0] for _ in range(len(inputArray2))])
    t3 = np.array([[0,0,1,0,0,0,0,0,0,0] for _ in range(len(inputArray3))])
    t4 = np.array([[0,0,0,1,0,0,0,0,0,0] for _ in range(len(inputArray4))])
    t5 = np.array([[0,0,0,0,1,0,0,0,0,0] for _ in range(len(inputArray5))])
    t6 = np.array([[0,0,0,0,0,1,0,0,0,0] for _ in range(len(inputArray6))])
    t7 = np.array([[0,0,0,0,0,0,1,0,0,0] for _ in range(len(inputArray7))])
    t8 = np.array([[0,0,0,0,0,0,0,1,0,0] for _ in range(len(inputArray8))])
    t9 = np.array([[0,0,0,0,0,0,0,0,1,0] for _ in range(len(inputArray9))])
    t10 = np.array([[0,0,0,0,0,0,0,0,0,1] for _ in range(len(inputArray10))])

    target = np.concatenate([t1,t2,t3,t4,t5,t6,t7,t8,t9,t10])
    print target.shape

    lnMax = 80000
    lnErr = 1e-5

    startTime = time.clock()

    for i in range(lnMax-1):
        err = bpn.train(inputArray,target,momentum = 0.3)
        if i % 1500 == 0:
            print "Iteration {0} \tError: {1:0.6f}".format(i,err)
        if err <= lnErr:
            print("Minimum error reached at iteration {0}".format(i))
            break

        endTime = time.clock()

    with open("network/" + "digit_network_words_" + mode + ".npy", 'w') as outfile:
        np.save(outfile,bpn.weights)



    lvOutput = bpn.forwardProc(inputArray)
    print("Output {0}".format(lvOutput))

    print "Time Elapsed: " + str(endTime - startTime) + " seconds"
    print "Total Iteration {0} \t Total Error: {1:0.6f}".format(i,err)
