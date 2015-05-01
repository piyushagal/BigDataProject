from __future__ import division
import numpy as np
import scipy.io.wavfile as wav

from features import mfcc


class TestingNetwork:

    layerCount = 0;
    shape  = None;
    weights = [];

    def __init__(self,layerSize,weights):

        self.layerCount = len(layerSize) - 1;
        self.shape = layerSize

        self._layerInput = []
        self._layerOutput = []
        self.weights = weights
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

    def sgm(self,x,Derivative=False):
        if not Derivative:
            return 1/ (1+np.exp(-x))
        else:
            out = self.sgm(x)
            return out*(1-out)


def testInit():
    f1 = open("network/Hidden_2/digit_network_words_4Train.npy", "rb")
    weights  = np.load(f1)
    testNet = TestingNetwork((260,25,25,10),weights)
    return testNet

def extractFeature(soundfile):
    (rate,sig) = wav.read(soundfile)
    duration = len(sig)/rate;
    mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
    print "MFCC Feature Length: " + str(len(mfcc_feat))
    s = mfcc_feat[:20]
    st = []
    for elem in s:
        st.extend(elem)
    st /= np.max(np.abs(st),axis=0)
    inputArray = np.array([st])
    return inputArray

def feedToNetwork(inputArray,testNet):
    outputArray = testNet.forwardProc(inputArray)



    indexMax = outputArray.argmax(axis = 1)[0]

    print outputArray

    outStr = None

    if indexMax == 0:
        outStr  = "Detected: 1";
    elif indexMax==1:
        outStr  = "Detected: 2";
    elif indexMax==2:
        outStr  = "Detected: 3";
    elif indexMax==3:
        outStr  = "Detected: 4";
    elif indexMax==4:
        outStr  = "Detected: 5";
    elif indexMax==5:
        outStr  = "Detected: 6";
    elif indexMax==6:
        outStr  = "Detected: 7";
    elif indexMax==7:
        outStr  = "Detected: 8";
    elif indexMax==8:
        outStr  = "Detected: 9";
    elif indexMax==9:
        outStr  = "Detected: 10";


    print outStr
    return indexMax

if __name__ == "__main__":

    testNet = testInit()
    inputArray = extractFeature("test_files/test.wav")
    feedToNetwork(inputArray,testNet)
