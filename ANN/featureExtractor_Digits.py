from __future__ import division
from features import mfcc
from operator import add
import scipy.io.wavfile as wav
import numpy as np

words = ['1','2','3','4','5','6','7','8','9','10']


for x in range(len(words)):
    fileString = words[x]+"_mfcc"
    data = []
    for i in range(60,70):
        (rate,sig) = wav.read("training_sets/digits/"+ words[x] + "-" + str(i+1) + ".wav")
        print "Reading: " + words[x] + "-" + str(i+1) + ".wav"
        duration = len(sig)/rate
        mfcc_feat = mfcc(sig,rate,winlen=duration/20,winstep=duration/20)
        s = mfcc_feat[:20]
        st = []
        for elem in s:
            st.extend(elem)

        st /= np.max(np.abs(st),axis=0)
        data.append(st)
        #print st

    with open("mfccData/temp/" + fileString+ ".npy", 'w') as outfile:
        np.save(outfile,data)



