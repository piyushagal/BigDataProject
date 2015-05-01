
words = ['1','2','3','4','5','6','7','8','9','10']

from anntester_single import *
count = 0.0
errcount = 0.0
for x in range(len(words)):
    for i in range(1,60):
        filename = "training_sets/digits/" + words[x] + "-" + str(i) + ".wav"
        print filename
        testNet = testInit()
        inputArray = extractFeature(filename)
        outStr = feedToNetwork(inputArray,testNet)
        print (outStr+1)
        count = count + 1.0
        if(str(outStr+1)!=words[x]):
            errcount = errcount + 1.0


print "Error is  :" + str((errcount)*100/count) + "%"



