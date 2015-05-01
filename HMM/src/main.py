__author__ = 'piyush'
import sys
from Tkinter import *
from Record import Recorder
from learner import *
import cPickle

class Window:
    def startWindow(self, l):
        self.root = Tk()
        self.root.geometry('350x100')
        self.learn = l
        frame1 = Frame(self.root)

        b = Button(frame1, text = 'Record', command = self.recordNdetect)
        self.l = Label(frame1, text=' ')
        self.l.pack(side=BOTTOM)
        b.pack(side=TOP)
        frame1.pack()
        self.root.mainloop()
        with open('hmmModel.pkl','wb') as fid:
            cPickle.dump(self.learn, fid)

    def stopWindow(self):
        self.root.destroy()

    def setLabel(self, value):
        self.l.configure(text = value)

    def recordNdetect(self):
        r = Recorder()
        data = r.record()
        r.save(data,"./TestData1/output.wav")
        self.speech = Speech('./TestData1/', 'output.wav')
        self.speech.extractFeature()
        self.setLabel("Detected : " + self.recognize())

    def onlineLearn(self, id):
        self.speech.categoryId = int(id)-1
        self.learn.speechRecognizerList[int(id)-1].trainData.append(self.speech.features)
        self.learn.speechRecognizerList[int(id)-1].hmmModel.fit(self.learn.speechRecognizerList[int(id)-1].trainData)
        with open('hmmModel.pkl','wb') as fid:
            cPickle.dump(self.learn, fid)
        print('Done Training')


    def recognize(self):
        scores = []
        for recognizer in self.learn.speechRecognizerList:
            score = recognizer.hmmModel.score(self.speech.features)
            scores.append(score)

        idx1 = scores.index(max(scores))
        scores[idx1] = -10000
        idx2 = scores.index(max(scores))
        predictCategoryId = self.learn.speechRecognizerList[idx1].categoryId

        return predictCategoryId

print(__name__)
if __name__  ==  '__main__':
    print('started')
    if os.path.exists('hmmModel.pkl'):
        with open('hmmModel.pkl','r') as fid:
             l = cPickle.load(fid)
    else:
        l = learn()
   # l = learn()
    w = Window()
    w.startWindow(l)




