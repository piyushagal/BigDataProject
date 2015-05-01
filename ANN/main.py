from Tkinter import *
from threading import Thread
from record import record_to_file
from anntester_single import *

class Application(Frame):

    def createWidgets(self):
        self.button_image = PhotoImage(file="button.gif")
        self.RECORD = Button(self, image=self.button_image, width="100", height="100", command=self.record_buttonpress)
        self.RECORD.pack()
        self.TEXTBOX = Text(self, height="1", width="30")
        self.TEXTBOX.pack()

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.pack()
        self.createWidgets()
        self.TEXTBOX.insert(INSERT, "Press to record")
        self.TEXTBOX.tag_config("recording", foreground="red", justify="center")
        self.TEXTBOX.tag_config("success", foreground="darkgreen", justify="center")
        self.TEXTBOX.configure(state="disabled")
        self.i = 1
        self.j = 61

    def record_buttonpress(self):

        recorder_thread = Thread(target=record_and_test, args=(self.TEXTBOX, self.RECORD, "training_sets/digits/" + str(self.i) + "-" + str(self.j) + ".wav"))
        recorder_thread.start()
        self.j = self.j+1
        if(self.j>70):
            self.i = self.i + 1
            self.j =  61

def record_and_test(textbox, button, filename):

    button.configure(state="disabled")
    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.insert(INSERT, "Recording")
    textbox.tag_add("recording", "1.0", END)
    textbox.configure(state="disabled")

    record_to_file(filename)

    testNet = testInit()
    inputArray = extractFeature(filename)
    print len(inputArray)
    outStr = "Detected : " + str(feedToNetwork(inputArray,testNet)+1)

    textbox.configure(state="normal")
    textbox.delete("1.0", END)
    textbox.tag_remove("recording", "1.0")
    textbox.insert(INSERT, outStr)
    textbox.tag_add("success", "1.0", END)
    textbox.configure(state="disabled")
    button.configure(state="normal")

if __name__ == '__main__':

    root = Tk()
    app = Application(master=root)
    app.mainloop()