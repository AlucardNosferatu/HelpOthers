# -*- coding: utf-8 -*-

import threading
import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt 

CHUNK = 512
data_temp = np.zeros((CHUNK,))
processed = np.zeros((CHUNK,))
p = pyaudio.PyAudio()

class Listener (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        Monitor()
        
class Printer (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        Painter()

class Inverter (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        Process()

class Player (threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
    def run(self):
        Play()

def Play():
    global processed
    global CHUNK
    global p
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    streamb = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    output=True,
                    frames_per_buffer=CHUNK)
    while(True):
        streamb.write(processed.tostring())
    

def Process():
    global data_temp
    global processed
    while(True):
        processed = np.negative(data_temp)

def Painter():
    global data_temp
    global CHUNK
    data_range = np.zeros((CHUNK*100,))
    plt.ion()
    while(True):
        data_range = np.hstack((data_range,data_temp))
        data_range = data_range[CHUNK:CHUNK*101]
        plt.clf()
        plt.plot(data_range)
        plt.pause(0.02)
    plt.ioff()
    plt.show()

def Monitor():
    global data_temp
    global CHUNK
    global p
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 48000
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    while(True):
        temp = stream.read(CHUNK)
        data_temp = np.fromstring(temp, dtype=np.short)

if __name__ == '__main__':
    P = Printer()
    L = Listener()
    I =Inverter()
    P2 = Player()
    P.start()
    L.start()
    I.start()
    P2.start()

    
    
