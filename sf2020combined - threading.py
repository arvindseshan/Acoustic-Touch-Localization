import os
import pyaudio
import numpy as np
import pylab
from pylab import *
import matplotlib
import matplotlib.pyplot as plt
import time
import sys
import threading
import logging
import math

##open('leftsamples.txt', 'w').close()
##open('rightsamples.txt', 'w').close()




hit = False
hit1 = False
hit2 = False
hit3 = False
lrdone = False
uddone = False
udcounter=0
lrcounter=0
left_samples=0
right_samples=0
left_samples1=0
right_samples1=0
i=0
a=7
b=4
aold=0

def hit_test (piezo, amplitude): #left_samples, right_samples, left_samples1, right_samples1
    for x in range(len(piezo)):
        if abs(piezo[x]) >= amplitude:
            return x
    return False

def detect_tap_lr() :
    global lrdone
    global lrcounter
    global a

    lhit = False
    ltime = 0.0
    rhit = False
    rtime = 0.0

    if lrdone==False:

        for x in range(len(left_samples)):
            if abs(left_samples[x]) >= 500 and ltime == 0.0:
                ltime = x/RATE
            if abs(left_samples[x]) >= 1000:
                lhit = True
                break

        for x in range(len(right_samples)):
            if abs(right_samples[x]) >= 500 and rtime == 0.0:
                rtime = x/RATE
            if abs(right_samples[x]) >= 1000:
                rhit = True
                break


        if rhit==True and lhit==True:
                list = [ltime, rtime]
                if list.index(min(list)) == 0 :
                    ltime2=0
                    rtime2=rtime-ltime
                    location=-23596.03587*rtime2+15.28315243
                if list.index(min(list)) == 1 :
                    ltime2=ltime-rtime
                    rtime2=0
                    location=23201.85615*ltime2+14.80046404
                lrdone=True
                lrcounter=0

                print(ltime2, rtime2, location)
                print("good\n")

##                with open('leftsamples.txt','ab') as f:
##                    np.savetxt(f, left_samples, fmt='%5d', delimiter=',')
##                with open('rightsamples.txt','ab') as f:
##                    np.savetxt(f, right_samples, fmt='%5d', delimiter=',')
                a = location
    else:
        if lrcounter>4:
            lrdone=False
        lrcounter = lrcounter+1


def detect_tap_ud() :
    global udcounter
    global uddone
    global b

    lhit = False
    ltime = 0.0
    rhit = False
    rtime = 0.0

    if uddone==False:
        for x in range(len(left_samples1)):
            if abs(left_samples1[x]) >= 500 and ltime == 0.0:
                ltime = x/RATE
            if abs(left_samples1[x]) >= 1000:
                lhit = True
                break

        for x in range(len(right_samples1)):
            if abs(right_samples1[x]) >= 500 and rtime == 0.0:
                rtime = x/RATE
            if abs(right_samples1[x]) >= 1000:
                rhit = True
                break

        if lhit==True and rhit==True:
                list = [ltime, rtime]
                if list.index(min(list)) == 0 :
                    ltime2=0
                    rtime2=rtime-ltime
                    location1=-8576.329331*rtime2+5.353430532
                if list.index(min(list)) == 1 :
                    ltime2=ltime-rtime
                    rtime2=0
                    location1=9950.248756*ltime2+4.994029851
                uddone=True
                udcounter=0
##                print("up down")
##                print(ltime2, rtime2, location1)
##                print("good\n")
                b=location1

    else:
        if udcounter>4:
            uddone=False
        udcounter = udcounter+1

FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 2
RATE = 192000 #192000
CHUNK = 8192 # (8192) 1024bytes of data read from a buffer
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "file.wav"
left_channel = 0
right_channel = 1

audio = pyaudio.PyAudio()

def lr():
    while stream.is_active():
        global left_samples
        global right_samples
        raw_data = stream.read(CHUNK)
        samples  = np.fromstring(raw_data, dtype=np.int16)
        left_samples = samples[left_channel::2]
        right_samples = samples[right_channel::2]
        detect_tap_lr()
##    with open('leftsamples.txt','ab') as f:
##        np.savetxt(f, left_samples, fmt='%5d', delimiter=',')
##    with open('rightsamples.txt','ab') as f:
##        np.savetxt(f, right_samples, fmt='%5d', delimiter=',')

def ud():
    while stream1.is_active():
        global left_samples1
        global right_samples1
        raw_data1 = stream1.read(CHUNK)
        samples1  = np.fromstring(raw_data1, dtype=np.int16)
        left_samples1 = samples1[left_channel::2]
        right_samples1 = samples1[right_channel::2]
        detect_tap_ud()

def calcPos():
    global aold
    while stream.is_active() and stream1.is_active():
        xpos=((2*sqrt(b)*sqrt(-1*(math.pow(a,4)*(-b)+11*math.pow(a,4)+60*math.pow(a,3)*b-660*math.pow(a,3)+math.pow(a,2)*math.pow(b,3)-22*math.pow(a,2)*math.pow(b,2)-973.75*math.pow(a,2)*b+12042.3*math.pow(a,2)-30*a*math.pow(b,3)+660*a*math.pow(b,2)+2212.5*a*b-64267.5*a+225*math.pow(b,3)-4950*math.pow(b,2)+34031*b-74868.8)))/(sqrt(-1*(121*math.pow(a,2)-3630*a+900*math.pow(b,2)-9900*b+27225))))
        ypos=sqrt((-math.pow(a,4)+math.pow(a,2)*math.pow(xpos,2)+(60)*math.pow(a,3)-(30)*(a)*math.pow(xpos,2)-(1125)*math.pow(a,2)+(6750)*(a))/(-math.pow(a,2)+30*(a)-225))
        if aold!=a:
            print(str(a) + ", " + str(b))
            print("Coordinates: (" + str(xpos) + ", " + str(ypos) + ")")
        aold=a

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 1,
                    input=True,
                    frames_per_buffer=CHUNK)

stream1 = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 2,
                    input=True,
                    frames_per_buffer=CHUNK)

global keep_going
keep_going = True


# Open the connection and start streaming the data
stream.start_stream()
stream1.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")
if __name__ == "__main__":
        t1 = threading.Thread(target=lr)
##        t2 = threading.Thread(target=ud)
##        t3 = threading.Thread(target=calcPos)

        t1.start()
##        t2.start()
##        t3.start()

        t1.join()
##        t2.join()
##        t3.join()


stream.stop_stream()
stream.close()
stream1.stop_stream()
stream1.close()

audio.terminate()
