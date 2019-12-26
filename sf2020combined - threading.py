try:
    import os
    import pyaudio
    import numpy as np
    import pylab
    from pylab import *
    import matplotlib
    import matplotlib.pyplot as plt
    from scipy.io import wavfile
    import time
    import sys
    import seaborn as sns
    import threading
    import logging
    import math
except:
    print ("Something didn't import")
open('leftsamples.txt', 'w').close()
open('rightsamples.txt', 'w').close()




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

##f,ax = plt.subplots(2)
##x = np.arange(10000)
##y = np.random.randn(10000)
### Plot 0 is for raw audio data
##li, = ax[0].plot(x, y)
##ax[0].set_xlim(0,2048)
##ax[0].set_ylim(-10000,10000)
##ax[0].set_title("Left")
##
### Plot 1 is for the FFT of the audio
##li2, = ax[1].plot(x, y)
##ax[1].set_xlim(0,2048)
##ax[1].set_ylim(-10000,10000)
##ax[1].set_title("Right")
##
##plt.pause(0.01)
##plt.tight_layout()


def plot_data(in_data):
    # get and convert the data to float
##    audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    # dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print

    li.set_xdata(np.arange(len(in_data)))
    li.set_ydata(in_data)
    #li2.set_xdata(np.arange(len(dfft))*10.)
    #li2.set_ydata(dfft)

def plot_data1(in_data):
    # get and convert the data to float
##    audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    # dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    
    li2.set_xdata(np.arange(len(in_data)))
    li2.set_ydata(in_data)
    #li4.set_xdata(np.arange(len(dfft))*10.)
    #li4.set_ydata(dfft) 
      
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        return True
    else:
        return False

def plot_data2(in_data):
    # get and convert the data to float
   # audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
   # dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    
    li.set_xdata(np.arange(len(in_data)))
    li.set_ydata(in_data)
    #li6.set_xdata(np.arange(len(dfft))*10.)
    #li6.set_ydata(dfft)
    
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        return True
    else:
        return False

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
            if abs(left_samples[x]) >= 500:
                lhit = True
                break

        for x in range(len(right_samples)):
            if abs(right_samples[x]) >= 500 and rtime == 0.0:
                rtime = x/RATE
            if abs(right_samples[x]) >= 500:
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

##def detect_tap_all() :
##    global hit
##    global hit1
##    global hit2
##    global hit3
##    global ltime
##    global rtime
##    global l1time
##    global r1time
##    global ltime2
##    global rtime2
##    global l1time2
##    global r1time2
##    global location
##    global counter
##    global done
##
##    if done==False:
##        if hit_test(left_samples, 1000) != False :
##            ltime = ((hit_test(left_samples, 500))/RATE)
##            hit = True
##        if hit_test(right_samples, 1000) != False :
##            rtime = ((hit_test(right_samples, 500))/RATE)
##            hit1 = True
##        if hit_test(left_samples1, 1000) != False :
##            l1time = ((hit_test(left_samples1, 500))/RATE)
##            hit2 = True
##        if hit_test(right_samples1, 1000) != False :
##            r1time = ((hit_test(right_samples1, 500))/RATE)
##            hit3 = True
##        if hit==True and hit1==True and hit2==True and hit3==True:
##                list = [ltime, rtime, l1time, r1time]
##                if list.index(min(list)) == 0 :
##                    ltime2=0
##                    rtime2=rtime-ltime
##                    l1time2=l1time-ltime
##                    r1time2=r1time-ltime
####                    location=-23596.03587*rtime2+15.28315243
##                if list.index(min(list)) == 1 :
##                    ltime2=ltime-rtime
##                    rtime2=0
##                    l1time2=l1time-rtime
##                    r1time2=r1time-rtime
####                    location=23201.85615*ltime2+14.80046404
##                if list.index(min(list)) == 2 :
##                    ltime2=ltime-l1time
##                    rtime2=rtime-l1time
##                    l1time2=0
##                    r1time2=r1time-l1time
####                    location=23201.85615*ltime2+14.80046404
##                if list.index(min(list)) == 3 :
##                    ltime2=ltime-r1time
##                    rtime2=rtime-r1time
##                    l1time2=l1time-r1time
##                    r1time2=0
####                    location=23201.85615*ltime2+14.80046404
##                
##                hit = False
##                hit1 = False
##                hit2 = False
##                hit3 = False
##                done=True
##                counter=0
##                print(ltime2, rtime2, l1time2, r1time2)
##                print(hit_test(left_samples, 500), hit_test(right_samples, 500), hit_test(left_samples1, 500), hit_test(right_samples1, 500))
##                print("good")
####                plot_data(left_samples)
####                plot_data1(right_samples)
##    else:
##        if counter>4:
##            done=False
##        counter = counter+1


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

def callback(in_data, frame_count, time_info, status):
    global left_samples
    global right_samples
    samples  = np.fromstring(in_data, dtype=np.int16)
    left_samples = samples[left_channel::2]
    right_samples = samples[right_channel::2]
    detect_tap_lr()
##    with open('leftsamples.txt','ab') as f:
##        np.savetxt(f, left_samples, fmt='%5d', delimiter=',')
##    with open('rightsamples.txt','ab') as f:
##        np.savetxt(f, right_samples, fmt='%5d', delimiter=',')
    return (None, pyaudio.paContinue)

def callback1(in_data, frame_count, time_info, status):
    global left_samples1
    global right_samples1
    samples1  = np.fromstring(in_data, dtype=np.int16)
    left_samples1 = samples1[left_channel::2]
    right_samples1 = samples1[right_channel::2]
    detect_tap_ud()
    return (None, pyaudio.paContinue)

# start Recording
stream = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 1,
                    input=True,
                    frames_per_buffer=CHUNK,

stream1 = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 2,
                    input=True,
                    frames_per_buffer=CHUNK,


global keep_going
keep_going = True


# Open the connection and start streaming the data
stream.start_stream()
stream1.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")
if __name__ == "__main__": 
    while stream.is_active() and stream1.is_active():
        t1 = threading.Thread(target=callback)
        t2 = threading.Thread(target=callback1)
        # starting thread 1 
        t1.start() 
        # starting thread 2 
        t2.start() 
##    xpos=((2*sqrt(b)*sqrt(-1*(math.pow(a,4)*(-b)+11*math.pow(a,4)+60*math.pow(a,3)*b-660*math.pow(a,3)+math.pow(a,2)*math.pow(b,3)-22*math.pow(a,2)*math.pow(b,2)-973.75*math.pow(a,2)*b+12042.3*math.pow(a,2)-30*a*math.pow(b,3)+660*a*math.pow(b,2)+2212.5*a*b-64267.5*a+225*math.pow(b,3)-4950*math.pow(b,2)+34031*b-74868.8)))/(sqrt(-1*(121*math.pow(a,2)-3630*a+900*math.pow(b,2)-9900*b+27225))))
##    ypos=sqrt((-math.pow(a,4)+math.pow(a,2)*math.pow(xpos,2)+(60)*math.pow(a,3)-(30)*(a)*math.pow(xpos,2)-(1125)*math.pow(a,2)+(6750)*(a))/(-math.pow(a,2)+30*(a)-225))
##    if aold!=a:
##        print(str(a) + ", " + str(b))
##        print("Coordinates: (" + str(xpos) + ", " + str(ypos) + ")")
##    aold=a
stream.stop_stream()
stream.close()
stream1.stop_stream()
stream1.close()

audio.terminate()
