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
    
open('leftsamples1.txt', 'w').close()
    



hit = False
hit1 = False
hit2 = False
hit3 = False
done = False
counter=0
    

i=0
##f,ax = plt.subplots(8)
##
### Prepare the Plotting Environment with random starting values
##x = np.arange(10000)
##y = np.random.randn(10000)
##
### Plot 0 is for raw audio data
##li, = ax[0].plot(x, y)
##ax[0].set_xlim(0,1000)
##ax[0].set_ylim(-5000,5000)
##ax[0].set_title("Raw Audio Signal")
##
##### Plot 1 is for the FFT of the audio
####li2, = ax[1].plot(x, y)
####ax[1].set_xlim(0,5000)
####ax[1].set_ylim(-100,100)
####ax[1].set_title("Fast Fourier Transform")
##
### Plot 2
##li3, = ax[2].plot(x, y)
##ax[2].set_xlim(0,1000)
##ax[2].set_ylim(-5000,5000)
##ax[2].set_title("Raw Audio Signal1")
##
##### Plot 3
####li4, = ax[3].plot(x, y)
####ax[3].set_xlim(0,5000)
####ax[3].set_ylim(-100,100)
####ax[3].set_title("Fast Fourier Transform1")
##
### Plot 4
##li5, = ax[4].plot(x, y)
##ax[4].set_xlim(0,1000)
##ax[4].set_ylim(-5000,5000)
##
##ax[4].set_title("Raw Audio Signal2")
##### Plot 5
####li6, = ax[5].plot(x, y)
####ax[5].set_xlim(0,5000)
####ax[5].set_ylim(-100,100)
####ax[5].set_title("Fast Fourier Transform2")
##
### Plot 6
##li7, = ax[6].plot(x, y)
##ax[6].set_xlim(0,1000)
##ax[6].set_ylim(-5000,5000)
##ax[6].set_title("Raw Audio Signal3")
##
##### Plot 7
####li8, = ax[7].plot(x, y)
####ax[7].set_xlim(0,5000)
####ax[7].set_ylim(-100,100)
####ax[7].set_title("Fast Fourier Transform3")
##
##
### Show the plot, but without blocking updates
##plt.pause(0.01)
##plt.tight_layout()


##f,ax = plt.subplots(2)
##x = np.arange(10000)
##y = np.random.randn(10000)
### Plot 0 is for raw audio data
##li, = ax[0].plot(x, y)
##ax[0].set_xlim(0,8192)
##ax[0].set_ylim(-4000,4000)
##ax[0].set_title("Left")
##
### Plot 1 is for the FFT of the audio
##li2, = ax[1].plot(x, y)
##ax[1].set_xlim(0,8192)
##ax[1].set_ylim(-4000,4000)
##ax[1].set_title("Right")
##
##plt.pause(0.01)
##plt.tight_layout()


FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 2
RATE = 192000 #192000
CHUNK = 8192 # (8192) 1024bytes of data read from a buffer
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "file.wav"
left_channel = 0
right_channel = 1

audio = pyaudio.PyAudio()

# start Recording
##stream = audio.open(format=FORMAT,
##                    channels=CHANNELS,
##                    rate=RATE,
##                    input_device_index = 1,
##                    input=True)#,
##                    #frames_per_buffer=CHUNK)

stream1 = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 2,
                    input=True)#,
                    #frames_per_buffer=CHUNK)


global keep_going
keep_going = True


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
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        return True
    else:
        return False

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


def plot_data3(in_data):
    # get and convert the data to float
  #  audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    #dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    
    li3.set_xdata(np.arange(len(in_data)))
    li3.set_ydata(in_data)
    #li8.set_xdata(np.arange(len(dfft))*10.)
    #li8.set_ydata(dfft)
    
    
def plot_data4(in_data):
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
    
    li5.set_xdata(np.arange(len(in_data)))
    li5.set_ydata(in_data)
    #li6.set_xdata(np.arange(len(dfft))*10.)
    #li6.set_ydata(dfft)


def plot_data5(in_data):
    # get and convert the data to float
  #  audio_data = np.fromstring(in_data, np.int16)
    # Fast Fourier Transform, 10*log10(abs) is to scale it to dB
    # and make sure it's not imaginary
    #dfft = 10.*np.log10(abs(np.fft.rfft(audio_data)))

    # Force the new data into the plot, but without redrawing axes.
    # If uses plt.draw(), axes are re-drawn every time
    #print audio_data[0:10]
    #print dfft[0:10]
    #print
    
    li7.set_xdata(np.arange(len(in_data)))
    li7.set_ydata(in_data)
    #li8.set_xdata(np.arange(len(dfft))*10.)
    #li8.set_ydata(dfft)
    
    # Show the updated plot, but without blocking
    plt.pause(0.01)
    if keep_going:
        return True
    else:
        return False

def hit_test (piezo, amplitude): #left_samples, right_samples, left_samples1, right_samples1
    for x in range(len(piezo)):
        if abs(piezo[x]) >= amplitude:
            
##            print("yay")
##            print(piezo, amplitude)
            return x
    return False    

##    if (sum(piezo) / len(piezo)) >= amplitude : 
##        print("yay")
##        print(piezo, amplitude)
##        return True
##    else:
##        print("boo")
##        return False

def detect_tap() :
    global hit
    global hit1
    global hit2
    global hit3
    global ltime
    global l1time
    global rtime
    global r1time
    global timeInitial
    global timeInitial1
    if hit==False and hit1==False and hit2==False and hit3==False:
        if hit_test(left_samples, 4000) != False or hit_test(right_samples, 4000)  != False or hit_test(left_samples1, 4000)  != False or hit_test(right_samples1, 4000) != False:
            timeInitial1 = time.time()
            if hit_test(left_samples, 4000)  != False:
                timeInitial = hit_test(left_samples, 2000)
                ltime = 0
                hit = True
                hit1 = False
                hit2 = False
                hit3 = False

            if hit_test(right_samples, 4000)  != False:
                timeInitial = hit_test(right_samples, 2000)
                rtime = 0
                hit1 = True
                hit = False
                hit2 = False
                hit3 = False

            if hit_test(left_samples1, 4000)  != False:
                timeInitial = hit_test(left_samples1, 2000)
                l1time = 0
                hit2 = True
                hit = False
                hit1 = False
                hit3 = False

            if hit_test(right_samples1, 4000) != False:
                timeInitial = hit_test(right_samples1, 2000)
                r1time = 0
                hit3 = True
                hit = False
                hit1 = False
                hit2 = False
                
##            while hit == False or hit1 == False or hit2 == False or hit3 == False:
##                if hit_test(left_samples, 500)  == True and hit == False:
##                    ltime = time.time() - timeInitial
##                    hit = True
##                if hit_test(right_samples, 500)  == True and hit1 == False:
##                    rtime = time.time() - timeInitial
##                    hit1 = True
##                if hit_test(left_samples1, 500)  == True and hit2 == False:
##                    l1time = time.time() - timeInitial
##                    hit2 = True
##                if hit_test(right_samples1, 500)  == True and hit3 == False:
##                    r1time = time.time() - timeInitial
##                    hit3 = True
        
            
    else:
         if hit_test(left_samples, 4000)  != False and hit == False:
##            ltime = time.time() - timeInitial
            ltime = ((hit_test(left_samples, 2000)-timeInitial)/RATE)
            hit = True
         if hit_test(right_samples, 4000)  != False and hit1 == False:
##            rtime = time.time() - timeInitial
            rtime = ((hit_test(right_samples, 2000)-timeInitial)/RATE)
            hit1 = True
         if hit_test(left_samples1, 4000)  != False and hit2 == False:
##            l1time = time.time() - timeInitial
            l1time = ((hit_test(left_samples1, 2000)-timeInitial)/RATE)
            hit2 = True
         if hit_test(right_samples1, 4000)  != False and hit3 == False:
##            r1time = time.time() - timeInitial
            r1time = ((hit_test(right_samples1, 2000)-timeInitial)/RATE)
            hit3 = True
         if hit==True and hit1==True and hit2==True and hit3==True:
            hit = False
            hit1 = False
            hit2 = False
            hit3 = False
            print(ltime, rtime, l1time, r1time)
            print("good")
##            return True
         if time.time() - timeInitial1 >= 1:
            hit = False
            hit1 = False
            hit2 = False
            hit3 = False


def detect_tap_new() :
    global hit
    global hit1
    global hit2
    global hit3
    global ltime
    global l1time
    global rtime
    global r1time
    global ltime2
    global l1time2
    global rtime2
    global r1time2
    if hit_test(left_samples, 4000) != False :
        ltime = ((hit_test(left_samples, 2000))/RATE)
        hit = True
    if hit_test(right_samples, 4000) != False :
        rtime = ((hit_test(right_samples, 2000))/RATE)
        hit1 = True
    if hit_test(left_samples1, 4000) != False :
        l1time = ((hit_test(left_samples1, 2000))/RATE)
        hit2 = True
    if hit_test(right_samples1, 4000) != False :
        r1time = ((hit_test(right_samples1, 2000))/RATE)
        hit3 = True
    if hit==True and hit1==True and hit2==True and hit3==True:
            list = (ltime, rtime, l1time, r1time)
            if list.index(min(list)) == 0 :
                ltime2=0
                rtime2=rtime-ltime
                l1time2=l1time-ltime
                r1time2=r1time-ltime
            if list.index(min(list)) == 1 :
                ltime2=ltime-rtime
                rtime2=0
                l1time2=l1time-rtime
                r1time2=r1time-rtime
            if list.index(min(list)) == 2 :
                ltime2=ltime-l1time
                rtime2=rtime-l1time
                l1time2=0
                r1time2=r1time-l1time
            if list.index(min(list)) == 3 :
                ltime2=ltime-r1time
                rtime2=rtime-r1time
                l1time2=l1time-r1time
                r1time2=0
            hit = False
            hit1 = False
            hit2 = False
            hit3 = False
            print(ltime2, rtime2, l1time2, r1time2)
            print("good")

def detect_tap_lr() :
    global hit
    global hit1
    global ltime
    global rtime
    global ltime2
    global rtime2
    global location
    global counter
    global done

    if done==False:
        if hit_test(left_samples, 1000) != False :
            ltime = ((hit_test(left_samples, 500))/RATE)
            hit = True
        if hit_test(right_samples, 1000) != False :
            rtime = ((hit_test(right_samples, 500))/RATE)
            hit1 = True
        if hit==True and hit1==True:
                list = [ltime, rtime]
                if list.index(min(list)) == 0 :
                    ltime2=0
                    rtime2=rtime-ltime
                    location=-23596.03587*rtime2+15.28315243
                if list.index(min(list)) == 1 :
                    ltime2=ltime-rtime
                    rtime2=0
                    location=23201.85615*ltime2+14.80046404
                hit = False
                hit1 = False
                done=True
                counter=0
                print(ltime2, rtime2, location)
                print(hit_test(left_samples, 500), hit_test(right_samples, 500))
                print("good")
##                plot_data(left_samples)
##                plot_data1(right_samples)
    else:
        if counter>4:
            done=False
        counter = counter+1

def detect_tap_all() :
    global hit
    global hit1
    global hit2
    global hit3
    global ltime
    global rtime
    global l1time
    global r1time
    global ltime2
    global rtime2
    global l1time2
    global r1time2
    global location
    global counter
    global done

    if done==False:
        if hit_test(left_samples, 1000) != False :
            ltime = ((hit_test(left_samples, 500))/RATE)
            hit = True
        if hit_test(right_samples, 1000) != False :
            rtime = ((hit_test(right_samples, 500))/RATE)
            hit1 = True
        if hit_test(left_samples1, 1000) != False :
            l1time = ((hit_test(left_samples1, 500))/RATE)
            hit2 = True
        if hit_test(right_samples1, 1000) != False :
            r1time = ((hit_test(right_samples1, 500))/RATE)
            hit3 = True
        if hit==True and hit1==True and hit2==True and hit3==True:
                list = [ltime, rtime, l1time, r1time]
                if list.index(min(list)) == 0 :
                    ltime2=0
                    rtime2=rtime-ltime
                    l1time2=l1time-ltime
                    r1time2=r1time-ltime
##                    location=-23596.03587*rtime2+15.28315243
                if list.index(min(list)) == 1 :
                    ltime2=ltime-rtime
                    rtime2=0
                    l1time2=l1time-rtime
                    r1time2=r1time-rtime
##                    location=23201.85615*ltime2+14.80046404
                if list.index(min(list)) == 2 :
                    ltime2=ltime-l1time
                    rtime2=rtime-l1time
                    l1time2=0
                    r1time2=r1time-l1time
##                    location=23201.85615*ltime2+14.80046404
                if list.index(min(list)) == 3 :
                    ltime2=ltime-r1time
                    rtime2=rtime-r1time
                    l1time2=l1time-r1time
                    r1time2=0
##                    location=23201.85615*ltime2+14.80046404
                
                hit = False
                hit1 = False
                hit2 = False
                hit3 = False
                done=True
                counter=0
                print(ltime2, rtime2, l1time2, r1time2)
                print(hit_test(left_samples, 500), hit_test(right_samples, 500), hit_test(left_samples1, 500), hit_test(right_samples1, 500))
                print("good")
##                plot_data(left_samples)
##                plot_data1(right_samples)
    else:
        if counter>4:
            done=False
        counter = counter+1


def detect_tap_lr1() :
    global hit
    global hit1
    global ltime
    global rtime
    global ltime2
    global rtime2
    global location
    global counter
    global done

    if done==False:
        if hit_test(left_samples1, 1000) != False :
            ltime = ((hit_test(left_samples1, 500))/RATE)
            hit = True
        if hit_test(right_samples1, 1000) != False :
            rtime = ((hit_test(right_samples1, 500))/RATE)
            hit1 = True
        if hit==True and hit1==True:
                list = [ltime, rtime]
                if list.index(min(list)) == 0 :
                    ltime2=0
                    rtime2=rtime-ltime
                    location=-8576.329331*rtime2+5.353430532
                if list.index(min(list)) == 1 :
                    ltime2=ltime-rtime
                    rtime2=0
                    location=9950.248756*ltime2+4.994029851
                hit = False
                hit1 = False
                done=True
                counter=0
                print("up down")
                print(ltime2, rtime2, location)
                print(hit_test(left_samples1, 500), hit_test(right_samples1, 500))
                print("good")
##                plot_data(left_samples1)
##                plot_data1(right_samples1)

                 
    else:
        if counter>4:
            done=False
        counter = counter+1


# Open the connection and start streaming the data
##stream.start_stream()
stream1.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")



# Loop so program doesn't end while the stream callback's
# itself for new data
while keep_going:
    try:
        # data1 = stream.read(CHUNK)     
        # plot_data(data1)
        # data2 = stream1.read(CHUNK)
        # plot_data1(data2)

        # When reading from our 16-bit stereo stream, we receive 4 characters (0-255) per
        # sample. To get them in a more convenient form, numpy provides
        # fromstring() which will for each 16 bits convert it into a nicer form and
        # turn the string into an array.
##        raw_data = stream.read(CHUNK) # always read a whole buffer.
        raw_data1 = stream1.read(CHUNK) # always read a whole buffer.
##        samples  = np.fromstring(raw_data, dtype=np.int16)
##        # Normalize by int16 max (32767) for convenience, also converts everything to floats
##        # normed_samples = samples / float(np.iinfo(np.int16).max)
##        # split out the left and right channels to return separately.
##        # audio data is stored [left-val1, right-val1, left-val2, right-val2, ...]
##        # so just need to partition it out.
##        left_samples = samples[left_channel::2]
##        right_samples = samples[right_channel::2]

       ## detect_tap_lr()
        

        samples1  = np.fromstring(raw_data1, dtype=np.int16)
        left_samples1 = samples1[left_channel::2]
        right_samples1 = samples1[right_channel::2]
        
##        plot_data(left_samples1)
##        plot_data1(right_samples1)
        
##        plot_data2(left_samples)
##        plot_data3(right_samples)
##        plot_data4(left_samples1)
##        plot_data5(right_samples1)
        with open('leftsamples1.txt','ab') as f:
            np.savetxt(f, left_samples1, fmt='%5d', delimiter=',')

        detect_tap_lr1()
##        print(detect_tap())
##        if detect_tap() == True:
##            print("cool!")
##            print(ltime, rtime, l1time, r1time)

        
        
    except KeyboardInterrupt:
        keep_going=False
#    except:
#        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
##stream.stop_stream()
##stream.close()

stream1.stop_stream()
stream1.close()

audio.terminate()
