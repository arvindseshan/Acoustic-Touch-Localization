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
    
    import pandas as pd
    from scipy.signal import find_peaks
    import plotly.graph_objects as go
except:
    print ("Something didn't import")
list1 = []
peaks = 0
FORMAT = pyaudio.paInt16 # We use 16bit format per sample
CHANNELS = 2
RATE = 192000 #192000
CHUNK = 8192 # (8192) 1024bytes of data read from a buffer
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "file.wav"
left_channel = 0
right_channel = 1

audio = pyaudio.PyAudio()

stream1 = audio.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input_device_index = 2,
                    input=True)#,
                    #frames_per_buffer=CHUNK)
 
def findpeak() :
    global peaks
    peaks=0
    i=0
    for i in range(len(list1)) :
        if abs(list1[i]) >= 1400:
            i = i + 50000
            peaks = peaks + 1
        else :
            i= i+1

global keep_going
keep_going = True

stream1.start_stream()
print ("\n+---------------------------------+")
print ("| Press Ctrl+C to Break Recording |")
print ("+---------------------------------+\n")

while keep_going:
    try:
        raw_data1 = stream1.read(CHUNK)
        samples1  = np.fromstring(raw_data1, dtype=np.int16)
        left_samples1 = samples1[left_channel::2]
        right_samples1 = samples1[right_channel::2]
        list1.extend(left_samples1)
        if input("Done?") == "" :
            findpeak()
            print(peaks)
            list1.clear()
            
    except KeyboardInterrupt:
        keep_going=False
#    except:
#        pass

# Close up shop (currently not used because KeyboardInterrupt
# is the only way to close)
stream1.stop_stream()
stream1.close()

##stream1.stop_stream()
##stream1.close()

audio.terminate()
