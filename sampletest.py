import sounddevice as sd

samplerates = 32000, 44100, 48000, 96000, 128000
device = 0

supported_samplerates = []
for fs in samplerates:
    try:
        sd.check_output_settings(device=device, samplerate=fs)
    except Exception as e:
        print(fs, e)
    else:
        supported_samplerates.append(fs)
print(supported_samplerates)
