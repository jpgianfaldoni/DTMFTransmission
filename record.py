import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavf

T = 5
fs = 44100
t = np.linspace(0, T, T*fs)

rec = sd.rec(int(T*fs), fs, channels = 1)
sd.wait()

wavf.write("C:/Users/jpgia/Desktop/DTMFTransmission/recording.wav", fs, rec)