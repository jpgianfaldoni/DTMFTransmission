import sounddevice as sd
import peakutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N  = len(signal)
    T  = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, fftshift(yf))

fs  = 44100
duration = 3  # seconds
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
ymyrecording = myrecording[:,0]
sd.wait()
X, Y = calcFFT(ymyrecording,fs)
plt.figure()
plt.plot(X,np.abs(Y))
plt.show()


index = peakutils.indexes(np.abs(Y), thres=0.2, min_dist=10)
print("index de picos {}" .format(index))
for freq in X[index]:
    print("freq de pico sao {}" .format(freq))