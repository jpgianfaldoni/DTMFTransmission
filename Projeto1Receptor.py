import sounddevice as sd
import peakutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
def generateSin(F, T, fs):
    n = T*fs #numero de pontos
    x = np.linspace(0.0, T, n)  # eixo do tempo
    s = np.sin(F*x*2*np.pi)
    return (x, s)

def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    #y  = np.append(signal, np.zeros(len(signal)*fs))
    N  = len(signal)
    T  = 1/fs
    xf = np.linspace(-1.0/(2.0*T), 1.0/(2.0*T), N)
    yf = fft(signal)
    return(xf, fftshift(yf))

invertedSignalTable = {
    (941, 1336) : 0,
    (697, 1209) : 1,
    (697,1336) : 2,
    (697,1477) : 3,
    (770,1209) : 4, 
    (770, 1336) : 5,
    (770, 1477) : 6, 
    (852, 1209) : 7,
    (852, 1336) : 8,
    (852, 1477) : 9
    
}


fs  = 44100
duration = 3  # seconds
t  = np.linspace(-duration/2,duration/2,duration*fs)
myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
ymyrecording = myrecording[:,0]
sd.wait()
x,s = generateSin(ymyrecording, duration, fs)
plt.plot(t,s)
plt.xlim((0,1/100))
plt.show()
X, Y = calcFFT(ymyrecording,fs)
plt.figure()
plt.plot(X,np.abs(Y))
plt.show()


index = peakutils.indexes(np.abs(Y), thres=0.2, min_dist=10)
print("index de picos {}" .format(index))
frequencies = []
allFrequencies = [697, 770, 852, 941, 1209, 1336, 1477, 1633]
for freq in X[index]:
    if freq > 0:
        for freq2 in allFrequencies:
            if (freq2 - 2) <= freq <= (freq2 + 2):
                frequencies.append(freq2)
    print("freq de pico sao {}" .format(freq))

frequenciesTuple = (frequencies[0], frequencies[1])
print("NUMERO ESCOLHIDO:",  invertedSignalTable[frequenciesTuple])