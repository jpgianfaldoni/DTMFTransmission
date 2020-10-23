import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fftpack import fft, fftshift
from suaBibSignal import *
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import soundfile   as sf

def generateSin(F, T, fs):
    n = T*fs #numero de pontos
    x = np.linspace(0.0, T, n)  # eixo do tempo
    s = np.sin(F*x*2*np.pi)
    return (x, s)


print("Escolha um número de 0 a 9")
chosenNumber = int(input())


signalTable = {
    0 : [941, 1336],
    1 : [697, 1209],
    2 : [697,1336],
    3 : [697,1477],
    4 : [770,1209],
    5 : [770, 1336],
    6 : [770, 1477],
    7 : [852, 1209],
    8 : [852, 1336],
    9 : [852, 1477]
}


frequency1 = signalTable[chosenNumber][0]
frequency2 = signalTable[chosenNumber][1]

fs  = 44100   # pontos por segundo (frequência de amostragem)
A   = 2   # Amplitude
T = 1
t   = np.linspace(-T/2,T/2,T*fs)
x1,s1 = generateSin(frequency1, T, fs)
plt.plot(t,s1)
plt.xlim((0,1/100))
plt.show()

x2,s2 = generateSin(frequency2, T, fs)
plt.plot(t,s2)
plt.xlim((0,1/100))
plt.show()

finalFrequency = s1 + s2
plt.plot(t,finalFrequency)
plt.xlim((0,1/100))
plt.show()
sd.play(finalFrequency, fs)
sd.wait()