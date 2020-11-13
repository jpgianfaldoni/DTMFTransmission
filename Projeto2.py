import sounddevice as sd
import peakutils
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft, fftshift
import soundfile   as sf
import matplotlib.pyplot as plt
from scipy import signal as sg


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


def normalizeSignal(signal):
    maxValue = max(abs(signal))
    normalizedSignal =  [x / maxValue for x in signal]
    return(normalizedSignal)
    
def LPF(signal, cutoff_hz, fs):
        #####################
        # Filtro
        #####################
        # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
        nyq_rate = fs/2
        width = 5.0/nyq_rate
        ripple_db = 120.0 #dB
        N , beta = sg.kaiserord(ripple_db, width)
        taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
        return( sg.lfilter(taps, 1.0, signal))    

def generateSin(F, T, fs):
    n = T*fs #numero de pontos
    x = np.linspace(0.0, T, n)  # eixo do tempo
    s = np.sin(F*x*2*np.pi)
    return (x, s)


yAudio, samplerate = sf.read('recording.wav')
lpfSignal = LPF(normalizeSignal(yAudio), 4000, samplerate)
samplesAudio = len(lpfSignal)
T = samplesAudio//samplerate
sd.play(lpfSignal)
sd.wait()
# plt.plot(lpfSignal)
# plt.grid()
# plt.show()
x1,s1 = generateSin(14000, T, samplerate)
modulatedSignal = lpfSignal * s1
sd.play(modulatedSignal)
sd.wait()
deModulatedSignal = modulatedSignal * s1
sd.play(deModulatedSignal)
sd.wait()
t = np.linspace(0,T,T*samplerate)


plt.subplot(5,1,1)
plt.title('Original message')
plt.plot(t, yAudio,'g')
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,2)
plt.title('Message normalized between [-1,1]')
plt.plot(t, normalizeSignal(yAudio))
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,3)
plt.title('Message filtered to remove frequencies above 4kHz')
plt.plot(t, lpfSignal)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,4)
plt.title('Message modulating carrier of 14kHz')
plt.plot(t, modulatedSignal)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplot(5,1,5)
plt.title('Demodulated message')
plt.plot(t, deModulatedSignal)
plt.ylabel('Amplitude')
plt.xlabel('time (s)')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()
fig.set_size_inches(16, 9)

fig.savefig('Time.png', dpi=160)


dataFFT = calcFFT(yAudio, samplerate)
dataNormalizedFFT = calcFFT(normalizeSignal(yAudio), samplerate)
dataFilteredFFT = calcFFT(LPF(normalizeSignal(yAudio), 4000, samplerate),  samplerate)
dataModulatedFFT = calcFFT(modulatedSignal, samplerate)
dataDemodulatedFFT = calcFFT(deModulatedSignal, samplerate)

print("Starting plot 1")
plt.subplot(5,1,1, label="test")
plt.title('Original message')
plt.plot(dataFFT[0], dataFFT[1],'g')
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 2")
plt.subplot(5,1,2, label="test2")
plt.title('Message normalized between [-1,1]')
plt.plot(dataNormalizedFFT[0], dataNormalizedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 3")
plt.subplot(5,1,3, label="test3")
plt.title('Message filtered to remove frequencies above 4kHz')
plt.plot(dataFilteredFFT[0], dataFilteredFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

print("Starting plot 4")
plt.subplot(5,1,4, label="test4")
plt.title('Message modulating carrier of 14kHz')
plt.plot(dataModulatedFFT[0], dataModulatedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

plt.subplot(5,1,5, label="test5")
plt.title('Demodulated message')
plt.plot(dataDemodulatedFFT[0], dataDemodulatedFFT[1])
plt.ylabel('Amplitude')
plt.xlabel('frequency (Hz)')

plt.subplots_adjust(hspace=1)
plt.rc('font', size=15)
fig = plt.gcf()
fig.set_size_inches(16, 9)

fig.savefig('Fourier.png', dpi=160)