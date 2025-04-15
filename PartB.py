import numpy as np
import math
import matplotlib.pyplot as plt
import scipy.fftpack
import random
from mpl_toolkits.mplot3d import Axes3D



def dtft(signal,srate):
    pnts = len(signal)  # number of time points
    fourTime = np.array(range(0, pnts)) / pnts
    fCoefs = np.zeros((len(signal)), dtype=complex)

    for fi in range(0, pnts):
        # create complex sine wave
        csw = np.exp(-1j * 2 * np.pi * fi * fourTime)
        # compute dot product between sine wave and signal
        # these are called the Fourier coefficients
        fCoefs[fi] = np.sum(np.multiply(signal, csw)) / pnts

    # extract amplitudes
    hz_len = math.floor(pnts / 2.) + 1
    ampls = np.abs(fCoefs[:hz_len] / pnts)
    ampls[1:] = 2 * ampls[1:]

    # compute frequencies vector
    hz = np.linspace(0, srate / 2, num=hz_len)

    return hz, ampls

def generate_signal(t,freq,amp,phases=None):
    if phases is None:
        phases = [0] * len(freq)

    signal = np.zeros_like(t, dtype=float)
    for f, A, phi in zip(freq, amp, phases):
        signal += A * np.sin(2 * np.pi * f * t + phi)
    return signal

def generate_cos_sin_signal(t,sin_freqs,sin_amps,cos_freqs,cos_amps):
    signal = np.zeros_like(t, dtype=float)
    for f, A in zip(sin_freqs, sin_amps):
        signal += A * np.sin(2 * np.pi * f * t)
    for f, A in zip(cos_freqs, cos_amps):
        signal += A * np.cos(2 * np.pi * f * t)
    return signal

# simulation parameters
srate = 1000 
duration = 2  
time = np.arange(0, duration, 1/srate) 
freq = [3,6]
amp = [6,3]

# generate signal
#6sin(2π3t) + 3sin(2π6t)
signal1 = generate_signal(time, freq, amp)
hz1, ampls1 = dtft(signal1, srate)

#sin(2π3t) + cos(2π6t)
signal2 = generate_cos_sin_signal(time, sin_freqs=[3], sin_amps=[1], cos_freqs=[6], cos_amps=[1])
hz2, ampls2 = dtft(signal2, srate)

plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.stem(hz1, ampls1)
plt.title("DTFT of 6sin(2π3t) + 3sin(2π6t)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 10)
plt.grid(True)

plt.subplot(2, 1, 2)
plt.stem(hz2, ampls2)
plt.title("DTFT of sin(2π3t) + cos(2π6t)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(0, 10)
plt.grid(True)

plt.tight_layout()
plt.show()


threshold1 = .001 
threshold2 = 0.0003
peaks1 = np.where(ampls1 > threshold1)[0]
peaks2 = np.where(ampls2 > threshold2)[0]

print("Signal 1 Frequencies (Hz):", hz1[peaks1])
print("Signal 2 Frequencies (Hz):", hz2[peaks2])