import numpy as np
import matplotlib.pyplot as plt

IQseparate = False
N = 128
M = 16
f_sub = [0,6,15,40,60]
amp_sub = [2,1,2,1,10]
s_sub = np.exp(1j * np.linspace(0,1,N) * np.array(f_sub).reshape(-1,1)) * np.array(amp_sub).reshape(-1,1)
s = s_sub.sum(axis=0)
s_sub = np.exp(1j * np.linspace(0,1,N*M) * np.array(f_sub).reshape(-1,1) * M) * np.array(amp_sub).reshape(-1,1) / M
sfft_sub = np.fft.fftshift(np.fft.fft(s_sub,N*M,axis=-1),axes=(-1,))
# s = 10*np.exp(1j*np.linspace(0,0,N)) + np.exp(1j*np.linspace(0,12,N)) + np.exp(1j*np.linspace(0,30,N)) + np.exp(1j*np.linspace(0,60,N))
sfft = np.fft.fftshift(np.fft.fft(s,N*M))
sfft_w = np.fft.fftshift(np.fft.fft(s*np.hanning(N),N*M))
sfft_w = sfft_w / abs(sfft_w).max() * abs(sfft).max() # np.sqrt(np.mean(np.hanning(N)**2))
sfft_ = sfft.copy()
for i in range(len(sfft)):
    i_plus = (i + M) % len(sfft)
    i_minus = i - M
    if IQseparate:
        w_r = -sfft[i].real / (sfft[i_minus].real + sfft[i_plus].real)
        if w_r < -0.5:
            temp_r = sfft[i].real + -0.5 * (sfft[i_minus].real + sfft[i_plus].real)
        elif w_r <= 0:
            temp_r = 0
        else:
            temp_r = sfft[i].real
        w_i = -sfft[i].imag / (sfft[i_minus].imag + sfft[i_plus].imag)
        if w_i < -0.5:
            temp_i = sfft[i].imag + -0.5 * (sfft[i_minus].imag + sfft[i_plus].imag)
        elif w_i <= 0:
            temp_i = 0
        else:
            temp_i = sfft[i].imag
        sfft_[i] = temp_r + 1j * temp_i
    else:
        w = -sfft[i] / (sfft[i_minus] + sfft[i_plus])  # -0.5
        w = w.real
        if w < -0.5:
            sfft_[i] = sfft[i] + -0.5 * (sfft[i_minus] + sfft[i_plus])
            continue
        if w <= 0:
            sfft_[i] = sfft[i] + w * (sfft[i_minus] + sfft[i_plus])
plt.figure()
plt.plot(sfft.real)
plt.plot(sfft_w.real)
plt.plot(sfft_.real)
plt.figure()
plt.plot(sfft.imag)
plt.plot(sfft_w.imag)
plt.plot(sfft_.imag)
plt.figure()
plt.plot(abs(sfft),label='FFT')
plt.plot(abs(sfft_w),label='Hanning')
plt.plot(abs(sfft_),label='SVA')
plt.plot(abs(sfft_sub.sum(axis=0)),label='GT')
plt.legend()
plt.show()