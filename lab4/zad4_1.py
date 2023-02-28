# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 20:21:40 2022

@author: maksi
"""
#podłączamy biblioteki do programu
from scipy.io import wavfile
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.fft import fft

#4.1. Algorytm szybkiej transformaty Fouerier’a.
#podajemy nazwę oraz ścieżkę do folderu z plikami .wav
path = 'voices'
files = os.listdir(path)
fs = 16000
seconds = 3
X_raw = np.zeros((len(files), fs*seconds))

#pętla utworzenia tablicy dla odczytu wszystkich plików .wav
for i, file in enumerate(files):
    X_raw[i,:] = wavfile.read(f"{path}\\{file}")[1]

#odczyt danych z pliku Genders_voices.xlsx
y = pd.read_excel('Genders_voises.xlsx').values

#4.2. Zastosowanie szybkiej transformaty Fourier'a
#wyznaczamy transformaty Fourier'a oraz deklarujemy sygnały czasowe
X_fft = np.abs(fft(X_raw, axis=-1))/X_raw.shape[1]
low_cut = 50*seconds
hight_cut = 280*seconds
X_fft_cut = X_fft[:, low_cut:hight_cut]
#generacja wykresu dla sygnału czasowego i widma amplitudy po tramsformacji Fourier'a
fig, ax = plt.subplots(2,1)
ax[0].plot(np.arange(X_raw.shape[1]), X_raw[0,:])
ax[1].scatter(np.arange(X_raw.shape[1]), X_fft[0,:], s = 0.5)
fig.tight_layout()

#4.3. Zmniejszenie rozdzielczości widma. Pozostawiamy tylko amplitudy potrzebnych częstotliwości.
#uśredniamy sygnał, zmniejszając liczbę próbek trzykrotnie, tak, by rozdzielczość wynosiła 1 Hz
mean_num = 3
X_fft = np.reshape(X_fft,(X_fft.shape[0], X_fft.shape[1]//mean_num, mean_num))
X_fft = X_fft.mean(axis=-1)
low_cut = 50
hight_cut = 280
X_fft_cut = X_fft[:, low_cut:hight_cut]
X_fft_cut = X_fft_cut/np.expand_dims(X_fft_cut.max(axis=1), axis=-1)

#generacja widma o rozdzielczości 1 Hz dla częstotliwości z przedziału [50,280] Hz
fig, ax = plt.subplots(1,1)
ax.plot(np.arange(X_fft_cut.shape[1]), X_fft_cut[0,:])
ax.set_xlabel("Częstotliwość")
ax.set_ylabel("Amplituda")
fig.tight_layout()