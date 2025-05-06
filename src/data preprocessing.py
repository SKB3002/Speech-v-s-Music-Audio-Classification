import os
import numpy as np
import pandas as pd
from tqdm import tqdm

data = []

for path in tqdm(audio_file):
    label = "speech" if "speech_wav" in path.lower() else "music"
    y, sr = lb.load(path, sr=None)
    #y = y[:sr * 10]  # Trim to 10 seconds
    y = y / np.max(np.abs(y))  # Normalize

    # MFCC
    mfcc = lb.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Chroma
    chroma = lb.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    delta1 = lb.feature.delta(mfcc, order = 2)
    delta_mean = np.mean(delta1, axis=1)
