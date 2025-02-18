import pandas as pd
import numpy as np
import re
import librosa
from tqdm import tqdm
import os
import seaborn as sns
import matplotlib.pyplot as plt
import librosa

def extract_mfcc(file_path, n_mfcc=20):
    try:
        # Cargar el archivo de audio
        y, sr = librosa.load(file_path, sr=None)

        # Extraer MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        # Calcular la media de cada coeficiente MFCC
        mfcc_means = np.mean(mfccs, axis=1)

        return mfcc_means
    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return np.full(n_mfcc, np.nan)

def extract_audio_features(file_path, n_mfcc=20):
    try:
        y, sr = librosa.load(file_path, sr=None)

        # Chroma
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)

        # Spectral Contrast
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)

        # Zero Crossing Rate
        zero_crossing = np.mean(librosa.feature.zero_crossing_rate(y), axis=1)

        # Spectral Rolloff
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr), axis=1)

        # Tempo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)


        return np.hstack([chroma, spectral_contrast, zero_crossing, rolloff, tempo])

    except Exception as e:
        print(f"Error procesando {file_path}: {e}")
        return np.full(12 + 7 + 1 + 1 + 1, np.nan)  # Rellenar con NaN en caso de error



canciones = pd.read_csv('../data/raw.csv')
canciones["filepath"] = canciones["filepath"].str.replace(".ogg", ".wav")
canciones["filepath"] = canciones.apply(lambda row: row["filepath"].replace("train", str(row["genre_id"])), axis=1)
canciones["filepath"] = canciones["filepath"].apply(lambda x: re.sub(r"/0+", "/", x))
canciones.loc[16360] = [0, '000000.ogg', '0/0.wav', 0, 'Electronic']

AUDIO_BASE_PATH = "/content/drive/MyDrive/Bootcamp_DS/ML/TRAIN_V2/data_out_2/"

mfcc_features = []
for _, row in tqdm(canciones.iterrows(), total=len(canciones)):
    file_path = os.path.join(AUDIO_BASE_PATH, row["filepath"])
    mfcc = extract_mfcc(file_path)
    mfcc_features.append(mfcc)


mfcc_df = pd.DataFrame(mfcc_features, columns=[f"MFCC_{i+1}" for i in range(20)])

# Concatenar con el dataset original
canciones = pd.concat([canciones, mfcc_df], axis=1)

print("¡Procesamiento de MFCC completado!")

features = []
for _, row in tqdm(canciones.iterrows(), total=len(canciones)):
    file_path = os.path.join(AUDIO_BASE_PATH, row["filepath"])
    audio_features = extract_audio_features(file_path)
    features.append(audio_features)

print("Procesamiento de features completado")

features_df = pd.DataFrame(features, columns=[f"Chroma_{i+1}" for i in range(12)] + [f"Spectral_contrast_{i+1}" for i in range(7)] + ["Zero_crossing_rate", "Spectral_Rolloff", "Tempo"])
canciones.loc[16360, "filepath"] = "0/0.wav"
mfcc_0 = extract_mfcc("/content/drive/MyDrive/Bootcamp_DS/ML/TRAIN_V2/data_out_2/0/0.wav")
canciones.loc[16360] = mfcc_0
canciones = pd.concat([canciones, features_df], axis=1)
canciones.dropna(inplace=True)
canciones.to_csv("/content/drive/MyDrive/Bootcamp_DS/ML/IA/processed.csv", index=False)