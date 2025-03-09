# Importing the necessary libraries

import numpy as np
import librosa
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.io import wavfile
import json
from utils import *
import argparse
import warnings
warnings.filterwarnings('ignore')

# Defining some global parameters that will remain common throughout the script
FRAME_SIZE = 1024
HOP_SIZE = 512
N_MFCC = 13


# Checking if there is any pre-existing output directory, due to previoud iterations. 
# This step is important to avoid corruption of data stored during runtime. If yes, we remove it and create a new one.

def check_output_dir():
    if os.path.exists(features_dir):
        os.system(f'rm -rf {features_dir}')
    os.makedirs(features_dir, exist_ok=True)
    return True


# Loading the audio file into a vector format

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr


# Saving the data in JSON format - helful to store dictionaries and lists 

def save_json(data, file_name):
    output_path = f'{features_dir}/{file_name}.json'
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4, default=str)


# Padding the audio to make sure that the audio is divisible into frames of size 1024

def pad_audio(audio):
    pad_length = (len(audio) - FRAME_SIZE) % HOP_SIZE
    return np.pad(audio, (0, pad_length))


# Dividing the audio into the frames of size = FRAME_SIZE and with a hop size of HOP_SIZE

def get_frames(padded_audio):
    num_frames = int((len(padded_audio) - FRAME_SIZE) / HOP_SIZE)
    frames = []
    
    for i in range(num_frames):
        frame = padded_audio[i * HOP_SIZE: i * HOP_SIZE + FRAME_SIZE]
        frames.append(frame)
        
    return frames


# Calculating the Zero Crossing Rate of the audio

def calculate_zcr(audio):
    padded_audio = pad_audio(audio)
    frames = get_frames(padded_audio)
    
    zcr_values = []
    for frame in frames:
        zcr = np.sum(np.abs(np.sign(frame[1:]) - np.sign(frame[:-1])) / 2) / FRAME_SIZE
        zcr_values.append(zcr)
    
    return np.mean(np.array(zcr_values))


# Calculating the Short Time Energy of the audio

def calculate_short_time_energy(audio):
    padded_audio = pad_audio(audio)
    frames = get_frames(padded_audio)
    
    energy_values = []
    for frame in frames:
        energy = np.sum(frame ** 2)
        energy_values.append(energy)
    
    return np.mean(np.array(energy_values))


# Extracting the Mel-Frequency Cepstral Coefficient (MFCC) features from the audio

def extract_mfcc_features(y, sr):
    mfcc_features = {}
    mfccs = librosa.feature.mfcc(
        y=y, 
        sr=sr, 
        n_mfcc=N_MFCC, 
        hop_length=HOP_SIZE, 
        n_fft=FRAME_SIZE
    )
    
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)
    
    for i in range(N_MFCC):
        mfcc_features[f'mfccs_mean_{i+1}'] = float(mfccs_mean[i])
        mfcc_features[f'mfccs_std_{i+1}'] = float(mfccs_std[i])
        
    return mfcc_features


# Extracting the audio features like ZCR, STE and MFCC from the audio

def extract_audio_features(y, sr):
    features = {}
    
    # Basic features
    features['zcr'] = float(calculate_zcr(y))
    features['ste'] = float(calculate_short_time_energy(y))
    
    # MFCC features
    mfcc_features = extract_mfcc_features(y, sr)
    features.update(mfcc_features)
    
    return features


# Processing the audio file to extract the features

def process_audio_file(file_path):
    y, sr = load_audio(file_path)
    features = extract_audio_features(y, sr)
    return features


# Extracting the features from all the files in the dataset

def extract_dataset_features(dataset_dir):
    
    print(f'Processing files in dir -- {dataset_dir}\n\n')
    
    features_all = {}
    all_files = os.listdir(dataset_dir)
    total_files = len(all_files)
    
    for i, file in enumerate(all_files):
        print(f'Processing {i+1}/{total_files} -- {file}')
        file_path = f'{dataset_dir}/{file}'
        
        features = process_audio_file(file_path)
        file_name = file[:-4]
        print(features)
        
        features_all[file] = features
        print('Features extracted\n')
    

    df = pd.DataFrame.from_dict(features_all, orient='index')    
    df.to_csv(f'features/all_features.csv', index=True)
    
    print('Processing finished')
    print(f'Features saved in dir -- {features_dir}')




if __name__ == '__main__':
    
    # We are loading all the files from the 'data' directory and extracting the features from them
    # The features are then saved in the 'features' directory
        
    data_dir =  'data'
    features_dir = 'features'
    
    check_output_dir()
    extract_dataset_features(data_dir)