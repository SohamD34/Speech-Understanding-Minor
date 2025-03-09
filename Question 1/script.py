import os
os.chdir('/home/soham/Desktop/IIT Jodhpur/Speech Understanding/Speech-Understanding-Minor/Question 1')

import librosa as lb
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import pandas as pd
from utils import log_text


def plot_audio_features(y, sr, times, f0):
    '''
    This function computes and plots the waveform, spectrogram and pitch of the audio file.
    '''

    # plotting the waveform in its raw format
    plt.figure(figsize=(15, 4))
    lb.display.waveshow(y, sr=sr)       
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(output_dir + '/waveforms', os.path.basename(file_path).replace('.wav', '_waveform.png')))
    plt.close()
    
    # converting the audio to spectrogram and plotting in Freq v/s Time domain
    plt.figure(figsize=(15, 4))
    D = lb.amplitude_to_db(np.abs(lb.stft(y)), ref=np.max)
    lb.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_dir + '/spectrograms', os.path.basename(file_path).replace('.wav', '_spectrogram.png')))
    plt.close()
    
    # Computing the pitch using the fundamental frequencies v/s time
    plt.figure(figsize=(15, 4))
    plt.plot(times, f0, label='f0', color='cyan', linewidth=2)
    plt.title('Fundamental Frequency (Pitch)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir + '/pitch', os.path.basename(file_path).replace('.wav', '_pitch.png')))
    plt.close()





def analyze_audio(file_path, output_dir):

    # We load the audio file here.
    y, sr = lb.load(file_path, sr=None)

    # We find out the fundamental frequency f0, between the range fmin to fmax (custom values).
    f0, voiced_flag, voiced_probs = lb.pyin(y, fmin=lb.note_to_hz('C2'), fmax=lb.note_to_hz('C7'), sr=sr)
    times = lb.times_like(f0)
    
    # Plotting the waveform, spectrogram and pitch of the audio file
    plot_audio_features(y, sr, times, f0)
    
    # RMS energy for each window/frame
    frame_length = 2048
    hop_length = 512
    rms_energy = lb.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    max_rms_idx = np.argmax(rms_energy)
    max_rms_time = max_rms_idx * hop_length / sr

    # RMS energy stats
    mean_rms_energy = np.mean(rms_energy)
    max_rms_energy = np.max(rms_energy)
    min_rms_energy = np.min(rms_energy)
    
    # Overall RMS energy of the signal (all windows)
    overall_rms = np.sqrt(np.mean(y**2))
    max_amplitude = np.max(np.abs(y))
    min_amplitude = np.min(y)
    
    # Zero crossing rate
    zero_crossing_rate = lb.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    mean_zcr = np.mean(zero_crossing_rate)

    # Discarding all the invalid NaN values - we set their stats to 0

    valid_f0 = f0[~np.isnan(f0)]
    if len(valid_f0) > 0:
        mean_pitch = np.mean(valid_f0)
        median_pitch = np.median(valid_f0)
        min_pitch = np.min(valid_f0)
        max_pitch = np.max(valid_f0)
    else:
        mean_pitch, median_pitch, min_pitch, max_pitch = 0, 0, 0, 0
    
    results = {
        "File": os.path.basename(file_path),
        "Duration (s)": len(y) / sr,
        "Sample Rate (Hz)": sr,
        "Max Amplitude": max_amplitude,
        "Min Amplitude": min_amplitude,
        "Overall RMS Energy": overall_rms,
        "Peak RMS Energy": np.max(rms_energy),
        "Peak RMS Time (s)": max_rms_time,
        "Mean ZCR": mean_zcr,
        "Mean Pitch (Hz)": mean_pitch,
        "Median Pitch (Hz)": median_pitch,
        "Min Pitch (Hz)": min_pitch,
        "Max Pitch (Hz)": max_pitch
    }
    
    return results




if __name__ == "__main__":

    # Here we create the output directory to store all our results, plots, etc.

    output_dir = "results"
    audio_dir = "data"
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    # Computing features for all audio files in the 'data' directory

    for file_name in os.listdir(audio_dir):
        if file_name.endswith(('.wav', '.mp3')):

            print(f"Analyzing {file_name}...\n")

            file_path = os.path.join(audio_dir, file_name)
            results = analyze_audio(file_path, output_dir)

            all_results.append(results)
            
    
    # Saving the features of each audio file in a dataframe & then excel file for easy reference

    summary_df = pd.DataFrame(all_results)
    summary_df.to_excel(os.path.join(output_dir, 'audio_analysis_summary.xlsx'), index=False)

    print(f"Analysis complete. Results saved to 'Question 1/{output_dir}'")