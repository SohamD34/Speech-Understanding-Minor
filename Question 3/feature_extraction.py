import numpy as np
import pandas as pd
import librosa
from scipy import signal
from typing import Tuple
import os
import warnings
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")


def compute_f0(y: np.ndarray, sr: int) -> float:
    """    
    Parameters :
    y : np.ndarray - Audio time series
    sr : int - Sampling rate
        
    Returns :
    f0 : float - Fundamental frequency in Hz
    """
    n_fft = 2048
    y_windowed = y * np.hanning(len(y)) if len(y) <= n_fft else y[:n_fft] * np.hanning(n_fft)
    
    corr = librosa.autocorrelate(y_windowed)
    min_freq, max_freq = 50, 400
    
    min_lag = int(sr / max_freq)
    max_lag = int(sr / min_freq)
    
    min_lag = max(min_lag, 1)
    max_lag = min(max_lag, len(corr) - 1)
    
    lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    f0 = sr / lag if lag > 0 else 0.0
    
    return f0



def extract_formants(y: np.array, order: int = 16, sr: int = 22050) -> Tuple[float, float, float]:
    """
    Parameters :
    y : np.ndarray - Audio signal 
    order : int - Order of the autoregressive filter (number of LPC coefficients)
    sr : int - Sampling rate of the audio
        
    Returns :
    Tuple[float, float, float, float] - F0, F1, F2, F3 formant frequencies in Hz
    """

    if not np.isfinite(y).all():
        raise ValueError("Audio contains NaN or Inf values")
    
    f0 = compute_f0(y, sr)
    
    y = librosa.effects.preemphasis(y)
    
    fwd_pred_error = y[1:]
    back_pred_error = y[:-1]
    
    autoregressive_coeffs = np.zeros(order + 1)
    autoregressive_coeffs[0] = 1.0
    autoregressive_coeffs_prev = autoregressive_coeffs.copy()
    
    reflection_coefficient = np.zeros(1)
    den = np.zeros(1)
    
    epsilon = np.finfo(y.dtype).eps    
    den[0] = np.sum(fwd_pred_error**2 + back_pred_error**2)
    
    for i in range(order):

        reflection_coefficient[0] = np.sum(back_pred_error * fwd_pred_error) * -2
        reflection_coefficient[0] /= den[0] + epsilon
        
        autoregressive_coeffs_prev, autoregressive_coeffs = autoregressive_coeffs, autoregressive_coeffs_prev
        for j in range(1, i + 2):
            autoregressive_coeffs[j] = autoregressive_coeffs_prev[j] + reflection_coefficient[0] * autoregressive_coeffs_prev[i - j + 1]
        
        fwd_pred_error_tmp = fwd_pred_error
        fwd_pred_error = fwd_pred_error + reflection_coefficient * back_pred_error
        back_pred_error = back_pred_error + reflection_coefficient * fwd_pred_error_tmp
        
        q = 1.0 - reflection_coefficient[0] ** 2
        den[0] = q * den[0] - back_pred_error[-1] ** 2 - fwd_pred_error[0] ** 2          
        
        fwd_pred_error = fwd_pred_error[1:]
        back_pred_error = back_pred_error[:-1]
    

    roots = np.roots(autoregressive_coeffs)    
    roots = roots[np.abs(roots) < 1]                # stable roots (mag < 1)
    roots = roots[np.imag(roots) > 0]               # positive imaginautoregressivey pautoregressivet (any one conjugate)
    
    angles = np.angle(roots)
    
    formants_freq = angles * (sr / (2 * np.pi))
    
    formants_freq = np.sort(formants_freq)

    if len(formants_freq) >= 3:
        return f0, formants_freq[0], formants_freq[1], formants_freq[2]
    
    elif len(formants_freq) == 2:
        return f0, formants_freq[0], formants_freq[1], 0.0
    
    elif len(formants_freq) == 1:
        return f0, formants_freq[0], 0.0, 0.0
    
    else:
        return f0, 0.0, 0.0, 0.0



if __name__ == "__main__":
        
        results_dir = os.path.join(os.getcwd(), 'results')
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        
        dir = os.path.join(os.getcwd(), 'data')
        os.chdir(dir)
        classes = os.listdir()

        data = pd.DataFrame(columns=['F0', 'F1', 'F2', 'F3', 'Label'])
        
        all_f1 = []
        all_f2 = []
        all_classes = []
        
        for c in classes:
            class_folder = os.path.join(dir, c)

            if os.path.isdir(class_folder):
                
                for vowel in os.listdir(class_folder):
                    vowel_dir = os.path.join(class_folder, vowel)

                    for audio_file in os.listdir(vowel_dir):

                        audio_file = os.path.join(vowel_dir, audio_file)
                        y, sr = librosa.load(audio_file, sr=None)
                        f0, f1, f2, f3 = extract_formants(y, order=16, sr=sr)

                        data = pd.concat([data, pd.DataFrame([{'F0': f0, 'F1': f1, 'F2': f2, 'F3': f3, 'Label': vowel}])], ignore_index=True)

                        all_f1.append(f1)
                        all_f2.append(f2)
                        all_classes.append(vowel)


        data.to_csv('audio_feature_data.csv', index=False)

        os.chdir('../results/')
        cmap = plt.get_cmap('viridis', len(set(all_classes)))
        colors = [cmap(i) for i in range(len(set(all_classes)))]

        class_to_color = {label: colors[i] for i, label in enumerate(set(all_classes))}
        color_list = [class_to_color[label] for label in all_classes]

        plt.scatter(all_f1, all_f2, c=color_list, label=all_classes)
        plt.xlabel('F1 (Hz)')
        plt.ylabel('F2 (Hz)')
        plt.title('F1 vs F2')

        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
        labels = list(class_to_color.keys())
        plt.legend(handles, labels, title="Classes")

        plt.savefig('F1-F2 Plots/F1_vs_F2.png')

        print("Data extracted successfully to 'Question 3/data/audio_feature_data.csv'")