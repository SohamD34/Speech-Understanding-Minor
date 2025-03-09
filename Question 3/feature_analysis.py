import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def import_data(file_path):
    data = pd.read_csv(file_path)
    return data


def f0_vs_vowels(data):

    vowels = ['a', 'e', 'i', 'o', 'u']
    f0 = data['F0']

    for i in range(1, len(vowels) + 1):
        vowel = vowels[i - 1]
        vowel_data = data[data['Label'] == vowel]
        plt.scatter([i for j in range(len(vowel_data))], vowel_data['F0'], alpha=0.5, label=vowel)

    plt.xticks(np.arange(1, len(vowels) + 1), vowels)
    plt.xlabel('Vowel')
    plt.ylabel('Count')
    plt.title('F0 Count vs Vowels')
    plt.legend()
    plt.savefig('results/Features/f0_vs_vowels.png')
    plt.close()


def f1_vs_vowels(data):

    vowels = ['a', 'e', 'i', 'o', 'u']
    f1 = data['F1']

    for i in range(1, len(vowels) + 1):
        vowel = vowels[i - 1]
        vowel_data = data[data['Label'] == vowel]
        plt.scatter([i for j in range(len(vowel_data))], vowel_data['F1'], alpha=0.5, label=vowel)

    plt.xticks(np.arange(1, len(vowels) + 1), vowels)
    plt.xlabel('Vowel')
    plt.ylabel('Count')
    plt.title('F1 Count vs Vowels')
    plt.legend()
    plt.savefig('results/Features/f1_vs_vowels.png')
    plt.close()


def f2_vs_vowels(data):

    vowels = ['a', 'e', 'i', 'o', 'u']
    f1 = data['F2']

    for i in range(1, len(vowels) + 1):
        vowel = vowels[i - 1]
        vowel_data = data[data['Label'] == vowel]
        plt.scatter([i for j in range(len(vowel_data))], vowel_data['F2'], alpha=0.5, label=vowel)

    plt.xticks(np.arange(1, len(vowels) + 1), vowels)
    plt.xlabel('Vowel')
    plt.ylabel('Count')
    plt.title('F2 Count vs Vowels')
    plt.legend()
    plt.savefig('results/Features/f2_vs_vowels.png')
    plt.close()


def f3_vs_vowels(data):

    vowels = ['a', 'e', 'i', 'o', 'u']
    f1 = data['F3']

    for i in range(1, len(vowels) + 1):
        vowel = vowels[i - 1]
        vowel_data = data[data['Label'] == vowel]
        plt.scatter([i for j in range(len(vowel_data))], vowel_data['F3'], alpha=0.5, label=vowel)

    plt.xticks(np.arange(1, len(vowels) + 1), vowels)
    plt.xlabel('Vowel')
    plt.ylabel('Count')
    plt.title('F3 Count vs Vowels')
    plt.legend()
    plt.savefig('results/Features/f3_vs_vowels.png')
    plt.close()


if __name__ == '__main__':
    data = import_data('data/audio_feature_data.csv')
    f0_vs_vowels(data)
    f1_vs_vowels(data)
    f2_vs_vowels(data)
    f3_vs_vowels(data)  