import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import os

# Load the feature data

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data


# Plot the feature data

def plot_feature(data, feature):

    plt.figure(figsize=(20,15))
    plt.scatter(np.arange(len(data)), data[feature], s=100)  
    plt.plot(np.arange(len(data)), data[feature], linewidth=5)  
    plt.title('Feature: ' + feature, fontsize=1000)  
    plt.grid()
    plt.title('Feature: ' + feature)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(np.arange(len(data)), data['Unnamed: 0'], rotation=90)

    plt.savefig('features/' + feature + '.png')
    plt.close()


# Main function
if __name__ == '__main__':
    file_path = 'features/all_features.csv'
    data = load_data(file_path)
    features = data.columns

\
    for feature in features:
        if(feature == 'Unnamed: 0'):
            continue
        plot_feature(data, feature)

