import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import os
from utils import *
os.chdir('/home/soham/Desktop/IIT Jodhpur/Speech Understanding/Speech-Understanding-Minor/Question 3')


def load_data():
    data = pd.read_csv('data/audio_feature_data.csv')
    return data


def validate_data(data):
    if data.isnull().values.any():
        return False
    return True


def preprocess_split_data(data):

    X = data.iloc[:, :-1].values
    Y = data.iloc[:, -1].values
    label_mapping = {'a': 0, 'e': 1, 'i': 2, 'o': 3, 'u': 4}
    Y = np.array([label_mapping[label] for label in Y])

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, Y_train, Y_test


def train_models(X_train, Y_train):

    svm = SVC()
    svm.fit(X_train, Y_train)
    svm_train_acc = accuracy_score(Y_train, svm.predict(X_train))

    dt = DecisionTreeClassifier()
    dt.fit(X_train, Y_train)
    dt_train_acc = accuracy_score(Y_train, dt.predict(X_train))

    gmm = GaussianMixture(n_components=3)
    gmm.fit(X_train, Y_train)
    gmm_train_acc = accuracy_score(Y_train, gmm.predict(X_train))

    return [svm, dt, gmm], [svm_train_acc, dt_train_acc, gmm_train_acc]


def test_models(models, X_test, Y_test):

    svm_test_acc = accuracy_score(Y_test, models[0].predict(X_test))
    dt_test_acc = accuracy_score(Y_test, models[1].predict(X_test))
    gmm_test_acc = accuracy_score(Y_test, models[2].predict(X_test))

    for i, model in enumerate(models):

        y_pred = model.predict(X_test)
        
        cm = confusion_matrix(Y_test, y_pred)
        plt.figure(figsize=(10, 7))
        fig = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix for Model {i+1}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'results/Confusion-Matrices/confusion_matrix_{i}.png')
        plt.close()

    return [svm_test_acc, dt_test_acc, gmm_test_acc]



if __name__ == "__main__":

    data = load_data()

    if not validate_data(data):
        print('Data contains null values')

    else:
        X_train, X_test, Y_train, Y_test = preprocess_split_data(data)
        models, train_acc = train_models(X_train, Y_train)
        test_acc = test_models(models, X_test, Y_test)

        result = f'''
Training accuracies:
SVM: {train_acc[0]}
Decision Tree: {train_acc[1]}
GMM: {train_acc[2]}

Testing accuracies:
SVM: {test_acc[0]}
Decision Tree: {test_acc[1]}
GMM: {test_acc[2]}
'''

        log_text('results/log.txt', result)
        print('Result stored in results/ directpry.')