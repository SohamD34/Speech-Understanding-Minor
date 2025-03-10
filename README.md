# CSL7770: Speech Understanding Minor Exam
## Submitted by - Soham Deshmukh(B21EE067)

This is the code repository for submission to the Minor Exam of CSL7770: Speech Understanding Course.
It contains source code files, scripts and reports for the problem statements given in the question paper.

## Directory Structure
```
root
├── .gitignore
├── requirements.txt
├── LICENSE
├── README.md
├── B21EE067 - CSL7770 - Minor Exam Report.pdf          (Report for the exam submission)
├── Question 1
│       ├── data
│       ├── results
│       │      ├── audio_analysis_summary.xlsx
│       │      ├── pitch
│       │      ├── spectrograms
│       │      └── waveforms
│       ├── script.py
│       └── utils.py
│ 
├── Question 2
│       ├── data
│       ├── features
│       ├── script.py
│       └── utils.py
│
└── Question 3
        ├── data
        ├── results
        │      ├── log.txt
        │      ├── Features
        │      ├── Confusion-Matrices
        │      └── F1-F2 Plots
        ├── classification.py
        ├── feature_extraction.py
        └── utils.py
        
```


## Directions for using the repository


### 1) Fetching from GitHub 
First, clone the repository to your local machine and navigate to the repository using the following commands:
```
> cd <folder_name>
> git clone https://github.com/SohamD34/Speech-understanding-Minor.git
> cd Speech-Understanding-Minor
```

Now we need to set up the Python (Conda) environment for this repository. Execute the following commands in the terminal.
```
> conda create -p b21ee067 -y python==3.10.12
> conda activate b21ee067/
> conda install --yes --file requirements.txt
```


## Question 1
To run the code for Question 1, you can follow the steps -
```
> cd Question 1/
> python script.py
```
Results can be observed in the ```Question 1/results/``` folder. 


## Question 2
To run the code for Question 1, you can follow the steps -
```
> cd ../Question 2/
> python script.py
```
The extracted features for each audio file can be observed in the ```Question 2/features/all_features.csv``` file. 


## Question 3
To run the code for Question 3, you can follow the steps -

### a) To extract the features - F0, F1, F2, F3 - from audio files
```
> cd ../Question 3/
> python feature_extraction.py
```
The extracted features are stored in ```Question 3/data/audio_feature_data.csv```.
The plot of F1 vs F2 feature values can be observed at ```Question 3/results/F1-F2 PLots```.

### b) For classification
```
> python classification.py
```
Results can be observed in the ```Question 3/results/Confusion-Matrices``` folder and ```Question 3/results/log.txt```.
