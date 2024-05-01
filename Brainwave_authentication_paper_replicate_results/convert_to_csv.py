import numpy as np
import mne
import os
import re
import pandas as pd
from collections import OrderedDict
from collections import defaultdict
from scipy.signal import welch
from statsmodels.tsa.ar_model import AutoReg
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

# Define file path
dir_path = 'Dataset'
raw_dict = {}

# Read and preprocess EDF files
for filename in os.listdir(dir_path):
    try:
        if filename.endswith('.edf'):
            file_path = os.path.join(dir_path, filename)
            raw = mne.io.read_raw_edf(file_path, preload=True)
            raw.filter(1, 50, fir_design='firwin')
            subject_number = re.findall(r'\d+', filename)[0]  
            new_key = f"subject_{int(subject_number):02d}"
            raw_dict[new_key] = raw
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Order raw_dict by subject number
raw_dict = OrderedDict(sorted(raw_dict.items(), key=lambda item: int(re.findall(r'\d+', item[0])[0])))

print("\nFinished ordering raw data.")
print(f'Number of keys in raw_dict: {len(raw_dict)}')
print(f'Keys in raw_dict: {list(raw_dict.keys())}')
#print(f'Contents of raw_dict: {raw_dict}\n')

# Load and sort event files
event_txt_dict = {}
for filename in os.listdir(dir_path):
    try:
        if filename.endswith('-events.txt'):
            file_path = os.path.join(dir_path, filename)
            event = np.loadtxt(file_path, skiprows=1, dtype={'names': ('Latency', 'Type', 'Duration'), 'formats': ('f4', 'U10', 'f4')})
            subject_number = re.findall(r'\d+', filename)[0]
            new_key = f"subject_{int(subject_number):02d}"
            event_txt_dict[new_key] = event
    except Exception as e:
        print(f"Error loading events for file {filename}: {e}")

event_txt_dict = OrderedDict(sorted(event_txt_dict.items(), key=lambda item: int(re.findall(r'\d+', item[0])[0])))

print("\nFinished ordering event data.")
print(f'Number of keys in event_txt_dict: {len(event_txt_dict)}')
#print(f'Contents of raw_dict: {event_txt_dict}\n')

# Process events to create epochs
epochs_dict = {}
for key in raw_dict.keys():
    if key in event_txt_dict:
        events = []
        for row in event_txt_dict[key]:
            sample_index = int(row['Latency'] * 256)  # sfreq
            
            # Define the Event Code
            if 'S1P300' in row['Type']:
                event_code = 1
            elif 'S2P300' in row['Type']:
                event_code = 2
            elif 'N400words' in row['Type']:
                event_code = 3
            elif 'N400sent' in row['Type']:
                event_code = 4
            elif 'Faces' in row['Type']:
                event_code = 5
            else:
                # If the event is not one of the above, skip it
                continue
            events.append([sample_index, 0, event_code])
        # Check if events list isn't empty
        if events:
            events = np.array(events, dtype=int)
            if key == 'subject_35':
                event_id={'S1P300': 1, 'S2P300': 2, 'N400words': 3, 'N400sent': 4}
            else:
                event_id={'S1P300': 1, 'S2P300': 2, 'N400words': 3, 'N400sent': 4, 'Faces': 5}
            epochs = mne.Epochs(raw_dict[key], events, event_id, tmin=-0.1, tmax=0.9, baseline=(None, 0), preload=True)
            epochs_dict[key] = epochs

print("\nFinished making the epochs from raw data and event data.")
print(f'Number of keys in epochs_dict: {len(epochs_dict)}')
#print(f'Contents of raw_dict: {epochs_dict}\n')


# Threshold for rejecting epochs based on peak-to-peak amplitude
reject_criteria = dict(eeg=150e-6)  

filtered_epochs = {}
for key in epochs_dict:
    
    # Apply rejection criteria and pick a subset of EEG channels
    epochs_ch_subset = epochs_dict[key].copy().pick_channels(epochs_dict[key].ch_names[2:16])
    filtered_epochs[key] = epochs_ch_subset.copy().drop_bad(reject = reject_criteria)
    
print("\nFinished filtering epoch data.\n")
    
del filtered_epochs['subject_42']
del filtered_epochs['subject_55']

print(f'Number of keys in filtered_epochs: {len(filtered_epochs)}')
#print(f'Contents of filtered_epochs: {filtered_epochs}\n')
for key in filtered_epochs:
    print(f'filtered epochs {key}: {(filtered_epochs[key].get_data().shape)}')
    
subject_event_dict = {}

for subject, epochs in filtered_epochs.items():
    subject_event_dict[subject] = {}
    
    for event_id in epochs.event_id:
        event_indices = (epochs.events[:, 2] == epochs.event_id[event_id])
        
        event_data = epochs.get_data()[event_indices]
        
        subject_event_dict[subject][event_id] = event_data

print(f'subject_event_dict shape: {len(subject_event_dict)}')
print(f'subject_event_dict subject_01: {subject_event_dict["subject_01"].keys()}')

def remove_insufficient_epochs(power_spectrum_features_dict):
    subjects_to_delete = []
    
    for subject in subject_event_dict:
        for event_id, epochs in subject_event_dict[subject].items():
            if epochs.shape[0] < 3:
                print(f'Deleting {subject} event: {event_id} due to insufficient epochs: {epochs.shape[0]}')
                subjects_to_delete.append((subject, event_id))
                
    for subject, event_id in subjects_to_delete:
        del subject_event_dict[subject][event_id]
        
    return subject_event_dict

subject_event_dict = remove_insufficient_epochs(subject_event_dict)

# Function to calculate power spectral density features
def calc_power_spectrum_features(epochs, sfreq=256, freq_bands=[(1, 10), (10, 13), (13, 30), (30, 50)]):
    power_features = []
    for epoch in epochs:
        psd_features = []
        for channel_data in epoch:
            freqs, psd = welch(channel_data, sfreq, nperseg=len(channel_data))
            band_powers = [np.sum(psd[(freqs >= band[0]) & (freqs <= band[1])]) for band in freq_bands]
            psd_features.append(band_powers)
        power_features.append(np.array(psd_features).flatten())  # Flatten to make a single feature vector per epoch
    return np.array(power_features)

# Function to calculate AR coefficients for each epoch
def calc_ar_coeffs(epochs, order=10):
    ar_coeffs = []
    for epoch in epochs:
        epoch_coeffs = []
        for channel_data in epoch:
            model = AutoReg(channel_data, lags=order)
            results = model.fit()
            epoch_coeffs.append(results.params)
        ar_coeffs.append(np.array(epoch_coeffs).flatten())  # Flatten to make a single feature vector per epoch
    return np.array(ar_coeffs)

# Apply feature extraction
combined_features_dict = {}
for subject, events in subject_event_dict.items():
    combined_features_dict[subject] = {}
    for event_id, epochs in events.items():
        power_features = calc_power_spectrum_features(epochs)
        ar_features = calc_ar_coeffs(epochs)
        # Combine features immediately for simplification
        combined_features = np.hstack((power_features, ar_features))
        combined_features_dict[subject][event_id] = combined_features

print("\nFeature extraction, filtering, scaling, and combination complete.")
print(f'Number of keys in combined_features_dict: {len(combined_features_dict)}\n')


data = []
for subject, events in combined_features_dict.items():
    for event_id, features in events.items():
        for feature in features:
            row = {'subject': subject, 'event_id': event_id}
            for i, value in enumerate(feature):
                row[f'feature_{i+1}'] = value
            data.append(row)

df = pd.DataFrame(data)

df.to_csv('combined_features.csv')

"""

def eer_scorer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label='authenticate')
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return 1 - eer




#pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42, probability=True))])
#pipe = Pipeline([('scaler', StandardScaler()), ('knn', KNeighborsClassifier())])
#pipe = Pipeline([('scaler', StandardScaler()), ('LG', LogisticRegression(random_state=42))])
#pipe = Pipeline([('scaler', StandardScaler()), ('RF', RandomForestClassifier(random_state=42))])
#pipe = Pipeline([('scaler', StandardScaler()), ('GNB', GaussianNB())])
#pipe = Pipeline([('scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])

# Model Training with GridSearchCV
#param_grid = {'svm__C': [10], 'svm__kernel': ['linear'], 'svm__class_weight': [{ 'authenticate': 10, 'reject': 1 }]}
#param_grid = {'svm__C': [0.001], 'svm__kernel': ['poly'], 'svm__degree': [2], 'svm__gamma': ['auto'], 'svm__class_weight': [{ 'authenticate': 10, 'reject': 1 }]}
#param_grid = {'knn__n_neighbors': [2]}
#param_grid = {'LG__C': [0.01]}
#param_grid = {'RF__n_estimators': [1000], 'RF__max_depth': [5]}
#param_grid = {'GNB__var_smoothing': [1e-9]}
#param_grid = {'LDA__solver': ['svd']}


cv_strategy = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
#scaler = StandardScaler()
pipe = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=42, probability=True, C = 0.001, kernel = 'linear', 
                                                           class_weight = { 'authenticate': 10, 'reject': 1 }))])
models = {}
eer_values = {event_id: [] for event_id in combined_features_dict['subject_01'].keys()}

for subject in combined_features_dict.keys():
    for event in combined_features_dict[subject].keys():
        # Create features_list and labels for each event for each subject
        x_train_auth = []
        x_train_reject = []
        labels_auth = []
        labels_reject = []
        for subj in combined_features_dict.keys():
            for evt in combined_features_dict[subj].keys():
                if subj == subject and evt == event:
                    # This is the current subject and event, label as 'authenticate'
                    x_train_auth.append(np.array(combined_features_dict[subj][evt]))
                    labels_auth.extend(['authenticate'] * len(combined_features_dict[subj][evt]))
                elif evt == event:
                    # This is the same event but from a different subject, label as 'reject'
                    x_train_reject.append(np.array(combined_features_dict[subj][evt]))
                    labels_reject.extend(['reject'] * len(combined_features_dict[subj][evt]))
        x_train_auth = np.concatenate(x_train_auth)
        x_train_reject = np.concatenate(x_train_reject)

        labels = labels_auth + labels_reject
        features_list = np.concatenate((x_train_auth, x_train_reject))

        #custom_scorer = make_scorer(eer_scorer, needs_proba=True)
        
        pipe.fit(features_list, labels)
        y_score = pipe.decision_function(features_list)
        
        fpr, tpr, thresholds = roc_curve(labels, y_score, pos_label = 'authenticate')
        frr = 1 - tpr
        #print(f'fnr: {fnr}')
        #print(f'fpr: {fpr}')
        if 'authenticate' not in labels:
            continue
        eer_index = np.nanargmin(np.absolute((frr - fpr)))
        eer_threshold = thresholds[eer_index]
        eer = fpr[eer_index]
        eer_values[event_id].append(eer)
        
        #grid = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring=custom_scorer)
        #grid.fit(features_list, labels)
        
        #eer_values[event].extend(grid.cv_results_['mean_test_score'])
        
average_eers = {event_id: (np.mean(eers), np.std(eers)) for event_id, eers in eer_values.items()}

print("Model training complete.")
print("RF Average EERs per event:", average_eers)

"""
