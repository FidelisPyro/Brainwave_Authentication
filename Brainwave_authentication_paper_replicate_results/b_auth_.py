import numpy as np
import pandas as pd
import mne
import os
import re
from collections import OrderedDict
from collections import defaultdict
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, make_scorer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


df = pd.read_csv('combined_features.csv')
df = df.drop(df.columns[0], axis=1)

"""
def eer_scorer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label='authenticate')
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    print(f'EER: {eer}')
    return 1 - eer
"""
def eer_scorer(y_true, y_score):
    '''
    Returns the equal error rate for a binary classifier output.
    '''
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label='authenticate')
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer


cv_strategy = RepeatedStratifiedKFold(n_splits=3, n_repeats=10, random_state=42)
scaler = StandardScaler()
"""
pipe = Pipeline([
    ('scaler', StandardScaler()), 
    ('selector', SelectKBest(score_func=f_classif)),
    ('svm', SVC(random_state=42, probability=True))
])

param_grid = {
    'selector__k': [5, 10, 25, 50, 100, 'all'],
    'svm__C': [0.1, 10], 
    'svm__kernel': ['rbf'], 
    'svm__gamma': [0.1],
    'svm__degree': [2],
    'svm__class_weight': [{ 'authenticate': 10, 'reject': 1 }, { 'authenticate': 1, 'reject': 1 }],
    'svm__coef0': [-1],
}
"""
pipe = Pipeline([('scaler', StandardScaler()), ('selector', SelectKBest(score_func=f_classif)), ('knn', KNeighborsClassifier())])
param_grid = {
    'selector__k': [5, 10, 25, 50, 100, 'all'],
    'knn__n_neighbors': [1, 2, 5, 10],
    'knn__weights': ['uniform', 'distance'],
    'knn__p': [1, 2]
}


"""
#param_grid = {'svm__C': [0.0001, 0.001, 0.01, 0.1, 1, 10], 'svm__kernel': ['linear', 'poly', 'rbf'], 
#              'svm__degree': [2, 3, 4], 'svm__class_weight': [{ 'authenticate': 10, 'reject': 1 }]}
#param_grid = {'svm__C': [0.001], 'svm__kernel': ['linear'], 'svm__class_weight': [{ 'authenticate': 10, 'reject': 1 }]}
#models = {}
"""

subjects = df['subject'].unique()
event_ids = df['event_id'].unique()

eer_values = {event_id: [] for event_id in event_ids}
#print(f'subjects: {subjects}\n')
#print(f'event_ids: {event_ids}\n')

for subject in subjects:
    for event_id in event_ids:
        print(f'\nsubject: {subject}, event_id: {event_id}')
        
        auth_df = pd.DataFrame(df[(df['subject'] == subject) & (df['event_id'] == event_id)])
        if auth_df.empty:
            print(f'No data for {subject} {event_id}')
            continue
        auth_df['target'] = 'authenticate'
        #print(f'auth_df:\n{auth_df.head()}')
        
        reject_df = pd.DataFrame(df[(df['subject'] != subject) & (df['event_id'] == event_id)])
        reject_df['target'] = 'reject'
        #print(f'reject_df:\n{reject_df.head()}')
        
        combined_df = pd.concat([auth_df, reject_df])
        #print(f'combined_df:\n{combined_df}')

        
        x = combined_df.drop(['subject', 'event_id', 'target'], axis=1)
        #print(f'x:\n{x.head()}')
        y = combined_df['target']
        #print(f'y:\n{y.head()}')
        #if 'authenticate' not in y:
        #    continue
        
        grid_search = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring=make_scorer(eer_scorer, greater_is_better=False, needs_proba=True))
        grid_search.fit(x, y)

        best_params = grid_search.best_params_
        best_score = abs(grid_search.best_score_)

        print(f'Best parameters for {subject} {event_id}: {best_params}')
        print(f'Best EER for {subject} {event_id}: {best_score}')

        eer_values[event_id].append(best_score)
        
        """
        scores = cross_val_score(pipe, x, y, cv=cv_strategy, scoring=make_scorer(eer_scorer, greater_is_better=False,
                                                                                         needs_proba=True))
        print(f'scores: {scores}')
        average_score = abs(np.mean(scores))
        eer_values[event_id].append(average_score)
        print(f'Average EER for {subject} {event_id}: {average_score}')
        """
        

        """
        models[subject] = {}
        
        df['target'] = np.where((df['subject'] == subject) & (df['event_id'] == event_id), 'authenticate', 'reject')
        
        x = df.drop(['subject', 'event_id', 'target'], axis=1)
        y = df['target']

        grid_search = GridSearchCV(pipe, param_grid, cv=cv_strategy, scoring=make_scorer(eer_scorer, greater_is_better=False,
                                                                                         needs_proba=True))
        grid_search.fit(x, y)
        
        models[subject][event_id] = grid_search.best_estimator_
        
        print(f'Best parameters for {subject} {event_id}: {grid_search.best_params_}')
        print(f'Best EER for {subject} {event_id}: {abs(grid_search.best_score_)}')

        eer_values[event_id].extend(grid_search.cv_results_['mean_test_score'])
        #print(f'EER2: {eer_scorer(y, grid_search.predict_proba(x)[:, 1])}')
        """
        """
        y_score = pipe.decision_function(x)
        
        fpr, tpr, thresholds = roc_curve(y, y_score, pos_label = 'authenticate')
        frr = 1 - tpr
        
        eer_index = np.nanargmin(np.absolute((frr - fpr)))
        eer_threshold = thresholds[eer_index]
        eer = fpr[eer_index]
        eer_values[event_id].append(eer)
        """
        
average_eers = {event_id: (np.mean(eers), np.std(eers)) for event_id, eers in eer_values.items()}

print("Model training complete.")
print("RF Average EERs per event:", average_eers)

