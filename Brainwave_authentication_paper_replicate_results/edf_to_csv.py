import os
import mne
import numpy as np

"""
edf = mne.io.read_raw_edf('Dataset/studie001_2019.05.08_10.15.34.edf')
header = ','.join(edf.ch_names)
np.savetxt('studie001_2019.05.08_10.15.34.csv', edf.get_data().T, delimiter=',', header=header)
"""



dir_path = 'Dataset'
target_dir_path = 'csv_dataset'

for filename in os.listdir(dir_path):
    if filename.endswith('.edf'):
        file_path = os.path.join(dir_path, filename)
        edf = mne.io.read_raw_edf(file_path)
        header = ','.join(edf.ch_names)
        csv_filename = filename.replace('.edf', '.csv')
        csv_file_path = os.path.join(target_dir_path, csv_filename)
        np.savetxt(csv_file_path, edf.get_data().T, delimiter=',', header=header)
