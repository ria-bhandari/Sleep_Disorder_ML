import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/Users/riab/Documents/ECS 170 Project/Modeling-Sleep-Apnea-ML/Sleep_health_and_lifestyle_dataset.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import mne
import pandas as pd
import h5py
from scipy.io import loadmat
import scipy.signal as sg
from scipy.integrate import simps
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import seaborn as sns
import csv

def get_epochs(file_path, annot_filepath): # epoch is checking all data at once
    data = loadmat(file_path)['record']
    annot = loadmat(annot_filepath)['anno_apnea']
    fs = data.shape[1]/22470 # will be different for each patient
    epoch_len = int(fs*60) # 1 min.
    epochs = np.array([data[:,x-epoch_len:x] for x in range(epoch_len,data.shape[1]+1,epoch_len)]);print(epochs.shape)
    epochs = np.reshape(epochs, (epoch_len,data.shape[0],data.shape[1]//epoch_len))
    targets = annot[:, :epochs.shape[-1]].flatten()
    normal_epochs = epochs[:,:,targets==0]
    apnea_epochs = epochs[:,:,targets==1]
    print(normal_epochs.shape, apnea_epochs.shape)
    return normal_epochs, apnea_epochs, targets

delta = 1,4
theta = 4,8
alpha = 8,13
beta = 13,30
gamma = 30,100

normal_epochs, apnea_epochs, labels = get_epochs("../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/ucddb002/ucddb002.mat", "../input/sleep-edf-and-apnea/10318976/Data/UCDPaper/ucddb002/ucddb002_anno.mat")