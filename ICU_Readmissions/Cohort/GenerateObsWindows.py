# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 14:27:03 2020

This code requires the output of FinalDataset121020.py. 

Then generates observation windows based on the length of observation window
defined in this script. Observation windows will go from the end of the ICU
back the length of the window, one will cover the entire ICU stay, and finally 
a window will be generated for each hour of the ICU stay.

@author: Kirby
"""
#%% Package setup.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time

#%% Inputs
dataset = pd.read_csv("ICU_readmissions_dataset.csv")

eicu_path = r"C:\Users\Kirby\OneDrive\JHU\Precision Care Medicine\eicu"
    
pat = pd.read_csv(eicu_path + "\\patient.csv", 
                  usecols=['patientunitstayid', 'unitdischargeoffset'])

#%% Attach LOS as the end of each window. 
los_dict = pat.set_index('patientunitstayid').to_dict(
    ).get('unitdischargeoffset')

#Calculate LOS for deferred stays. 
def get_los(stay_id,orig,orig2,orig3,orig4):
    los = 0
    for stay in [stay_id,orig,orig2,orig3,orig4]:
        los += los_dict.get(stay,0)
    return los

dataset['LOS'] = dataset.apply(lambda row: get_los(
    row['patientunitstayid'],row['original_unitstayid'],
    row['2nd_orig_unitstayid'],row['3rd_orig_unitstayid'],
    row['4th_orig_unitstayid']),axis=1)



#Define observation window lengths to try.
for obs_hours in [1,3,6,12,24]:
    start = time()
    
    obs_minutes = obs_hours*60
    
    
    
    
    
    
    # #look at dist of LOS
    # plt.figure()
    # plt.title('LOS < 12 hours Histogram')
    # plt.xlabel('Hours')
    # plt.ylabel('Stay Counts')
    # (dataset[dataset['unitdischargeoffset']<720]['unitdischargeoffset']/60).hist()
    
    calc_time = time() - start