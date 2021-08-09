# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:34:13 2021

Split out PTS data into separate files of just our dataset's values. 

Does remove implausible values. No interpolation here. 

Runtime: 11 minutes

@author: Kirby
"""

#%% Package setup
import numpy as np
import pandas as pd
#import multiprocessing as mp
from time import time
from pathlib import Path
import inspect as insp
import os

start = time()

# Get relative file paths.
filename = insp.getframeinfo(insp.currentframe()).filename
file_path = os.path.dirname(os.path.abspath(filename))
wd = Path(file_path)
print(file_path)
cohort_path = wd.parent.parent.joinpath('Cohort')
eicu_path = wd.parent.parent.parent.joinpath('eicu')

#%% Load in data. 

# Get LOS data. 
los = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','unitdischargeoffset'])
los.rename(columns={'unitdischargeoffset':'los'},inplace=True)

# Load in vitals data. 

vitals = pd.read_csv(eicu_path.joinpath('vitalPeriodic.csv'),
                     usecols=['patientunitstayid','observationoffset',
                              'sao2','heartrate','respiration',
                              'systemicsystolic','systemicdiastolic',
                              'systemicmean'])
avitals = pd.read_csv(eicu_path.joinpath('vitalAperiodic.csv'),
                     usecols=['patientunitstayid','observationoffset',
                              'noninvasivesystolic','noninvasivediastolic',
                              'noninvasivemean'])

#%% Split out data.

cols = list(vitals.columns)
cols.extend(list(avitals.columns[2:5]))
cols = cols[2:len(cols)]
bounds = pd.read_excel('pts_bounds.xlsx').set_index('col').to_dict()
for col in cols:
    if col.startswith('noninvasive'):
        data = avitals[['patientunitstayid','observationoffset',col]].copy()
    else:
        data = vitals[['patientunitstayid','observationoffset',col]].copy()
    #Remove impossible values. 
    low = bounds.get('low').get(col)
    high = bounds.get('high').get(col)
    orig_data = data[~data[col].isna()].shape[0]
    #Replace implausible data with nans. 
    data.loc[data[col] < low, col] = np.nan
    data.loc[data[col] > high, col] = np.nan
    print(col + ': ' + str((data[~data[col].isna()].shape[0])/orig_data) + 
          ' data was plausible')
    #Remove data before ICU Stay.
    data = data[data['observationoffset'] >= 0]
    # Remove data after ICU stay.
    data = data.merge(los,on='patientunitstayid',how='left')
    data = data[data['observationoffset'] < data['los']]
    data = data[['patientunitstayid','observationoffset',col]]
    # Handle time stamps with multiple entries. 
    data = data.groupby(['patientunitstayid','observationoffset']).mean()
    data.reset_index(inplace=True)
    #Sort it. 
    data.sort_values(['patientunitstayid','observationoffset'],inplace=True)
    data.to_csv('all_pre-processed_' + col + '.csv',index=False)

#%% Remove BP data where pulse pressure is less than 5. 

for source in ['systemic','noninvasive']:
    # Load in data.
    dia_bp = pd.read_csv('all_pre-processed_' + source + 'diastolic.csv')
    sys_bp = pd.read_csv('all_pre-processed_' + source + 'systolic.csv')
    mean_bp = pd.read_csv('all_pre-processed_' + source + 'mean.csv')
    # Combine the data. 
    comb_cols = ['patientunitstayid','observationoffset']
    both_bp = dia_bp.merge(sys_bp,on=comb_cols,how='outer')
    both_bp = both_bp.merge(mean_bp,on=comb_cols,how='outer')
    # Remove those where pulse pressure was so low. 
    pulse_pres = both_bp[source + 'systolic'] - both_bp[source + 'diastolic']
    sbp_mbp = both_bp[source + 'systolic'] - both_bp[source + 'mean']
    #Replace implausible data with nans. 
    value_cols = [source + 'systolic', source + 'diastolic', source + 'mean']
    both_bp.loc[pulse_pres < 5, value_cols] = np.nan
    both_bp.loc[pulse_pres > 200, value_cols] = np.nan
    both_bp.loc[sbp_mbp < 3, value_cols] = np.nan
    # Save back off the data. 
    dia_bp = both_bp[['patientunitstayid','observationoffset',
                      source + 'diastolic']].drop_duplicates()
    dia_bp.to_csv('all_pre-processed_' + source + 'diastolic.csv',index=False)
    sys_bp = both_bp[['patientunitstayid','observationoffset',
                      source + 'systolic']].drop_duplicates()
    sys_bp.to_csv('all_pre-processed_' + source + 'systolic.csv',index=False)
    mean_bp = both_bp[['patientunitstayid','observationoffset',
                       source + 'mean']].drop_duplicates()
    mean_bp.to_csv('all_pre-processed_' + source + 'mean.csv',index=False)



#%% Make combined BP data, including interpolation. 
# for col in ['systolic','diastolic','mean']:
#     #Just get invasive BP data. 
#     data = pd.read_csv('all_pre-processed_systemic' + col + '.csv')
#     data.rename(columns={'systemic' + col:'all' + col},inplace=True)
#     #Tack on non-invasive with it. 
#     noninv = pd.read_csv('all_pre-processed_noninvasive' + col + '.csv')
#     noninv.rename(columns={'noninvasive' + col:'all' + col},inplace=True)
#     data = pd.concat([data,noninv])
#     #Sort it. 
#     data.sort_values(['patientunitstayid','observationoffset'],inplace=True)
#     data.to_csv('all_pre-processed_all' + col + '.csv',index=False)

calc = time() - start

# #%% 