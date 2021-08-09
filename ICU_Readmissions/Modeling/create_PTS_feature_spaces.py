# -*- coding: utf-8 -*-
"""
Created on Thu May 20 09:14:34 2021

Combines PTS features into two big ol' csvs, one with all 
numeric data, one with all categorical data. Also generates a list of all
column names, of all numerical columns then all categorical columns.

Combines data over the same period of time together, or data with consecutive
intervals together. 

runtime: 

@author: Kirby
"""
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from datetime import datetime
import inspect as insp
import os

start = time()
now = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

# Get relative file paths.
filename = insp.getframeinfo(insp.currentframe()).filename
file_path = os.path.dirname(os.path.abspath(filename))
wd = Path(file_path)
print(file_path)
parent = wd.parent
cohort_path = wd.parent.joinpath('Cohort')
feature_path = wd.parent.joinpath('Features')

#%% Load in data. 

# Set up signal/timeframe names. 
signals = ['sao2','heartrate','respiration','allsystolic',
           'alldiastolic','allmean']
# Use this if just analyzing some of the signals.
# signals = ['sao2','heartrate','respiration']


timeframes = ['_1beforedisch','_3beforedisch','_6beforedisch','_12beforedisch',
              '_24beforedisch','_36beforedisch']
int_timeframes = ['_1hrinterval' + str(x) + 'hrbeforedisch' 
                  for x in range(1,13)]
# timeframes.extend(int_timeframes)

# Function to figure out if a column is binary or not. 
# Takes in column as a series. 
def is_binary(col):
    if col.value_counts().shape[0] == 2:
        return True
    else:
        return False
    
#%% Put together signal features over 1,3,6,12,24,36 hr time frames. 
#Load in all data for this time frame. 
#timeframe = '_1beforedisch'
for timeframe in timeframes:
    # Freshly load in IDs to merge onto. 
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'),
                  usecols=['patientunitstayid'])
    
    for signal in signals:
        # Load in data for that signal and timeframe. 
        data_name = signal + timeframe
        data = pd.read_csv(feature_path.joinpath('PTS', data_name + '.csv'))
        data.rename(columns={'Unnamed: 0':'patientunitstayid'},inplace=True)
        # Combine it with the others.
        ids = ids.merge(data,on='patientunitstayid',how='left')
    
    # Split out categorical and numeric data.
    num_cols = []
    cat_cols = []
    for col in ids.columns:
        if is_binary(ids[col]) == True:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    
    # Remove patientunitstayid from the numeric data. 
    num_cols.pop(0)
    
    num = ids[num_cols]
    cat = ids[cat_cols]
    
    # Add dummy columns to to ensure it's interpreted as DF if not enough cols.
    if cat.shape[1] < 2:
        cat.loc[:,'cat_dummy_col1'] = 0
        cat.loc[:,'cat_dummy_col2'] = 0
    if num.shape[1] < 2:
        num.loc[:,'num_dummy_col1'] = 0
        num.loc[:,'num_dummy_col2'] = 0    
    
    # Save off the data. 
    num.to_csv('numeric_data' + timeframe + '.csv',index=False)
    cat.to_csv('categorical_data' + timeframe + '.csv',index=False)
    
    # Save off column names.
    cols1 = num.columns.to_frame(index=False)
    cols2 = cat.columns.to_frame(index=False)
    all_cols = pd.concat([cols1,cols2],axis=0)
    # Convert commas in columns names to under scores. 
    all_cols.replace(to_replace=',',value='_',inplace=True,regex=True)
    all_cols.to_csv('column_names' + timeframe + '.csv',index=False)
    
#%% Put together all 1hr interval features over 12 hours for each signal.

for signal in signals:
    # Freshly load in IDs to merge onto. 
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'),
                  usecols=['patientunitstayid'])
    for timeframe in int_timeframes:
        # Load in data for that signal and timeframe. 
        data_name = signal + timeframe
        data = pd.read_csv(feature_path.joinpath('PTS', data_name + '.csv'))
        # Tack on suffix to each feature. 
        suffix = '_'+ timeframe.split('interval')[1].split('before')[0]
        data = data.add_suffix(suffix)
        # Correct patientunitstayid column.
        data.rename(columns={'Unnamed: 0' + suffix:'patientunitstayid'},
                    inplace=True)
        # Combine it with the others.
        ids = ids.merge(data,on='patientunitstayid',how='left')
        
    # Split out categorical and numeric data.
    num_cols = []
    cat_cols = []
    for col in ids.columns:
        if is_binary(ids[col]) == True:
            cat_cols.append(col)
        else:
            num_cols.append(col)
    
    # Remove patientunitstayid from the numeric data. 
    num_cols.pop(0)
    
    num = ids[num_cols]
    cat = ids[cat_cols]

    # Add dummy columns to to ensure it's interpreted as DF if not enough cols.
    if cat.shape[1] < 2:
        cat.loc[:,'cat_dummy_col1'] = 0
        cat.loc[:,'cat_dummy_col2'] = 0
    if num.shape[1] < 2:
        num.loc[:,'num_dummy_col1'] = 0
        num.loc[:,'num_dummy_col2'] = 0    

    # Save off the data. 
    num.to_csv('numeric_data_' + signal + '_' + '1hrintervals' + '.csv',
               index=False)
    cat.to_csv('categorical_data_'  + signal + '_' + '1hrintervals'  + '.csv',
               index=False)
    
    # Save off column names.
    cols1 = num.columns.to_frame(index=False)
    cols2 = cat.columns.to_frame(index=False)
    all_cols = pd.concat([cols1,cols2],axis=0)
    # Convert commas in columns names to under scores. 
    all_cols.replace(to_replace=',',value='_',inplace=True,regex=True)
    all_cols.to_csv('column_names_'+ signal + '_' + '1hrintervals' + '.csv',
                    index=False)
    

calc_time = time() - start