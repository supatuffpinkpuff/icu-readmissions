# -*- coding: utf-8 -*-
"""
Created on Tue May 11 17:59:51 2021

Extract PTS features. All needed files should be placed in the same folder 
as this script. 

Runtime, on local computer.
1 hour long interval - 4 min
36 hour long interval - 35 min
Estimated total - 3 hours per signal analyzed. 36 hours for all 12.

Extra safe estimate, assuming 35 hr interval times for all runs - 126 hr total

@author: Kirby
"""

#%% Package setup
import numpy as np
import pandas as pd
#import multiprocessing as mp
from time import time
from pathlib import Path
# from tsfresh import extract_relevant_features
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh.utilities.dataframe_functions import impute
import inspect as insp
import os

start = time()

filename = insp.getframeinfo(insp.currentframe()).filename
file_path = os.path.dirname(os.path.abspath(filename))
wd = Path(file_path)
print(file_path)
parent = wd.parent
cohort_path = wd.parent.parent.joinpath('Cohort')
eicu_path = wd.parent.parent.parent.joinpath('eicu')

#%% Load in data.
#Get the patient ids.
comp = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'),
                   usecols=['patientunitstayid','bad_disch_plan'])

#Get LOS.
pat = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','unitdischargeoffset'])
pat = pat[pat['patientunitstayid'].isin(comp['patientunitstayid'])]
comp = comp.merge(pat,on='patientunitstayid',how='left')
comp.rename(columns={'unitdischargeoffset':'LOS'},inplace=True)

#Get all PTS column names.
cols = ['sao2','heartrate','respiration','allsystolic','alldiastolic','allmean']

#%% Get different start/stop times. 

# Store column names without start/end in here. 
timeframes = []

# Last 1,3,6,12,24,36 hours pre-discharge.
for hours in [1,3,6,12,24,36]:
    col_str = str(hours) + 'beforedisch'
    comp[col_str + 'start'] = comp['LOS'] - (hours*60)
    comp[col_str + 'end'] = comp['LOS']
    timeframes.append(col_str)
# 12 1-hour intervals before discharge. 
for hours in list(range(1,13)):
    col_str = '1hrinterval' + str(hours) + 'hrbeforedisch'
    comp[col_str + 'start'] = comp['LOS'] - (hours * 60)
    comp[col_str + 'end'] = comp[col_str + 'start'] + 60
    timeframes.append(col_str)

#File to track progress.
f = open("pts_extraction_progress.txt", "w")
f.close()

#%% Calculate features. 
for timeframe in timeframes:
    for col in cols:
        #Load the pre-processed PTS data. 
        pts = pd.read_csv('pre-processed_' + col +'.csv')
        #Pre-processing already removed patient stays I don't care about.
        
        # Get the relevant window of pts given a start/stop time bound.
        start_col = timeframe + 'start'
        end_col = timeframe + 'end'
        windows = comp[['patientunitstayid',start_col,end_col]]
        
        # Drop data outside window. 
        pts = pts.merge(windows,on='patientunitstayid',how='left')
        pts = pts[pts['observationoffset'] >= pts[start_col]]
        pts = pts[pts['observationoffset'] <= pts[end_col]]
        pts = pts[['patientunitstayid','observationoffset',col]]
        
        # Clear out any remaining nans or duplicates, although there shouldn't be any. 
        pts.dropna(inplace=True)
        pts.drop_duplicates(inplace=True)
        
        #Calculate features. 
        ext = extract_features(pts, column_id="patientunitstayid", 
                               column_sort="observationoffset",
                               n_jobs=24)
        #Get rid of nan features.
        impute(ext)
        
        #Ensure IDs in data and target series match.
        pts_ids = ext.index.drop_duplicates()
        temp_comp = comp[comp['patientunitstayid'].isin(pts_ids)]
        temp_comp= temp_comp.set_index('patientunitstayid')['bad_disch_plan']
        
        #Pick ones with relevant p-values.
        feat = select_features(ext,temp_comp)
        
        feat.to_csv(col + '_' + timeframe + '.csv')
        
        # Track progress in txt file.
        f = open("pts_extraction_progress.txt", "a")
        f.write(col + '_' + timeframe + ' done! ' + str(time()-start) + 
                ' seconds since code started\n')
        f.close()

    
calc = time() - start