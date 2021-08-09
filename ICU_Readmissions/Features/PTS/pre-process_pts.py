# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:34:13 2021

Split out pre-processed PTS data into separate files of just our dataset's values. 

1 hour interpolation for non BP data, and 
a special 2-hour interpolation for BP data that also combines the non-invasive 
and invasive data, generally assuming that invasive data is more accurate.

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

#Get the patient ids.
comp = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))


#%% Interpolation function that only interpolates if less than threshold 
    # consecutive nans and is same unit stay.
def interp_thresh(vitals,thresh,col):
    # Get stays. 
    stays = vitals['patientunitstayid'].drop_duplicates()
    df_list = list()
    
    for stay in stays:
        temp_vitals = vitals[vitals['patientunitstayid'] == stay].copy()
        # Only interpolates if gap less than 2 hour. 
        mask = temp_vitals.copy()
        # Create column that makes new id for each value or consecutive group of nans.
        grp = pd.DataFrame(
            ((mask[col].notnull() != mask[col].shift().notnull()).cumsum()),
            columns=[col])
        grp['ones'] = 1
        # Group by that column and count to find where less than thresh consecutive nans are.
        mask['mask'] = ((grp.groupby(col)['ones'].transform('count') <= thresh) | 
                        temp_vitals[col].notnull())
        temp_vitals[col] = temp_vitals[col].interpolate().bfill()[mask['mask']]
        df_list.append(temp_vitals)
    
    vitals = pd.concat(df_list)
    return vitals

# But can't allow interpolation to cross over unit stays...
# process each patient unit stay id's data separately?'

#%% Split out and interpolate non BP data.

cols = pd.read_excel('pts_bounds.xlsx')['col'][0:3]
for col in cols:
    vitals = pd.read_csv('all_pre-processed_' + col + '.csv')    
    # Just get patients we care about. 
    vitals = vitals[vitals['patientunitstayid'].isin(comp['patientunitstayid'])]
    # 1 hour interpolation.
    vitals = interp_thresh(vitals,12,col)
    # Save off data. 
    vitals.to_csv('pre-processed_' + col + '.csv', index=False)
    print(col + ' done!')
calc = time() - start

#%% Combine and interpolate BP data. 

for bp_type in ['systolic','diastolic','mean']:
    inv = pd.read_csv('all_pre-processed_systemic' + bp_type + '.csv')
    noninv = pd.read_csv('all_pre-processed_noninvasive' + bp_type + '.csv')
    # Just get patient stays we care about. 
    inv = inv[inv['patientunitstayid'].isin(comp['patientunitstayid'])]
    noninv = noninv[noninv['patientunitstayid'].isin(comp['patientunitstayid'])]
    # 2 hour interpolation of invasive data.
    inv = interp_thresh(inv,24,'systemic' + bp_type)
    # Drop nas from invasive data, no longer needed.
    inv.dropna(inplace=True)
    # Sort data to prepare for merge_asof.
    inv.sort_values('observationoffset',inplace=True)
    noninv.sort_values('observationoffset',inplace=True)
    # Remove non invasive data with invasive data within 5 min.
    # Find said data. 
    overlap = pd.merge_asof(noninv,inv,by='patientunitstayid',
                         on='observationoffset',tolerance=5,
                         direction='nearest')
    overlap.dropna(inplace=True)
    overlap['stay_offset'] = (overlap['patientunitstayid'].astype(str) + '_' + 
                              overlap['observationoffset'].astype(str))
    # Remove non invasive data with those stay/offset combos. 
    noninv['stay_offset'] = (noninv['patientunitstayid'].astype(str) + '_' + 
                             noninv['observationoffset'].astype(str))
    noninv = noninv[~(noninv['stay_offset'].isin(overlap['stay_offset']))]
    noninv.drop(columns=['stay_offset'],inplace=True)
    noninv.dropna(inplace=True)
    # Combine data together. 
    inv.rename(columns={'systemic' + bp_type: 'all' + bp_type}, inplace=True)
    noninv.rename(columns={'noninvasive' + bp_type: 'all' + bp_type}, inplace=True)
    both = pd.concat([inv,noninv])
    both.sort_values(['patientunitstayid','observationoffset'],inplace=True)
    # resample down to 1 minute.
    stays = both['patientunitstayid'].drop_duplicates()
    df_list = list()
    for stay in stays:
        temp_data = both[both['patientunitstayid'] == stay]
        first = temp_data.iloc[0,1]
        last = temp_data.iloc[-1,1]
        temp_data = temp_data.set_index('observationoffset')
        temp_data = temp_data.reindex(list(range(first,last)))
        temp_data.reset_index(inplace=True)
        temp_data['patientunitstayid'] = stay
        df_list.append(temp_data)
    both = pd.concat(df_list)
        
    # Interpolate 2 hr gaps one more time.
    both = interp_thresh(both,120,'all' + bp_type)
    # Save off data. 
    both.to_csv('pre-processed_all' + bp_type + '.csv', index=False)
    print(bp_type + ' done!')

#%% For testing data.
# inv = pd.read_csv('all_pre-processed_systemic' + bp_type + '.csv',nrows=100000)

# noninv = pd.read_csv('all_pre-processed_noninvasive' + bp_type + '.csv',nrows=100000)