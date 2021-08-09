# -*- coding: utf-8 -*-
"""
Created on Tue May 18 17:27:58 2021

Analyzes overlap of invasive/non-invasive BP data with tolerances. 

Runtime: 1 minute.

@author: Kirby
"""

#%% Package setup
import numpy as np
import pandas as pd
import os
#import multiprocessing as mp
from time import time
import statistics as stat
from pathlib import Path
import matplotlib.pyplot as plt

start = time()

file_path = Path(__file__)
parent = file_path.parent
cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')

#%% Load in data.
#Get the patient ids.
comp = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))

# Get LOS data.
los = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','unitdischargeoffset'])
los.rename(columns={'unitdischargeoffset':'los'},inplace=True)
# Make it into days.
los.loc[:,'los'] = los['los']/1440

# Dataframe to save off results to.
overlap = pd.DataFrame()

# Just get the column names for separate BP data. 
cols = ['systolic','diastolic','mean']

# Set tolerance in minutes.
tol = 5


#%% Find overlapping data for each type of BP data.
overlap = dict()
prop_overlap = dict()
for col in cols:
    #Load the pre-processed PTS data. 
    inv = pd.read_csv('pre-processed_systemic' + col +'.csv')
    noninv = pd.read_csv('pre-processed_noninvasive' + col +'.csv')
    #Pre-processing already removed patient stays I don't care about.
    
    #Find data from both sources at the same time within tolerance.
    id_and_offset = ['patientunitstayid','observationoffset']
    # Sort it by observation offset so merge_asof can work.
    inv.sort_values('observationoffset',inplace=True)
    noninv.sort_values('observationoffset',inplace=True)
    both = pd.merge_asof(inv, noninv, by='patientunitstayid',
                          on='observationoffset', tolerance=tol,
                          direction='nearest').dropna()
    both.drop_duplicates(inplace=True)
    
    both['diff_' + col] = both['systemic' + col] - both['noninvasive' + col]
    
    overlap[col] = both
    
    prop_overlap[col] = both.shape[0]/inv.shape[0]

#%% Get pulse pressure data. 
# Combine systolic and diastolic into one DF.
sys = overlap.get('systolic')
dia = overlap.get('diastolic')
mea = overlap.get('mean')
pulse = sys.merge(dia, on=id_and_offset, how='inner')
pulse = pulse.merge(mea, on=id_and_offset, how='inner')

# Calculate pulse pressure.
pulse['systemicpulse'] = pulse['systemicsystolic'] - pulse['systemicdiastolic']
pulse['noninvasivepulse'] = pulse['noninvasivesystolic'] - pulse['noninvasivediastolic']
pulse['diff_pulse'] = pulse['systemicpulse'] - pulse['noninvasivepulse']

#%% Analysis looking for signs monitor is failing. 

# Filter data based on a column value being less than threshold, 
# find proportion of data affected, how many icu stays, and los distribution.
def filter_on_col(col,thresh,pulse):
    temp = pulse[pulse[col] < thresh]
    temp_prop = temp.shape[0]/pulse.shape[0]
    temp_stays = temp[['patientunitstayid']].drop_duplicates()
    all_stays = pulse['patientunitstayid'].drop_duplicates().shape[0]
    temp_stays_prop = temp_stays.shape[0]/all_stays
    temp_los = temp_stays.merge(los,on='patientunitstayid',how='left')
    plt.figure()
    plt.title(col + ' < ' + str(thresh) + ' LOS dist')
    temp_los['los'].hist()
    plt.figure()
    plt.title(col + ' < ' + str(thresh) + ' LOS dist, LOS < 10 days')
    temp_los[temp_los['los'] < 10]['los'].hist()
    return temp, temp_prop, temp_stays, temp_stays_prop

# How much of the invasive pulse pressure is less than 10? 
pulse10, pulse10prop, pulse10stays, pulse10staysprop = filter_on_col('systemicpulse',10,pulse)

# How often is the non-invasive systolic 20 more than invasive systolic?
sys20,sys20prop, sys20stays, sys20staysprop = filter_on_col('diff_systolic',-20,pulse)

# How often is the non-invasive mean 20 more than invasive mean?
mean20, mean20prop, mean20stays, mean20staysprop = filter_on_col('diff_mean',-20,pulse)

#%% Generate histograms of the differences. 
    
#Function for making histogram figures.
def make_histo(data,title,xlabel,filename,folder):
    plt.figure()
    plt.xlabel(xlabel)
    plt.title(title)
    data.hist()
    plt.savefig(parent.joinpath(folder,filename + '.png'))

folder = 'BP Overlap ' + str(tol) + ' min'

# Histogram of pulse pressure differences. 
make_histo(pulse['diff_pulse'],
           ' Pulse Pressure Difference (Systemic - Noninvasive)', 
           'difference (mmHg)', 'pulse', folder)
# Zoomed in histogram. 
small_pulse = pulse['diff_pulse'].where(lambda x : abs(x) < 50).dropna()
make_histo(small_pulse,
           'Pulse Pressure Difference (Systemic - Noninvasive), < 50 abs diff', 
           'difference (mmHg)', 'small_pulse', folder)
# Differences of systolic, diastolic, and mean data. 
for col in cols:
    data = overlap.get(col)
    data = data['diff_' + col]
    make_histo(data,col + ' Difference (Systemic - Noninvasive)',
               'difference (mmHg)',col + '_diff', folder)
    data = data.where(lambda x : abs(x) < 50).dropna()
    make_histo(data,col + ' Difference (Systemic - Noninvasive), < 50 abs diff',
               'difference (mmHg)',col + '_diff_small', folder)


calc = time() - start