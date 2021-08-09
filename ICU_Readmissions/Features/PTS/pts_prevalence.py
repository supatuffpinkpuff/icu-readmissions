# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:27:58 2021

Analyzes prevalence of HR, BP, RR, SaO2. 


Runtime: 10 minute.

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

#Get ICU LOS data. 
los = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','unitdischargeoffset'])
los = los[los['patientunitstayid'].isin(comp['patientunitstayid'])]

# Get last 24h denominator, either 24h or LOS, whichever is shorter. 
def get_denom(los):
    if los <= 1440:
        return los
    else: 
        return 1440
los['denom_24h'] = los['unitdischargeoffset'].apply(get_denom)

#DF to save prevalence data to. 
prev = pd.DataFrame(columns=['prevalence','mean_counts','median_counts',
                             'std_counts'])

cols = ['sao2','heartrate','respiration','allsystolic','alldiastolic','allmean']

#Save off which stays had that pts/didn't. 
stays_prev = comp[['patientunitstayid']].copy()

# Save of proportion of coverage for each stay.
stays_prop = comp[['patientunitstayid']].copy()
stays_prop24 = comp[['patientunitstayid']].copy()

#%% Receive a lab name to look for.
for col in cols:
    #Load the pre-processed PTS data. 
    pts = pd.read_csv('pre-processed_' + col +'.csv')
    # Remove patient stays I don't care about.
    pts = pts[pts['patientunitstayid'].isin(comp['patientunitstayid'])]
    
    #Remove nans.
    pts.dropna(inplace=True)
    pts.drop_duplicates(inplace=True)
    
    #Get stays that had this pts. 
    stays_with_pts = pts['patientunitstayid'].drop_duplicates()
    overall = stays_with_pts.shape[0]
    #Save off which patients had/didn't have the PTS. 
    stays_prev[col] = stays_prev['patientunitstayid'].isin(stays_with_pts)
    #Get overall prevalence. 
    prev.loc[col,'prevalence'] = overall/comp.shape[0]
    
    #Function for making histogram figures.
    def make_figs(data,title,xlabel,col,threshold,folder):
        plt.figure()
        plt.xlabel(xlabel)
        plt.title(title + ', ' + col)
        data.hist()
        plt.savefig(parent.joinpath(folder,col + '.png'))
        plt.figure()
        plt.xlabel(xlabel)
        plt.title(title + ', below threshold, ' + col)
        data[data <= threshold].hist()
        plt.savefig(parent.joinpath(folder,col + '_short.png'))
    
    #Function for saving off mean, median, standard deviations. 
    def get_stats(data,data_name,col):
        prev.loc[col,'mean_' + data_name] = data.mean()
        prev.loc[col,'median_' + data_name] = data.median()
        prev.loc[col,'std_' + data_name] = data.std()
    
    #Counts per ICU stay. 
    counts = pts.groupby('patientunitstayid').count()['observationoffset']
    get_stats(counts,'counts',col)
    make_figs(counts,'Counts Per Stay','Counts',col,1000,'Counts Per Stay Histograms')
    
    #Frequency of PTS values.
    freq = pts.copy()
    freq['last_stay_same'] = (freq['patientunitstayid'].shift(periods=1) == 
                              freq['patientunitstayid'])
    freq['time_between_pts'] = freq['observationoffset'].diff()
    #Get minutes between pts.
    freq = freq[freq['last_stay_same'] == True]['time_between_pts']
    get_stats(freq,'freq',col)
    make_figs(freq,'Days Between Lab Values','Minutes',col,720,'Frequency Histograms')
    
    #Counts per LOS. 
    #Get counts with stays.
    counts_los = pts.groupby('patientunitstayid').count().reset_index()
    #Attach LOS.
    counts_los = counts_los.merge(los,on='patientunitstayid',how='left')
    #Divide out count of pts per day. 
    counts_los = counts_los[col]/counts_los['unitdischargeoffset']*1440
    get_stats(counts_los,'count_by_los',col)
    make_figs(counts_los,'Counts Per Day LOS','pts/Day of LOS',col,300,'Counts Per LOS Histograms')
    
    #Size of gaps.
    #Get data where it's the same stay. 
    gaps = pts.copy()
    gaps['last_stay_same'] = (gaps['patientunitstayid'].shift(periods=1) == 
                              gaps['patientunitstayid'])
    #Get data where the time between data.
    gaps['time_between_pts'] = gaps['observationoffset'].diff()
    #Get where gap is longer than 5 min, and it's the same stay. 
    gaps = gaps[(gaps['time_between_pts'] > 5) & 
                (gaps['last_stay_same'] == True)]
    #Get just the gap data, in minutes.
    gaps = gaps['time_between_pts']
    get_stats(gaps,'gap_length',col)
    make_figs(gaps,'Gap Length','Gap Length (Minutes)',col,720,'Gap Size Histograms')    
    
    #Proportion of coverage. Assumes each data point covers individual 5 min.
    #Drop duplicates. 
    prop = pts.drop_duplicates()
    prop = prop[['patientunitstayid','observationoffset']].drop_duplicates()
    prop = prop.groupby('patientunitstayid').count().reset_index()
    #Attach LOS.
    prop = prop.merge(los,on='patientunitstayid',how='left')
    #Each data point covers 5 minutes, or 1 min for BP data. 
    if col in ['sao2','respiration','heartrate']:
        prop['observationoffset'] = prop['observationoffset']*5
    prop['prop'] = prop['observationoffset']/prop['unitdischargeoffset']
    # Get stats and make histograms.
    get_stats(prop['prop'],'prop_covered',col)
    make_figs(prop['prop'],'Proportion of LOS Covered','Proportion',col,1,'Coverage Proportion')
    # Save off proportion of last 24h covered per stay for each PTS.
    prop = prop[['patientunitstayid','prop']]
    prop.rename(columns={'prop':'prop' + col},inplace=True)
    stays_prop = stays_prop.merge(prop,on='patientunitstayid',how='left')
    
    #Proportion of coverage of last 24h. Assumes each data point covers individual 5 min.
    #Drop duplicates. 
    prop_24 = pts.drop_duplicates()
    #Attach LOS.
    prop_24 = prop_24.merge(los,on='patientunitstayid',how='left')
    # Drop data before 24h before discharge. 
    prop_24['cutoff'] = prop_24['unitdischargeoffset'] - 1440
    prop_24 = prop_24[prop_24['observationoffset'] >= prop_24['cutoff']]
    # Count up number of points. 
    prop_24 = prop_24[['patientunitstayid','observationoffset']].drop_duplicates()
    prop_24 = prop_24.groupby('patientunitstayid').count().reset_index()
    #Each data point covers 5 minutes, unless its BP in which case leave it at 1 min.
    if col in ['sao2','respiration','heartrate']:
        prop_24['observationoffset'] = prop_24['observationoffset']*5
    # Divide it out by LOS or 24 h.
    prop_24 = prop_24.merge(los, on='patientunitstayid', how='left')
    prop_24['prop_24'] = prop_24['observationoffset']/prop_24['denom_24h']
    # Get stats and make histograms.
    get_stats(prop_24['prop_24'],'prop_covered_24h',col)
    make_figs(prop_24['prop_24'],'Proportion of LOS Covered Last 24h','Proportion',col,1,
              'Coverage Proportion 24h')
    # Save off proportion of last 24h covered per stay for each PTS.
    prop_24 = prop_24[['patientunitstayid','prop_24']]
    prop_24.rename(columns={'prop_24':'prop_24' + col},inplace=True)
    stays_prop24 = stays_prop24.merge(prop_24,on='patientunitstayid',how='left')
    
#Round values for readability. 
for col in prev.columns:
    prev[col] = prev[col].astype(float).round(decimals=3)

#%% Examine stays that would be excluded based on coverage thresholds.

# Get how many stays meet threshold requirements for SaO2, RR, HR.
# And pull class imbalance in those data set. 
def stays_above_thresh(stays_prop,thresh):
    temp = stays_prop[(stays_prop.iloc[:,1] > thresh) &
                      (stays_prop.iloc[:,2] > thresh) &
                      (stays_prop.iloc[:,3] > thresh) & 
                      (stays_prop.iloc[:,4] > thresh) &
                      (stays_prop.iloc[:,5] > thresh) &
                      (stays_prop.iloc[:,6] > thresh)]
    prop_data_kept = temp.shape[0]/stays_prop.shape[0]
    labels = comp[['patientunitstayid','bad_disch_plan']]
    labels = labels.merge(temp,on='patientunitstayid',how='right')
    class_count = labels['bad_disch_plan'].value_counts()
    total_stays = class_count.sum()
    class_balance = class_count[1]/class_count.sum()
    return prop_data_kept,total_stays,class_balance

stays_prop_50 = stays_above_thresh(stays_prop,.50)
stays_prop24_50 = stays_above_thresh(stays_prop24,.50)
stays_prop_80 = stays_above_thresh(stays_prop,.80)
stays_prop24_80 = stays_above_thresh(stays_prop24,.80)

#%% Figure out which patients had individual PTS, which had all of em. 
#Function to get series of stays that didn't have pts in list from stays_prev. 
def get_stays_without_pts(stays_prev,pts_list):
    for pts in pts_list:
        stays_prev = stays_prev[stays_prev[pts] == False]
    return stays_prev[['patientunitstayid']]

#Function to make LOS histogram figures for stays in and out of group. 
def los_hists(group,group_desc):
    plt.figure()
    plt.xlabel('LOS (days)')
    plt.title('LOS of stays with ' + group_desc)
    temp = los[los['patientunitstayid'].isin(group['patientunitstayid'])]
    temp.loc[:,'unitdischargeoffset'] = temp['unitdischargeoffset']/1440
    temp[temp['unitdischargeoffset'] <= 3]['unitdischargeoffset'].hist()
    #temp['unitdischargeoffset'].hist(
    plt.savefig(parent.joinpath('LOS of Missing PTS',group_desc + '_yes.png'))
    
    plt.figure()
    plt.xlabel('LOS (days)')
    plt.title('LOS of stays without ' + group_desc)
    temp = los[~los['patientunitstayid'].isin(group['patientunitstayid'])]
    temp.loc[:,'unitdischargeoffset'] = temp['unitdischargeoffset']/1440
    temp[temp['unitdischargeoffset'] <= 3]['unitdischargeoffset'].hist()
    #temp['unitdischargeoffset'].hist()
    plt.savefig(parent.joinpath('LOS of Missing PTS',group_desc + '_no.png'))
    
#Find which patients were missing all pts. 253/30018
no_pts = get_stays_without_pts(stays_prev,cols)
#Find LOS distribution of those who had and didn't. 
los_hists(no_pts,'No PTS')

#Save off results.
prev.to_csv('PTS_prevalence.csv')
stays_prop.to_csv('PTS_proportion_covered_whole_stay.csv',index=False)
stays_prop24.to_csv('PTS_proportion_covered_24h.csv',index=False)

calc = time() - start