# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 12:21:39 2021

Generates demographics table of the ICU readmission dataset in eICU. 

V2 has changes for readability, modularization, and efficiency, as well as a 
few new additional outcomes. Also adds mann-whitney and chi-squared testing. 

@author: Kirby
"""

#%% Package setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
from time import time

start = time()

file_path = Path(__file__)
eicu_path = file_path.parent.parent.parent.joinpath('eicu')
hist_path = file_path.parent.parent.joinpath('Features','History')
comorb_path = file_path.parent.parent.joinpath('Features','Comorbidity')

#%% Import data. 
comp = pd.read_csv('ICU_readmissions_dataset.csv')

pat = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','gender', 'age',
                           'ethnicity','unitdischargeoffset',
                           'unitadmitsource','hospitaldischargestatus'])
apache = pd.read_csv(eicu_path.joinpath('apachepatientresult.csv'),
                     usecols=['patientunitstayid','apachescore',
                              'apacheversion'])
diag = pd.read_csv(eicu_path.joinpath('admissiondx.csv'),
                   usecols=['patientunitstayid','admitdxname'])
names = pd.read_csv('admissiondx_organ_system_paths.csv')

#%% Data cleaning.

#Get just patient stays we care about.
for data in [pat,apache,diag]:
    data.drop(data[
        ~data['patientunitstayid'].isin(comp['patientunitstayid'])].index,
              inplace=True)

#Just get organ systems from diag. 
diag = diag[diag['admitdxname'].isin(names['admitdxname'])]

#Convert age to numeric.
def age_to_nums(age):
    if age == '> 89':
        return 90
    else:
        return float(age)

pat['age'] = pat['age'].apply(age_to_nums)

#Get just apacheiv scores, make -1s nans.
apache = apache[apache['apacheversion']=='IV']
apache = apache[['patientunitstayid','apachescore']]
apache = apache[~(apache['apachescore']==-1)]
apache = apache.merge(pat[['patientunitstayid']],on='patientunitstayid',
                      how='right')

#Make unitdischargeoffset into hours. 
pat['unitdischargeoffset'] = pat['unitdischargeoffset']/60

# Combine rare admission sources into one "Other" category. 
def other_adm_sources(admit_source):
    if admit_source in ['Operating Room','Recovery Room','PACU','Floor',
                        'Emergency Department']:
        return admit_source
    else:
        return 'Other'
pat['unitadmitsource'] = pat['unitadmitsource'].apply(other_adm_sources)

#%% Get separate groups. 
#Which patients were postive? Which were negative?
readm_pos = comp[comp['bad_disch_plan']==1]['patientunitstayid']
readm_neg = comp[comp['bad_disch_plan']==0]['patientunitstayid']

#Merge all the data together. 
pat = pat.merge(apache,on='patientunitstayid',how='left')
pat = pat.merge(diag,on='patientunitstayid',how='left')

#Get positive only versions of all extracted info. 
pat_pos = pat[pat['patientunitstayid'].isin(readm_pos)]

#Get negative only versions of all extracted info. 
pat_neg = pat[pat['patientunitstayid'].isin(readm_neg)]

#%% Get demographic data.

#Add proportion function. Takes a column of raw categorical data, gets counts, 
#and adds proportion while converting to string. 
def add_prop(col):
    col = col.value_counts()
    return ('   ' + col.astype(str) + ' (' + 
            (np.round(col/(col.sum())*100,decimals=2)).astype(str) + '%)')

#Gets proportion and counts for each cohort, makes a dataframe of it.
def all_prop(col_name):
    df = pd.DataFrame(columns=['No Readmit/Death','Readmit/Death','Total','p-Value'])
    df['No Readmit/Death'] = add_prop(pat_neg[col_name])
    df['Readmit/Death'] = add_prop(pat_pos[col_name])
    df['Total'] = add_prop(pat[col_name])
    #Chi squared testing, 
    #Find if difference between no delirium/delirium groups is significant. 
    num_df = df[['No Readmit/Death','Readmit/Death']].copy()
    #Convert the data to just proportions.
    for col in num_df.columns:
        #Replace nans with 0s. 
        num_df.replace(to_replace=np.nan,value='(0%',inplace=True)
        #Parse string to get percentage.
        num_df[col] = num_df[col].str.split('(',1)
        num_df[col] = num_df[col].apply(lambda x: x[1])
        num_df[col] = num_df[col].str.split('%',1)
        num_df[col] = num_df[col].apply(lambda x: x[0])
        num_df[col] = num_df[col].astype(float)
    chisq,p = chisquare(num_df[['Readmit/Death']],num_df[['No Readmit/Death']])
    df.iloc[0,3] = np.round(p[0],decimals=3)
    return df

#Gender
gender = all_prop('gender')
#Ethnicity
ethn = all_prop('ethnicity')
#Hospital Mortality
mort = all_prop('hospitaldischargestatus')
#Unit Admit Source
source = all_prop('unitadmitsource')
#Primary AdmissionDx Group
groups = all_prop('admitdxname')

#Takes in raw data column, outputs string of median and IQR.
def get_med_iqr(col):
    return (str(np.round(col.median(),decimals=2)) + ' [' + 
            str(np.round(col.quantile(q=0.25),decimals=2)) + '-' + 
            str(np.round(col.quantile(q=0.75),decimals=2)) + ']')

#Takes in raw data column,a title and a unit, makes IQR dataframe out of it 
#for each cohort. 
def all_med_iqr(col_name,title,units):
    neg_col = pat_neg[col_name]
    pos_col = pat_pos[col_name]
    df = pd.DataFrame(columns=['No Readmit/Death','Readmit/Death','Total','p-Value'])
    if units == '':
        full_title = 'Median '+ title + ' [IQR]'
    else:
        full_title = 'Median '+ title + ' [IQR] (' + units + ')'
    df.loc[full_title,'No Readmit/Death'] = get_med_iqr(neg_col)
    df.loc[full_title,'Readmit/Death'] = get_med_iqr(pos_col)
    df.loc[full_title,'Total'] = get_med_iqr(pat[col_name])
    #Mann Whitney U test.
    #Find if difference between no delirium/delirium groups is significant. 
    u,p = mannwhitneyu(neg_col,pos_col,alternative='two-sided')
    df.iloc[0,3] = np.round(p,decimals=3)
    return df

#Age
age = all_med_iqr('age','Age','Years')
#ICU LOS
los = all_med_iqr('unitdischargeoffset','ICU LOS','Hours')
#APACHE Score
score = all_med_iqr('apachescore','First 24 Hour APACHE IV Score','')



#%% Put all of it together.

#Function that returns a header dataframe for formatting to concat in.
def header(title):
    df = pd.DataFrame(columns=['No Readmit/Death','Readmit/Death','Total'])
    df.loc[title] = np.nan
    return df

final = pd.concat([header('Patient Characteristics'),
                   header('Gender'),
                   gender,
                   header('Age'),
                   age,
                   header('Ethnicity'),
                   ethn,
                   score,
                   header('Admission Source'),
                   source,
                   header('Diagnostic Groupings'),
                   groups,
                   header('Outcomes'),
                   los,
                   header('Hospital Mortality'),
                   mort])

final.to_csv('ICU_patient_demographics.csv')

runtime = time() - start