# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:41:00 2021

Search careplangeneral,apacheapsvar, and treatment for dialysis information.
Generates a binary feature that marks if a patient had dialysis before/during
the ICU Stay.

Runtime: 30 seconds

@author: Kirby
"""
def full_script():

    #%% Import packages.
    import numpy as np
    import pandas as pd
    import os
    #import multiprocessing as mp
    from time import time
    from pathlib import Path
    
    start = time()
    filepath = Path(__file__)
    eicu_path = filepath.parent.parent.parent.parent.joinpath('eicu')
    cohort_path = filepath.parent.parent.parent.joinpath('Cohort')
    
    #%% Load in data. 
    
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    
    # Attach LOS as end, make admission start of window. 
    ids = ids.merge(pat,on='patientunitstayid',how='left')
    ids.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    ids['start'] = 0
    
    cpl = pd.read_csv(eicu_path.joinpath('CarePlanGeneral.csv'),
                      usecols=['patientunitstayid','cplitemoffset',
                               'cplitemvalue'])
    
    apache = pd.read_csv(eicu_path.joinpath('ApacheApsVar.csv'),
                         usecols=['patientunitstayid','dialysis'])
    
    treat = pd.read_csv(eicu_path.joinpath('Treatment.csv'),
                        usecols=['patientunitstayid', 'treatmentoffset',
                                 'treatmentstring'])
    
    #%% Filter out irrelevant rows.
    
    #Just get data for our patient stays.
    for data in [cpl,apache,treat]:
        drop_index = data[~data['patientunitstayid'].isin(
            ids['patientunitstayid'])].index
        data.drop(drop_index, inplace=True)
    
    #Drop data after the observation window for each patient. 
    lookup = ids.set_index('patientunitstayid')
    def keep_row(current_ID,offset):
        #Get window time stamps.
        window_start = lookup.loc[current_ID,'start']
        window_end = lookup.loc[current_ID,'end']
        #If the dialysis took place before/in window, keep it. 
        if (offset <= window_end):
            return 1
        else:
            return 0
    
    #Drop cpl data outside desired time frame. 
    cpl['keep'] = cpl.apply(lambda row: keep_row(
        row['patientunitstayid'],row['cplitemoffset']),axis=1)
    cpl = cpl[cpl['keep']==1]
    cpl = cpl[cpl['cplitemvalue']=='Dialysis']
    dialysis = cpl['patientunitstayid']
    
    apache = apache[apache['dialysis']==1]
    dialysis = dialysis.append(apache['patientunitstayid'],
                               ignore_index=True)
    
    #Drop treat data outside desired time frame. 
    treat['keep'] = treat.apply(lambda row: keep_row(
        row['patientunitstayid'],row['treatmentoffset']),axis=1)
    treat = treat[treat['keep']==1]
    treat = treat[treat['treatmentstring'].str.contains('dialysis')]
    dialysis = dialysis.append(treat['patientunitstayid'],
                               ignore_index=True)
    
    dialysis.drop_duplicates(inplace=True)
    
    ids['dialysis'] = ids['patientunitstayid'].isin(dialysis).astype(int)
    ids.to_csv('dialysis_feature.csv',index=False)
    
    calc = time() - start


if __name__ == '__main__':
    full_script()