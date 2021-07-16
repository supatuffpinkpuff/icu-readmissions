# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 20:15:15 2021

Pulls a number of features regarding intake output, include total intake, 
total output, net total, urine output, blood loss, and blood product 
transfusions during the ICU stay.

Runtime: 1 minute.

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
    
    #%% Load data 
    
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    
    # Attach LOS as end, make admission start of window. 
    ids = ids.merge(pat,on='patientunitstayid',how='left')
    ids.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    ids['start'] = 0
    
    io = pd.read_csv(eicu_path.joinpath('IntakeOutput.csv'),
                     usecols=['patientunitstayid', 'intakeoutputoffset',
                              'intaketotal','outputtotal','nettotal',
                              'cellpath', 'cellvaluenumeric'])
    
    treat = pd.read_csv(eicu_path.joinpath('treatment.csv'),
                     usecols=['patientunitstayid', 'treatmentoffset',
                              'treatmentstring'])
    
    #%% Filter data.
    for data in [io,treat]:
        #Just our patients.
        pats = ids['patientunitstayid']
        data.drop(data[~(data['patientunitstayid'].isin(pats))].index,
                  inplace=True)
        
    #Drop data after the observation window for each patient. 
    lookup = ids.set_index('patientunitstayid')
    def keep_row(current_ID,offset):
        #Get window time stamps.
        window_start = lookup.loc[current_ID,'start']
        window_end = lookup.loc[current_ID,'end']
        #If the surgery took place before/in window, keep it. 
        if (offset <= window_end):
            return 1
        else:
            return 0
    
    io['keep'] = io.apply(lambda row: keep_row(
        row['patientunitstayid'],row['intakeoutputoffset']),axis=1)
    io = io[io['keep']==1]
    
    treat['keep'] = treat.apply(lambda row: keep_row(
        row['patientunitstayid'],row['treatmentoffset']),axis=1)
    treat = treat[treat['keep']==1]
    
    #%% Total intake, output, and net. 
    #Commented out, because the data's clearly wrong per Dr. Stevens. 
    #We're not going to use this data for features. Like the data is missing 
    #documentation of fluids. 
    
    # totals = io.groupby('patientunitstayid').last().reset_index()
    # totals = totals[['patientunitstayid','intaketotal', 'outputtotal',
    #                  'nettotal']]
    
    #Get hists of this info. 
    # totals['intaketotal'].hist()
    # totals[totals['intaketotal']<500]['intaketotal'].hist()
    # totals['outputtotal'].hist()
    # totals[totals['outputtotal']<500]['outputtotal'].hist()
    # totals['nettotal'].hist()
    # totals[(totals['nettotal']>-500) & (totals['nettotal']<500)]['nettotal'].hist()
    
    #%% Find urine information.
    urine_paths = pd.read_csv('urinepaths.csv')
    urine = io.merge(urine_paths,on='cellpath',how='inner')
    urine = urine[urine['cellvaluenumeric']>=0]
    #Add on end times.
    ends = ids[['patientunitstayid','end']]
    urine = urine.merge(ends,on='patientunitstayid',how='left')
    #Only get urine 24hrs before end of observation window.
    urine = urine[urine['intakeoutputoffset']>=(urine['end']-1440)]
    urine = urine.groupby('patientunitstayid').sum().reset_index()
    urine = urine[['patientunitstayid','cellvaluenumeric']]
    urine.rename(columns={'cellvaluenumeric':'last_24hr_urine'},
                 inplace=True)
    ids = ids.merge(urine,on='patientunitstayid',how='left')
    
    #Get hists of info.
    urine['last_24hr_urine'].hist()
    urine[urine['last_24hr_urine']<1000]['last_24hr_urine'].hist()
    
    #%%Blood transfusions and loss. 
    
    #Create binary variables for transfusions.
    path_files = ['RBCpaths.csv','plasmapaths.csv',
                  'plateletpaths.csv']
    strings = ['packed red blood cells',
               'fresh frozen plasma|cryo-poor plasma|cryoprecipitate',
               'coagulation factors|platelet concentrate']
    col_names = ['tranfuse_rbc','tranfuse_plasma',
                 'tranfuse_platelet']
    for i in range(0,3):
        #Get data from Intake/Output.
        temp_paths = pd.read_csv(path_files[i])
        temp_io = io.merge(temp_paths,on='cellpath',how='inner')
        temp_io = temp_io['patientunitstayid'].drop_duplicates()
        #Get data from treatment.
        treat_str = strings[i]
        temp_treat = treat[treat['treatmentstring'].str.contains(
            treat_str,case=False)]
        temp_treat = temp_treat['patientunitstayid'].drop_duplicates()
        both = temp_io.append(temp_treat).drop_duplicates()
        ids[col_names[i]] = \
            ids['patientunitstayid'].isin(both).astype(int)
            
    #Get blood loss. Prevalence too low, casts doubt on the data quality.
    # paths = pd.read_csv('bloodlosspaths.csv')
    # col_name = 'first_24hr_blood_loss'
    # data = io.merge(temp_paths,on='cellpath',how='inner')
    # data = data.groupby('patientunitstayid').sum().reset_index()
    # data = data[['patientunitstayid','cellvaluenumeric']]
    # data.rename(columns={'cellvaluenumeric':col_name},inplace=True)
    # comp = data.merge(comp,on='patientunitstayid',how='right')
    # comp[col_name] = comp[col_name].fillna(0)
    
    #Save off features.
    ids.to_csv('urine_transfusions_features.csv',index=False)
    
    #Get prevalences.
    prev = dict()
    for col in col_names:
        prev.update({col:ids[col].value_counts()})
    
    calc = time() - start
    
if __name__ == '__main__':
    full_script()