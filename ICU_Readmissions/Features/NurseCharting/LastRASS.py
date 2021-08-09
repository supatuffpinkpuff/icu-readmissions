# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 11:39:10 2020

Get last RASS data before/in the observation window, taken from the
nursecharting table.

Runtime: 2 minutes.

@author: Kirby
"""
def full_script():
    #%% Package setup
    import pandas as pd
    import numpy as np
    import time
    import datetime
    from pathlib import Path
    
    start_timer = time.time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
    eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')
    
    #%% Load in data. 
    
    #Pulls list of Stay IDs and offsets we care about.
    
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    
    # Attach LOS as end, make admission start of window. 
    ids = ids.merge(pat,on='patientunitstayid',how='left')
    ids.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    ids['start'] = 0
    
    keep_list = ['RASS','SEDATION SCORE','Sedation Scale/Score/Goal']
    #Just get rass data.
    rass_data = pd.read_csv(eicu_path.joinpath("nurseCharting.csv"),
                                   nrows=0,
                                   usecols=['patientunitstayid',
                                            'nursingchartoffset',
                                            'nursingchartcelltypevallabel',
                                            'nursingchartcelltypevalname',
                                            'nursingchartvalue'])
    for chunk in pd.read_csv(eicu_path.joinpath("nurseCharting.csv"), 
                             chunksize=1000000,
                             usecols=['patientunitstayid','nursingchartoffset',
                                      'nursingchartcelltypevallabel',
                                      'nursingchartcelltypevalname',
                                      'nursingchartvalue']):
        rass_rows = chunk[chunk['nursingchartcelltypevallabel'].isin(keep_list)]
        rass_data = pd.concat([rass_data,rass_rows])
    
    #%% Clean up and combine the RASS data. 
    #Get the 'RASS' data.
    rass = rass_data[rass_data['nursingchartcelltypevallabel']=='RASS']
    #Get the 'SEDATION SCORE' data.
    caps_score = rass_data[rass_data['nursingchartcelltypevallabel']==
                           'SEDATION SCORE']
    #Get the other data. 
    scale = rass_data[rass_data['nursingchartcelltypevallabel']==
                            'Sedation Scale/Score/Goal']
    scale = scale[scale['nursingchartcelltypevalname']=='Sedation Scale']
    #Drop the data that isn't RASS.
    scale = scale[scale['nursingchartvalue']=='RASS']
    scale = scale[['patientunitstayid','nursingchartoffset']]
    #Get the scores.
    score = rass_data[rass_data['nursingchartcelltypevallabel']==
                            'Sedation Scale/Score/Goal']
    score = score[score['nursingchartcelltypevalname']=='Sedation Score']
    #Only keep the scores that were RASS.
    score = score.merge(scale,on=['patientunitstayid','nursingchartoffset'],
                        how='inner')
    
    #Combine all 3 sources of RASS data.
    rass_data = pd.concat([rass,caps_score,score])
    rass_data = rass_data[['patientunitstayid', 'nursingchartoffset',
           'nursingchartvalue']]
    
    #%% Process the data.
    #Just get data on patients we care about. 
    rass_data = rass_data[
        rass_data['patientunitstayid'].isin(ids['patientunitstayid'])]
    
    #Drop data after the observation window for each patient. 
    lookup = ids.set_index('patientunitstayid')
    def keep_row(current_ID,offset):
        #Get window time stamps.
        window_start = lookup.loc[current_ID,'start']
        window_end = lookup.loc[current_ID,'end']
        #If the rass score took place before/in window, keep it. 
        if (offset <= window_end):
            return 1
        else:
            return 0
    
    rass_data['keep'] = rass_data.apply(
        lambda row: keep_row(
            row['patientunitstayid'],row['nursingchartoffset']),axis=1)
    rass_data = rass_data[rass_data['keep']==1]
    
    #Make the data all numeric.
    rass_data['nursingchartoffset'] = pd.to_numeric(
        rass_data['nursingchartoffset'],errors='coerce')
    
    #Discard columns I don't care about.
    rass_data = rass_data[[
        'patientunitstayid','nursingchartoffset','nursingchartvalue']]
    
    #%% Get the last rass value for each patient in the observation window. 
    
    #Make sure all the data's in order by patientstayid and offset.
    rass_data.sort_values(['patientunitstayid','nursingchartoffset'],
                                 inplace=True)
    
    #Generate column of last rass for each ID.
    last_rass = rass_data.groupby('patientunitstayid').last().reset_index(
        drop=False)
    last_rass.rename(columns={'nursingchartvalue':'last_rass'},inplace=True)
    ids = ids.merge(last_rass,how='left',on='patientunitstayid')
    
    #Save off results.
    ids = ids[['patientunitstayid','last_rass']]
    ids.to_csv('rass_feature.csv',index=False)
    
    #For performance testing. 
    calculation_timer = time.time()-start_timer
    

if __name__ == '__main__':
    full_script()