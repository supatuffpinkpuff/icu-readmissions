# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:57:28 2020

#This pulls the value the last lab for each patient stay.
#Must be run after AllLabsDuringStay.py
 
Run time: 10 min

@author: Kirby
"""
def full_script():

    #%% Package setup
    import numpy as np
    import pandas as pd
    import os
    #import multiprocessing as mp
    from time import time
    import statistics as stat
    from pathlib import Path
    
    start = time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
    
    #%% Inputs
    #Get list of lab names
    lab_list = pd.read_csv("LabsList.csv")
    #Uncommet if only running part of the labs.
    # lab_list = pd.read_csv("TempLabsList.csv")
    
    lab_list = lab_list.transpose()
    lab_list = lab_list.values.tolist()[0]
    
    #Get the patient ids.
    comp = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get normal values. 
    norms = pd.read_excel('feature_normal_ranges_KG.xlsx')
    norms = norms.set_index('Feature')
    norms = norms.to_dict().get('Normal Value')
    
    
    #%% Receive a lab name to look for.
    for lab_name in lab_list:
        
        #%% Pulls list of Stay IDs and offsets we care about.
        ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
        
        #Load the list of the relevant labs
        script_dir = os.path.dirname(__file__)
        folder = "AllLabsDuringStay"
        full_file_path = os.path.join(script_dir,folder,lab_name+".csv")
        labs = pd.read_csv(full_file_path,usecols=['patientunitstayid',
                                                   'labresultoffset','labresult'])
        #Filter out labs from patients I don't care about.
        labs = labs[labs['patientunitstayid'].isin(ids['patientunitstayid'])]
        
        #Make sure all the data's in order by patientstayid and offset.
        labs.sort_values(['patientunitstayid','labresultoffset'],inplace=True)
        
        #%% Get features.
        #Get the last lab for each patientstayid. 
        last_labs = labs.groupby('patientunitstayid').last().reset_index(drop=False)
        last_labs.rename(columns={'labresult':'last_'+lab_name},inplace=True)
        ids = ids.merge(last_labs,how='left',on='patientunitstayid')
        
        #Get mean, min, max, count.
        mean_labs = labs.groupby('patientunitstayid').mean().reset_index(drop=False)
        mean_labs.rename(columns={'labresult':'mean_'+lab_name},inplace=True)
        ids = ids.merge(mean_labs,how='left',on='patientunitstayid')
        
        min_labs = labs.groupby('patientunitstayid').min().reset_index(drop=False)
        min_labs.rename(columns={'labresult':'min_'+lab_name},inplace=True)
        ids = ids.merge(min_labs,how='left',on='patientunitstayid')
        
        max_labs = labs.groupby('patientunitstayid').max().reset_index(drop=False)
        max_labs.rename(columns={'labresult':'max_'+lab_name},inplace=True)
        ids = ids.merge(max_labs,how='left',on='patientunitstayid')
        
        count_labs = labs.groupby('patientunitstayid').count().reset_index(drop=False)
        count_labs.rename(columns={'labresult':'count_'+lab_name},inplace=True)
        ids = ids.merge(count_labs,how='left',on='patientunitstayid')
        
        #Get difference of last and second to last lab. 
        labs['diff'] = labs.groupby('patientunitstayid').diff()['labresult']
        diff_labs = labs.groupby('patientunitstayid').last().reset_index(drop=False)
        diff_labs.rename(columns={'diff':'diff_'+lab_name},inplace=True)
        ids = ids.merge(diff_labs,how='left',on='patientunitstayid')
        
        #Get normal value for this lab.
        norm = norms.get(lab_name)
        
        #Get last distance from normal.
        ids['dist_' + lab_name] = ids['last_' + lab_name] - norm
        
        #Also square it.
        ids['dist2_' + lab_name] = ids['dist_' + lab_name]**2
        
        #Get reversion factor (diff)(normal - last lab) 
        ids['reversion_' + lab_name] = (ids['diff_' + lab_name] * -1 * 
                                        ids['dist_' + lab_name])
        
        #Only keep the columns I care about for the model.
        ids = ids[['patientunitstayid','last_'+lab_name,'mean_'+lab_name,
                   'min_'+lab_name,'max_'+lab_name,'count_'+lab_name,
                   'diff_'+lab_name,'dist_' + lab_name,'dist2_' + lab_name,
                   'reversion_' + lab_name]]
        
        comp = comp.merge(ids,on='patientunitstayid',how='inner')
    
    #%% Save off results.
    comp.to_csv('lab_feature_data.csv',index=False)
    calc_time = time() - start
    
if __name__ == '__main__':
    full_script()