# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 16:30:34 2021

Pulls ICD9 codes of diagnoses that were entered before discharge, and 
calculates Elixhauser comorbidity indices.

Used ICD mappings to Elixhauser from :
Quan H, Sundararajan V, Halfon P, Fong A, Burnand B, Luthi JC, Saunders LD, 
Beck CA, Feasby TE, Ghali WA. Coding algorithms for defining comorbidities 
in ICD-9-CM and ICD-10 administrative data. Med Care. 2005 Nov;43(11):1130-9. 
doi: 10.1097/01.mlr.0000182534.19832.83. PMID: 16224307.

Runtime: 45 seconds.

@author: Kirby
"""
def full_script():
    #%% Import packages.
    import numpy as np
    import pandas as pd
    #import multiprocessing as mp
    from time import time
    from pathlib import Path
    import icd
    import csv
    
    start = time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
    eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')
    
    #%% Load in and prepare relevant data.
    
    ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    
    # Attach LOS as end, make admission start of window. 
    ids = ids.merge(pat,on='patientunitstayid',how='left')
    ids.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    ids['start'] = 0
    
    #Get diagnosis data. 
    diag = pd.read_csv(eicu_path.joinpath("diagnosis.csv"),
                       usecols=['diagnosisid','patientunitstayid', 
                                'diagnosisoffset', 'icd9code',
                                'activeupondischarge'])
    
    #%%Filter diagnosis data.
    
    #Just our patients.
    cutoffs = ids[['patientunitstayid','end']]
    diag = diag.merge(cutoffs,on='patientunitstayid',how='right')
    
    # #Just diagnoses that were entered before disharge.
    # diag = diag[diag['activeupondischarge']==True]
    
    #Just diagnoses entered before ICU discharge. 
    diag = diag[diag['diagnosisoffset']<=diag['end']]
    
    #Convert nans to empty strings.
    diag = diag.fillna('')
    
    diag = diag[['diagnosisid','patientunitstayid','icd9code']].reset_index(
        drop=True)
    
    #%% Get Elixhauser comorbidity indices.
    #Convert to short format. 
    diag = icd.long_to_short_transformation(diag,"patientunitstayid",['icd9code'])
    
    #Get Elixhauser mappings. 
    with open('Elixhauser_mappings_ICD9.csv', newline='') as f:
        reader = csv.reader(f)
        icd_lists = list(reader)
    #Make it into a dict.
    mapping = dict()
    icd_lists.pop(0)
    for icd_list in icd_lists:
        while('' in icd_list): 
            icd_list.remove('')
        mapping.update({'elix_'+icd_list.pop(0):icd_list})
    
    icd_cols = []
    for i in range(0,36):
        icd_cols.append('icd_' + str(i))
    comorb = icd.icd_to_comorbidities(diag, "patientunitstayid", icd_cols, 
                                      mapping=mapping)
    
    comorb.drop(columns='patientunitstayid',inplace=True)
    
    #Convert it all to 0s and 1s. 
    for col in comorb.columns:
        comorb[col] = comorb[col].astype(int)
    
    comorb.to_csv('Elixhauser_features.csv')
    
    #Get prevalences. 
    prev = pd.DataFrame(columns=mapping.keys())
    for col in comorb.columns:
        prev[col] = comorb[col].value_counts()
    
    calc = time() - start
    
if __name__ == '__main__':
    full_script()