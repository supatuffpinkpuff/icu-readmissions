# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:25:04 2020

Pulls the labs that occurred during the stay only. Sorts them by patientstayid
and offset. Called by AllLabsDuringStay.py

Runtime: 30 seconds a loop.

@author: Kirby
"""

def labs_before_delirium(lab_name):
    #%% Package setup. 
    import numpy as np
    import pandas as pd
    from time import time
    from pathlib import Path
    
    start = time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath("Cohort")
    eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')
    
    #%%Pulls list of Stay IDs.
    comp = pd.read_csv(cohort_path.joinpath("ICU_readmissions_dataset.csv"))
    all_ids = comp['patientunitstayid'
                   ].append(comp['original_unitstayid']
                        ).append(comp['2nd_orig_unitstayid']
                                 ).append(comp['3rd_orig_unitstayid']
                                          ).append(comp['4th_orig_unitstayid'])
    all_ids = all_ids.dropna().drop_duplicates()                                      
    
    #Get dict of LOS for each stay.
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"), 
                  usecols=['patientunitstayid', 'unitdischargeoffset'])
    # Pull stay LOS for each stay. 
    los_dict = pat.set_index('patientunitstayid').to_dict(
        ).get('unitdischargeoffset')

    #Pulls all lab info and drop columns we don't need.
    lab = pd.read_csv(eicu_path.joinpath("lab.csv"),
                      usecols=['patientunitstayid','labresultoffset',
                               'labname','labresult'])
    
    #Only keeps the lab we want.
    lab = lab[lab['labname']==lab_name]
    #Only keeps the patient stays we want.
    lab = lab[lab['patientunitstayid'].isin(all_ids)]
    #Adds delirium start time info to each row.
    lab['LOS'] = lab['patientunitstayid'].apply(los_dict.get)
    #If they occurred before the ICU stay remove them.
    lab = lab[lab['labresultoffset'] >= 0]
    #Only keep labs that happened during the ICU stay. 
    lab = lab[lab['labresultoffset'] < lab['LOS']]
    lab.sort_values(by=['patientunitstayid','labresultoffset'],inplace=True)
    
    calc_time = time() - start
    return lab

if __name__ == '__main__':
    #Change out this name for different labs. 
    #String for lab_name must exactly match the labname used in the Lab table of eICU.
    from pathlib import Path
    lab_name = 'BUN'
    file_path = Path(__file__)
    test = labs_before_delirium(lab_name)
    test.to_csv(file_path.parent.joinpath("AllLabsDuringStay",lab_name + ".csv"),index=False)
