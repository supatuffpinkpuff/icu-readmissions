# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 12:53:00 2020

This code is meant to pull data on whether patients had infections diagnosed
in their eICU data, based on diagnosis information, looking both at diagnosis
strings and the ICD9 codes if listed. This code only counts it if the diagnosis
was added in or before the observation window provided in the file. 

Then combines that with SOFA scores calculated separately to determine if 
patients had sepsis or septic shock.

Run time: ~15 sec

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
    pat_stays = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    
    # Attach LOS as end, make admission start of window. 
    pat_stays = pat_stays.merge(pat,on='patientunitstayid',how='left')
    pat_stays.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    pat_stays['start'] = 0
    
    #Just get diagnosis data. 
    infect_data = pd.read_csv(eicu_path.joinpath("diagnosis.csv"),
                              usecols=['patientunitstayid','diagnosisoffset',
                                       'diagnosisstring','icd9code'])
    
    #Get SOFA data.
    sofa = pd.read_csv('suspected_sepsis.csv')
    
    #Load lists of icd9 codes to look for. 
    rounded_codes = pd.read_csv('ICD9_codes_rounded.csv',header=None)
    rounded_codes = rounded_codes.values.astype(str).tolist()[0]
    exact_codes = pd.read_csv('ICD9_codes_exact.csv',header=None)
    exact_codes = exact_codes.values.astype(str).tolist()[0]
    
    
    #%% Process the data.
    #Just get data on patients we care about. 
    infect_data = infect_data[
        infect_data['patientunitstayid'].isin(
            pat_stays['patientunitstayid'])]
    
    #Drop data after the observation window for each patient. 
    lookup = pat_stays.set_index('patientunitstayid')
    def keep_row(current_ID,offset):
        #Get window time stamps.
        window_start = lookup.loc[current_ID,'start']
        window_end = lookup.loc[current_ID,'end']
        #If the infection took place before/in window, keep it. 
        if (offset <= window_end):
            return 1
        else:
            return 0
    
    infect_data['keep'] = infect_data.apply(
        lambda row: keep_row(
            row['patientunitstayid'],row['diagnosisoffset']),axis=1)
    infect_data = infect_data[infect_data['keep']==1]
    
    #Make it all lowercase.
    infect_data = infect_data.applymap(
        lambda s:s.lower() if type(s) == str else s)
    
    #This function takes in the icd9code as a lower case string, harvests the 
    #first part if present, removes any entries with letters, and converts it to a float. 
    #and returns nan if there isn't any value. 
    def shorten_icd9(icd9):
        if type(icd9) == float:
            return np.nan
        else:
            #Get the first part separated by commas
            icd9 = icd9.split(',')[0]
            #Check if it's got letters in it. If so, get rid of it. 
            if icd9.upper() != icd9:
                    return np.nan
            else: 
                return float(icd9)
    
    infect_data['icd9'] = infect_data.apply(lambda row: 
                                            shorten_icd9(row['icd9code']),
                                            axis=1)
    
    #%% Find whether patients had infections or not in time period.
    
    #Keep the rows that have ICD9 codes related to infections.
    rounded_code_stays = infect_data[np.floor(infect_data['icd9']
                                              ).isin(rounded_codes)]
    exact_code_stays = infect_data[infect_data['icd9'].isin(exact_codes)]
    
    #Keep the rows where the diagnosis string contains these words
    search_terms_list = ['infection','infectious']
    string_stays = infect_data[infect_data['diagnosisstring'].str.contains(
        '|'.join(search_terms_list),na=False)]
    #Drop the rows where diagnosis string contains "non-infectious"
    string_stays = string_stays[np.logical_not(
        string_stays['diagnosisstring'].str.contains('non-infectious'))]
    
    #Combine all the stays.
    all_stays = pd.concat([rounded_code_stays,exact_code_stays,
                           string_stays])
    all_stays.drop_duplicates(inplace=True)
    all_stays.sort_values(['patientunitstayid','diagnosisoffset'],
                          inplace=True)
    
    #Just get the stay IDs
    all_stays = all_stays[['patientunitstayid']]
    all_stays.drop_duplicates(inplace=True)
    
    #Create a column for the infection feature
    pat_stays['infection'] = pat_stays['patientunitstayid'].isin(
        all_stays['patientunitstayid'])
    
    #%% Find if they had sepsis.
    pat_stays = pat_stays.merge(sofa,how='left',on=['patientunitstayid'])
    #Check infection column for True, and suspected sepsis column for 1 to get
    #sepsis feature. Can also be used with septic shock.
    def get_sepsis(infection,suspected_sepsis):
        if (infection == True) & (suspected_sepsis == 1): 
            return 1
        else:
            return 0
        
    pat_stays['sepsis'] = pat_stays.apply(
        lambda row: get_sepsis(
            row['infection'],row['suspected_sepsis']), axis=1)
    pat_stays['septic_shock'] = pat_stays.apply(
        lambda row: get_sepsis(
        row['infection'],row['suspected_septic_shock']), 
        axis=1)
    
    pat_stays = pat_stays[['patientunitstayid','infection',
                           'suspected_sepsis','sepsis',
                           'suspected_septic_shock','septic_shock']]
    
    pat_stays['infection'] = pat_stays['infection'].astype(int)
    pat_stays['suspected_sepsis'] = pat_stays['suspected_sepsis'].astype(int)
    pat_stays['suspected_septic_shock'] = pat_stays['suspected_septic_shock'].astype(int)
    
    
    #Save off results.
    pat_stays.to_csv('sepsis_and_infection.csv', index=False)
    
    #For performance testing. 
    calculation_timer = time.time()-start_timer

if __name__ == '__main__':
    full_script()