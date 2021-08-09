# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 01:34:06 2021

This code pulls mechanical ventilation duration during the ICU stay by finding 
start times from NurseCharting, RespiratoryCharting, or Treatment, and 
oxygen therapy or end of the observation window as stop times, again from the same 
locations. 

Based off of "The Search for Optimal Oxygen Saturation Targets in Critically
 Ill Patients: Observational Data from Large ICU Databases" 
 (https://doi.org/10.1016/j.chest.2019.09.015).

Run time: 4 minutes

@author: Kirby
"""
def full_script():
    #%% Package setup
    import pandas as pd
    import numpy as np
    from time import time
    import datetime
    from pathlib import Path
    import matplotlib.pyplot as plt
    
    start_timer = time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
    eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')
    
    #%% Load in data.
    
    #Set up list of counts of patients that had no MV duration.
    no_mv_counts = list()
    
    #Pulls list of Stay IDs and offsets we care about.
    pat_stays = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #Get patient info table for LOS, serve as end of window. 
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                      usecols=['patientunitstayid', 'unitdischargeoffset'])
    pat_stays = pat_stays.merge(pat,on='patientunitstayid',how='left')
    pat_stays.rename(columns={'unitdischargeoffset':'end'},inplace=True)
    pat_stays['start'] = 0
    
    # #Get physicalexam data.
    # phys = pd.read_csv(eicu_path.joinpath("physicalexam.csv"),
    #                    usecols=['patientunitstayid','physicalexamoffset',
    #                             'physicalexamtext'])
    #Get treatment
    treat = pd.read_csv(eicu_path.joinpath("treatment.csv"),
                        usecols=['patientunitstayid','treatmentoffset',
                                'treatmentstring'])
    
    #Get respiratorycharting
    resp = pd.read_csv(eicu_path.joinpath("respiratoryCharting.csv"),
                       usecols=['patientunitstayid','respchartoffset',
                                'respchartvaluelabel','respchartvalue'])
    
    #Get nursecharting
    nurse = pd.read_csv(eicu_path.joinpath("nurseCharting.csv"),
                       usecols=['patientunitstayid','nursingchartoffset',
                                'nursingchartcelltypevallabel',
                                'nursingchartvalue'])
    
    #%% Pre-processing
    #Only keep the stays we care about and add obs window times. 
    # phys = phys.merge(pat_stays,on='patientunitstayid',how='right')
    treat = treat.merge(pat_stays,on='patientunitstayid',how='right')
    resp = resp.merge(pat_stays,on='patientunitstayid',how='right')
    nurse = nurse.merge(pat_stays,on='patientunitstayid',how='right')
    
    #Drop data after the observation window for each patient stay. 
    # phys = phys[phys['physicalexamoffset']<=phys['end']]
    treat = treat[treat['treatmentoffset']<=treat['end']]
    resp = resp[resp['respchartoffset']<=resp['end']]
    nurse = nurse[nurse['nursingchartoffset']<=nurse['end']]
    
    #Drop data before the ICU stay. 
    # phys = phys[phys['physicalexamoffset']>=0]
    treat = treat[treat['treatmentoffset'] >= 0]
    resp = resp[resp['respchartoffset'] >= 0]
    nurse = nurse[nurse['nursingchartoffset'] >= 0]
    
    #Combine resp and nurse data. 
    comb = nurse.rename(columns={'nursingchartvalue':'string',
                                 'nursingchartoffset':'offset'})
    resp_label = resp.rename(columns={'respchartoffset':'offset',
                                      'respchartvaluelabel':'string'})
    resp_value = resp[resp['respchartvaluelabel'].str.contains(
        'o2 device|respiratory device|ventilator type|oxygen delivery method',
        case=False)]
    resp_value.rename(columns={'respchartoffset':'offset',
                               'respchartvalue':'string'},inplace=True)
    comb = pd.concat([comb,resp_label,resp_value])
    
    #%%Find mechanical ventilation starts/settings, indicating it was going.
    
    #Find rows containing "vent" but not "Wake up assessment"
    comb_mv = comb[comb['string'].str.contains('vent',case=False,na=False)]
    comb_mv = comb_mv[~comb_mv['string'].str.contains(
        'Wake up assessment',case=False,na=False)]
    #Combine with other searched rows.
    vent_words = pd.read_csv('ventilation_search_strings.csv')
    vent_words = list(vent_words['string'])
    temp = comb[comb['string'].str.contains(
        '|'.join(vent_words),case=False,na=False)]
    
    #Add in the MV info from treatment and physical exam. 
    # phys_mv = phys[phys['physicalexamtext']=='ventilated'].copy()
    treat_mv = treat[treat['treatmentstring'].str.contains(
        'mechanical ventilation',case=False)].copy()
    # phys_mv.rename(columns={'physicalexamoffset':'offset'},inplace=True)
    treat_mv.rename(columns={'treatmentoffset':'offset'},inplace=True)
    #Put it all together.
    comb_mv = pd.concat([comb_mv,temp,treat_mv]).drop_duplicates()
    
    #Get times when MV was going. 
    comb_mv = comb_mv[['patientunitstayid', 'offset']]
    comb_mv['MV'] = 1
    
    #%% Find MV stops via O2 therapy/end of ICU stay. 
    
    #Find oxygen therapy, also indicating stops. 
    #From respiratorycare or nurse charting.
    o2_words = pd.read_csv('o2therapy_search_strings.csv')
    o2_words = list(o2_words['string'])
    comb_o2 = comb[comb['string'].str.contains('|'.join(o2_words),
                                               case=False,na=False)]
    #From treatment.
    treat_o2 = treat[treat['treatmentstring'].str.contains(
        'oxygen therapy|non-invasive ventilation',case=False)].copy()
    treat_o2.rename(columns={'treatmentoffset':'offset'},inplace=True)
    #From physicalexam.
    #No relevant data found. 
    
    #nurseCharting has some o2 therapy documented as O2 L/%. Find those.
    nurse_o2 = nurse[
        nurse['nursingchartcelltypevallabel'] == 'O2 L/%'].copy()
    nurse_o2['nursingchartvalue'] = \
        nurse_o2['nursingchartvalue'].astype(float)
    nurse_o2 = nurse_o2[(nurse_o2['nursingchartvalue'] > 0) &
                        (nurse_o2['nursingchartvalue'] <= 100)]
    nurse_o2.rename(columns={'nursingchartoffset':'offset'},inplace=True)
    
    #respiratoryCare also has some FiO2 documented we can use. 
    #is this something we should use? Per Dr. Stevens, could be NIV or MV. 
    
    #Add a stop at the end time of the observation window. 
    ends = pat_stays[['patientunitstayid','end']].copy()
    ends.rename(columns={'end':'offset'},inplace=True)
    
    #Put it all together.
    comb_o2 = pd.concat([comb_o2,treat_o2,nurse_o2,ends]).drop_duplicates()
    
    #Get times when MV was stopped. 
    comb_o2 = comb_o2[['patientunitstayid', 'offset']]
    comb_o2['MV'] = 0
    
    
    #%% Get durations.
    #Combine all the info together.
    all_info = pd.concat([comb_mv,comb_o2]).drop_duplicates()
    #Sort it by stay id, time.
    all_info.sort_values(['patientunitstayid','offset'],inplace=True)
    #Remove rows where MV status doesn't change.
    #Get next row's stay ID and MV status.
    all_info['last_MV'] = all_info['MV'].shift(periods=1)
    all_info['last_stay'] = all_info['patientunitstayid'].shift(periods=1)
    #Check if next row is same ICU stay and MV status. 
    all_info['last_MV_same?'] = all_info['MV'] == all_info['last_MV']
    all_info['last_stay_same?'] = \
        all_info['patientunitstayid'] == all_info['last_stay']        
    all_info = all_info[(all_info['last_MV_same?'] == False) | 
                        (all_info['last_stay_same?'] == False)]
    #Calculate durations of instances of MV. 
    all_info['last_offset'] = all_info['offset'].shift(periods=1)
    def calc_dur(curr_mv,last_mv,curr_offset,last_offset,last_stay_same):
        if (curr_mv == 0) & (last_mv == 1) & (last_stay_same == True):
            return curr_offset-last_offset
        else:
            return 0
    all_info['mv_duration'] = all_info.apply(
        lambda row: calc_dur(row['MV'],row['last_MV'],row['offset'],
                             row['last_offset'],row['last_stay_same?']),
        axis=1)
    #Sum it up per patient stay. 
    feat = all_info.groupby('patientunitstayid').sum()['mv_duration']
    feat = feat.reset_index()
    
    #Save off results. 
    feat.to_csv('MV_duration.csv',index=False)
    
    #%%Generate histograms and count patients w/out MV.
    plt.figure()
    plt.title('All MV Duration (hours)')
    (feat['mv_duration']/60).hist()
    plt.figure()
    plt.title('All MV Duration (hours, excluding 0 hrs)')
    ((feat[feat['mv_duration'] > 0])['mv_duration']/60).hist()
    plt.figure()
    plt.title('All MV Duration (hours, more than 0, less than 24h)')
    ((feat[(feat['mv_duration'] > 0) & 
           (feat['mv_duration'] <= 1440)])['mv_duration']/60).hist()
    no_mv_counts.append(feat[feat['mv_duration'] == 0]
                        ['mv_duration'].count()) #~30% have MV durations.
    #For performance testing. 
    calc_timer = time()-start_timer
    
if __name__ == '__main__':
    full_script()