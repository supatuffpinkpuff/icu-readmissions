# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 15:09:19 2020

Converting old SQL code to Python.

Pull static features from eICU, that are used in the first 24 hour models. 
Will pull from the Patient, Hospital, and ApachePatientResult, 

runtime: 10 sec

@author: Kirby
"""
def full_script():
    
    #%% Packages
    import numpy as np
    import pandas as pd
    import time as time
    from datetime import datetime
    from pathlib import Path
    from time import time
    
    start = time()
    
    file_path = Path(__file__)
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
    eicu_path = file_path.parent.parent.parent.parent.joinpath('eicu')
    
    #%% Load in needed data.
    
    comp = pd.read_csv(cohort_path.joinpath("ICU_readmissions_dataset.csv"))
    today = datetime.now().replace(hour=0,minute=0,second=0,microsecond=0)
    pat = pd.read_csv(eicu_path.joinpath("patient.csv"),
                                         usecols=['patientunitstayid','age',
                                                  'gender','ethnicity',
                                                  'hospitalid',
                                                  'admissionheight',
                                                  'hospitaladmittime24',
                                                  'hospitaladmitoffset',
                                                  'hospitaladmitsource','unittype',
                                                  'unitadmittime24',
                                                  'unitadmitsource',
                                                  'unitvisitnumber',
                                                  'admissionweight',
                                                  'dischargeweight',
                                                  'unitdischargeoffset'],
                                         parse_dates=['hospitaladmittime24',
                                                      'unitadmittime24'])
    hosp = pd.read_csv(eicu_path.joinpath("hospital.csv"))
    apache = pd.read_csv(eicu_path.joinpath("apachepatientresult.csv"),
                                            usecols=['patientunitstayid',
                                                     'apachescore',
                                                     'apacheversion'])
    
    #%% Get apache scores.
    apache = apache[apache['apacheversion']=='IV']
    comp = comp.merge(apache,how='left',on='patientunitstayid')
    comp.drop(columns='apacheversion',inplace=True)
    comp['apachescore'] = comp['apachescore'][comp['apachescore']!=-1]
    
    #%% Get patient info.
    comp = comp.merge(pat,how='left',on='patientunitstayid')
    
    #Convert age to numeric.
    def age_to_nums(age):
        if age == '> 89':
            return 90
        else:
            return age
    
    comp['age'] = comp['age'].apply(age_to_nums).astype(int)
    
    #Age exponents.
    comp['age^2'] = comp['age']**2
    comp['age^3'] = comp['age']**3
    comp['age^4'] = comp['age']**4
    #Convert times of days to minutes since midnight.
    comp['hospitaladmittime24'] = (comp['hospitaladmittime24']-today).dt.seconds/60
    comp['unitadmittime24'] = (comp['unitadmittime24']-today).dt.seconds/60
    #BMI
    comp['admissionheight'] = comp['admissionheight'][comp['admissionheight']>=50]
    comp['BMI'] = comp['admissionweight']/comp['admissionheight']
    #Weight change
    comp['weightchange'] = comp['dischargeweight'] - comp['admissionweight']
    
    #Hospital stay time.
    comp['hospitalstaytime'] = (comp['unitdischargeoffset'] - 
                                comp['hospitaladmitoffset'])
    
    #Time of unit discharge. 
    
    #%% Get hospital traits.
    comp = comp.merge(hosp,how='left',on='hospitalid')
    
    #%% Drop unneeded columns.
    comp.drop(columns=['hospitalid'],inplace=True)
    
    #%% One hot encode.
    comp = pd.get_dummies(comp)
    
    #%% Save off results.
    comp.to_csv('static_features.csv',index=False)
    
    calc_time = time() - start
    
if __name__ == '__main__':
    full_script()