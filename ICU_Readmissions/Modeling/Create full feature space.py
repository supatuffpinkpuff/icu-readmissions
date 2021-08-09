# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:14:34 2021

Combines all our current features into two big ol' csvs, one with all 
numeric data, one with all categorical data. Also generates a list of all
column names, of all numerical columns then all categorical columns.

runtime: a few seconds.

@author: Kirby
"""
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from datetime import datetime

start = time()
now = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

file_path = Path(__file__)
feature_path = file_path.parent.parent.joinpath('Features')
cohort_path = file_path.parent.parent.joinpath('Cohort')

#%% Load in data. 

ids = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
#Numeric data.
static_num = pd.read_csv(feature_path.joinpath('Static','static_features.csv'), 
                     usecols=range(4,20))
labs = pd.read_csv(feature_path.joinpath('Labs','lab_feature_data.csv'), 
                     usecols=range(4,355))
gcs = pd.read_csv(feature_path.joinpath('NurseCharting','GCS_feature.csv'), 
                     usecols=range(1,5))
rass = pd.read_csv(feature_path.joinpath('NurseCharting','rass_feature.csv'), 
                     usecols=range(1,1))
temp = pd.read_csv(feature_path.joinpath('NurseCharting','temp_feature.csv'), 
                     usecols=range(1,1))
urine = pd.read_csv(feature_path.joinpath('IntakeOutput',
                                          'urine_transfusions_features.csv'),
                    usecols=['last_24hr_urine'])
vent = pd.read_csv(feature_path.joinpath('Ventilation',
                                          'MV_duration.csv'),
                   usecols=[1])

#Categorical/binary data.
static_cat = pd.read_csv(feature_path.joinpath('Static','static_features.csv'), 
                     usecols=range(20,76))
meds = pd.read_csv(feature_path.joinpath('Medications','AllDrugFeatures.csv'), 
                     usecols=range(4,55))
hist = pd.read_csv(feature_path.joinpath('History','HistoryFeatures.csv'), 
                     usecols=range(4,58))

transf = pd.read_csv(feature_path.joinpath('IntakeOutput',
                                          'urine_transfusions_features.csv'),
                    usecols=range(7,10))
dial = pd.read_csv(feature_path.joinpath('Dialysis','dialysis_feature.csv'),
                   usecols=['dialysis'])
elix = pd.read_csv(feature_path.joinpath('Comorbidity',
                                         'Elixhauser_features.csv'),
                   usecols=range(1,32))
seps = pd.read_csv(feature_path.joinpath('Sepsis','sepsis_and_infection.csv'),
                   usecols=range(1,6))


#%% Put it all together
num = pd.concat([static_num,labs,gcs,rass,temp,urine],axis=1)
cat = pd.concat([static_cat,meds,hist,vent,transf,dial,elix],axis=1)
num.to_csv('numeric_data.csv',index=False)
cat.to_csv('categorical_data.csv',index=False)

#Column names.
cols1 = num.columns.to_frame(index=False)
cols2 = cat.columns.to_frame(index=False)
all_cols = pd.concat([cols1,cols2],axis=0)
all_cols.to_csv('column_names.csv',index=False)

#final.to_csv('All_Features.csv',index=False)