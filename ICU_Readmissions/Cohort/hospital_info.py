# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 21:25:18 2021

Generates table of hospital information. 

@author: Kirby
"""

#%% Package setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import mannwhitneyu
from scipy.stats import chisquare
from time import time

start = time()

file_path = Path(__file__)
eicu_path = file_path.parent.parent.parent.joinpath('eicu')
hist_path = file_path.parent.parent.joinpath('Features','History')
comorb_path = file_path.parent.parent.joinpath('Features','Comorbidity')

#%% Import data. 
comp = pd.read_csv('ICU_readmissions_dataset.csv')

pat = pd.read_csv(eicu_path.joinpath('patient.csv'),
                  usecols=['patientunitstayid','hospitalid'])

hosp = pd.read_csv(eicu_path.joinpath('hospital.csv'))

#%%%

pat = pat[pat['patientunitstayid'].isin(comp['patientunitstayid'])]
hosp = hosp[hosp['hospitalid'].isin(pat['hospitalid'])]

size = hosp['numbedscategory'].value_counts().reset_index()
teach = hosp['teachingstatus'].value_counts().reset_index()
region = hosp['region'].value_counts().reset_index()

size['prop'] = size['numbedscategory']/(size['numbedscategory'].sum())
teach['prop'] = teach['teachingstatus']/(teach['teachingstatus'].sum())
region['prop'] = region['region']/(region['region'].sum())



    

