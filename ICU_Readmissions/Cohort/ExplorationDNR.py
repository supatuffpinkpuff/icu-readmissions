# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 19:57:28 2020

#This pulls the last GCS value for each patient at the end of the observation
 window.

Run time: 5 minutes

@author: Kirby
"""
#%% Import packages.
import numpy as np
import pandas as pd
import os
#import multiprocessing as mp
from time import time
from pathlib import Path

start = time()

file_path = Path(__file__)
cohort_path = file_path.parent.parent.joinpath('Cohort')
eicu_path = file_path.parent.parent.parent.joinpath('eicu')

eol = pd.read_csv(eicu_path.joinpath('careplaneol.csv'))

cpg = pd.read_csv(eicu_path.joinpath('careplangeneral.csv'))

pat = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))

#%% Load in and prepare relevant data.
groups = cpg['cplgroup'].drop_duplicates()
vals = cpg[cpg['cplgroup']=='Care Limitation']['cplitemvalue'].drop_duplicates()

#look at all groups/items.
all_vals = cpg[cpg['patientunitstayid'].isin(pat['patientunitstayid'])][['cplgroup','cplitemvalue']].value_counts()
all_vals.to_excel('careplangeneral_items.xls')
#Search for "Comfort measures only", "Do not resuscitate", "No CPR"?, 
#"No augmentation of care"? 

#%%Get counts of pat stays with each possible thing. 

cmo = cpg[cpg['cplitemvalue']=='Comfort measures only'] #
cmo = cmo['patientunitstayid'].drop_duplicates()
dnr = cpg[cpg['cplitemvalue']=='Do not resuscitate'] #
dnr = dnr['patientunitstayid'].drop_duplicates() 
no = cpg[cpg['cplitemvalue']=='No CPR'] #
no = no['patientunitstayid'].drop_duplicates()
naoc = cpg[cpg['cplitemvalue']=='No augmentation of care'] #
naoc = naoc['patientunitstayid'].drop_duplicates()
any_of = cpg[(cpg['cplitemvalue']=='Comfort measures only') | 
             (cpg['cplitemvalue']=='Do not resuscitate') |
             (cpg['cplitemvalue']=='No CPR') |
             (cpg['cplitemvalue']=='No augmentation of care')] #
any_of = any_of['patientunitstayid'].drop_duplicates()

# All eICU (200,859 stays):
# CMO: 4,418 (2.2%)
# DNR: 27,058 (13.5%)
# No CPR: 5,960 (3.0%)
# No Aug. Care: 1,468 (0.7%)
# Any of these: 28,192 (14%)


#%%Get counts of pat stays in our original criteria with each thing. 
our_stays = pat['patientunitstayid'].drop_duplicates()
cpg = cpg[cpg['patientunitstayid'].isin(our_stays)]
cmo = cpg[cpg['cplitemvalue']=='Comfort measures only']
cmo = cmo['patientunitstayid'].drop_duplicates()
dnr = cpg[cpg['cplitemvalue']=='Do not resuscitate'] 
dnr = dnr['patientunitstayid'].drop_duplicates() 
no = cpg[cpg['cplitemvalue']=='No CPR'] 
no = no['patientunitstayid'].drop_duplicates()
naoc = cpg[cpg['cplitemvalue']=='No augmentation of care'] 
naoc = naoc['patientunitstayid'].drop_duplicates()
any_of = cpg[(cpg['cplitemvalue']=='Comfort measures only') | 
             (cpg['cplitemvalue']=='Do not resuscitate') |
             (cpg['cplitemvalue']=='No CPR') |
             (cpg['cplitemvalue']=='No augmentation of care')] 
any_of = any_of['patientunitstayid'].drop_duplicates()


# Our Current Dataset (30,200 stays):
# CMO: 93 (0.3%)
# DNR: 1,001 (3.3%)
# No CPR: 202  (0.6%)
# No Aug. Care: 33 (0.1%)
# Any of these: 1,041 (3.4s%)


#For performance testing. 
calc = time()-start
