# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 16:28:04 2020

This code uses the inclusion/exclusion criteria for the ICU readmissions 
prediction model's dataset. 

Changes: Now also excludes patients that don't have at least 50% of their LOS
covered by all teh PTS signals'

Run time: ~10 seconds.

@author: Kirby
"""

#%% Package Setup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from time import time
from pathlib import Path

start = time()

file_path = Path(__file__)
pts_path = file_path.parent.parent.joinpath('Features','PTS')
eicu_path = file_path.parent.parent.parent.joinpath('eicu')

#%% Import relevant tables and make everything lowercase.
adm_dx = pd.read_csv(eicu_path.joinpath("admissionDx.csv"),
                     usecols=['patientunitstayid', 'admitdxenteredoffset',
                              'admitdxpath','admitdxname'])
apache = pd.read_csv(eicu_path.joinpath("apachepredvar.csv"),
                     usecols=['patientunitstayid', 'admitdiagnosis', 
                              'electivesurgery', 'admitsource'])
treat = pd.read_csv(eicu_path.joinpath("treatment.csv"),
                    usecols=['patientunitstayid', 'treatmentoffset',
                             'treatmentstring'])
pat = pd.read_csv(eicu_path.joinpath("patient.csv"), 
                  usecols=['patientunitstayid', 'patienthealthsystemstayid',
                           'hospitaladmitoffset','hospitaladmitsource',
                           'hospitaldischargestatus',
                           'hospitaldischargeoffset','unitadmitsource',
                           'unitvisitnumber','unitstaytype',
                           'unitdischargeoffset', 'unitdischargelocation',
                           'unittype','uniquepid','wardid','hospitalid'])
hosp = pd.read_csv(eicu_path.joinpath("hospital.csv"), 
                   usecols=['hospitalid','numbedscategory'])
cpg = pd.read_csv(eicu_path.joinpath('careplangeneral.csv'),
                  usecols=['patientunitstayid','cplitemvalue'])

stays_prop = pd.read_csv(pts_path.joinpath('PTS_proportion_covered_whole_stay.csv'))
stays_prop24 = pd.read_csv(pts_path.joinpath('PTS_proportion_covered_24h.csv'))

# vitalp = pd.read_csv(eicu_path.joinpath("vitalperiodic.csv"), 
#                    usecols=['patientunitstayid','sao2','heartrate',
#                             'respiration','systemicsystolic',
#                             'systemicdiastolic','systemicmean'])
# vitala = pd.read_csv(eicu_path.joinpath("vitalaperiodic.csv"), 
#                    usecols=['patientunitstayid','noninvasivesystolic',
#                             'noninvasivediastolic','noninvasivemean'])

#%% Exploring Patient for sequence mistakes in unit visit numbers.
#Maximum number of unit visits is 18.
all_eicu_unitvisitnums = pat[['patientunitstayid','patienthealthsystemstayid',
                              'unitvisitnumber']]

#Storing results of check here.
all_eicu_hosp_stays = pat[['patienthealthsystemstayid']].drop_duplicates() #166,355 unique hospital stays.
all_eicu_hosp_stays.set_index('patienthealthsystemstayid',inplace=True)
all_eicu_hosp_stays['missing_visit_nums?'] = False

#Checks if unit visit numbers of hospital stays are consecutive. 

#Create dictionary of lists of unit visit numbers, per patienthealthsystemid. 
visit_num_dict = dict()
def make_visit_num_dict(health_system_id,visit_num):
    temp_list = visit_num_dict.get(health_system_id,[])
    temp_list.append(visit_num)
    visit_num_dict[health_system_id] = temp_list
    
all_eicu_unitvisitnums.apply(lambda row: make_visit_num_dict(
    row['patienthealthsystemstayid'],row['unitvisitnumber']),axis=1)

#Then create a function that looks through the lists, that sorts them, and 
#checks if the first item is 1, and all entries are consecutive.
for health_system_id, visit_nums_list in visit_num_dict.items():
    visit_nums_list.sort()
    if visit_nums_list[0] != 1:
        all_eicu_hosp_stays.loc[health_system_id,'missing_visit_nums?'] = True
        continue
    diff = np.diff(visit_nums_list)
    diff_set = set(diff)
    if (diff_set == set()) | (diff_set == {1}) :
        continue
    else: 
        all_eicu_hosp_stays.loc[health_system_id,'missing_visit_nums?'] = True
 
all_eicu_hosp_stays.reset_index(inplace=True)
hosp_stays_to_exclude = all_eicu_hosp_stays[
    all_eicu_hosp_stays['missing_visit_nums?']==True]

#%% Identify surgical patients in AdmissionDx/ApachePredVar

all_dxs = adm_dx[['admitdxpath']].drop_duplicates().sort_values(['admitdxpath'])
#Keyword searching through all paths. 
operative = all_dxs[all_dxs['admitdxpath'].str.contains("\\|Operative")]
oper_room = all_dxs[all_dxs['admitdxpath'].str.contains("O\.R\.")]
elective = all_dxs[all_dxs['admitdxpath'].str.contains("Elective")]

#Find all patients that had operative dxs.
op_dx_pats = adm_dx[adm_dx['admitdxpath'].isin(operative['admitdxpath'])]
op_dx_pats = op_dx_pats[['patientunitstayid']].drop_duplicates() #32,243 patients.

#Get all rows that had operation diagnoses of some sort.
op_dx_pats_info = adm_dx[adm_dx['patientunitstayid'].isin(
    op_dx_pats['patientunitstayid'])]

#Only keep patients that had S- prefixes in admit diagnosis. 
s_pats_info = apache[apache['admitdiagnosis'].str.contains('S-',na=False)]
s_pats = s_pats_info[['patientunitstayid']]

admdx_apache = op_dx_pats.merge(s_pats,on='patientunitstayid',how='outer')
admdx_apache_info = op_dx_pats_info.merge(
    s_pats_info,on='patientunitstayid',how='outer')
admdx_apache_info = admdx_apache_info.merge(
    pat,on='patientunitstayid',how='left')

#Get how many unit stays, hospital stays, and unique patients there are after removing bad unitvisitnumber stays.
surg_cleaned_unit_stays = admdx_apache_info[
    ~admdx_apache_info['patienthealthsystemstayid'].isin(
        hosp_stays_to_exclude['patienthealthsystemstayid'])][
            'patientunitstayid'].drop_duplicates()
surg_cleaned_hosp_stays = admdx_apache_info[
    ~admdx_apache_info['patienthealthsystemstayid'].isin(
        hosp_stays_to_exclude['patienthealthsystemstayid'])][
            'patienthealthsystemstayid'].drop_duplicates()
surg_cleaned_unique_pats = admdx_apache_info[
    ~admdx_apache_info['patienthealthsystemstayid'].isin(
        hosp_stays_to_exclude['patienthealthsystemstayid'])][
            'uniquepid'].drop_duplicates()

#Remove the stays that have bad unit visit number data in the hospital stay.
admdx_apache = admdx_apache[admdx_apache['patientunitstayid'].isin(
    surg_cleaned_unit_stays)]
admdx_apache_info = admdx_apache_info[
    admdx_apache_info['patienthealthsystemstayid'].isin(
        surg_cleaned_hosp_stays)]

#%% Identify the index surgeries among these patient unit stays.
admdx_apache_pat = admdx_apache.merge(pat,on='patientunitstayid',how='left')

# Split out admissionDx/ApachePredVar folks by unitvisitnumber.
unitvisitnum1 = admdx_apache_pat[admdx_apache_pat['unitvisitnumber']==1]
unitvisitnum2 = admdx_apache_pat[admdx_apache_pat['unitvisitnumber']==2]
unitvisitnum3 = admdx_apache_pat[admdx_apache_pat['unitvisitnumber']==3]
unitvisitnum4 = admdx_apache_pat[admdx_apache_pat['unitvisitnumber']==4]
unitvisitnum5 = admdx_apache_pat[admdx_apache_pat['unitvisitnumber']>=5]

#All surgical paitents where unitvisitnumber = 1 must be having their first surgical stay.
#Therefore, keep all of unitvisitnum1.
first_surg_icu_stays = unitvisitnum1

# Find overlap of unitvisitnumber levels of the surgical patients. 
# If an indicator = both, then the right side unit visit isn't the first surgical stay.
overlap1_2 = first_surg_icu_stays[['patienthealthsystemstayid']].merge(
    unitvisitnum2[['patienthealthsystemstayid']],on='patienthealthsystemstayid',
    how='inner')
unitvisitnum2 = unitvisitnum2[~unitvisitnum2['patienthealthsystemstayid'].isin(
    overlap1_2['patienthealthsystemstayid'])]
first_surg_icu_stays = pd.concat([first_surg_icu_stays,unitvisitnum2])

overlap12_3 = first_surg_icu_stays[['patienthealthsystemstayid']].merge(
    unitvisitnum3[['patienthealthsystemstayid']],on='patienthealthsystemstayid',
    how='inner')
unitvisitnum3 = unitvisitnum3[~unitvisitnum3['patienthealthsystemstayid'].isin(
    overlap12_3['patienthealthsystemstayid'])]
first_surg_icu_stays = pd.concat([first_surg_icu_stays,unitvisitnum3])

overlap123_4 = first_surg_icu_stays[['patienthealthsystemstayid']].merge(
    unitvisitnum4[['patienthealthsystemstayid']],on='patienthealthsystemstayid',
    how='inner')
unitvisitnum4 = unitvisitnum4[~unitvisitnum4['patienthealthsystemstayid'].isin(
    overlap123_4['patienthealthsystemstayid'])]
first_surg_icu_stays = pd.concat([first_surg_icu_stays,unitvisitnum4])

#

#%% Remove stays that didn't have at least 12 hours. 
first_surg_icu_stays = first_surg_icu_stays[first_surg_icu_stays['unitdischargeoffset']>=120]
#32,386


#%% Split out the groups by different discharge locations of these patient unit stays.

#Defining discharge locations for each group.
readmission_locs = ['Floor','Telemetry','Acute Care/Floor',
                    'Step-Down Unit (SDU)'] 
sdu_locs = ['Step-Down Unit (SDU)']
non_readmission_locs = ['Home','Skilled Nursing Facility','Rehabilitation',
                        'Nursing Home']
exception_locs = ['Death',np.nan,'Other Hospital','Other External','Other',
                  'Other Internal','Operating Room']
defer_locs = ['Other ICU','ICU','Other ICU (CABG)']  

stays_discharge_locs = first_surg_icu_stays[['patientunitstayid',
                                             'patienthealthsystemstayid',
                                             'unitdischargelocation']]
pot_readmission_stays = stays_discharge_locs[
    stays_discharge_locs['unitdischargelocation'].isin(readmission_locs)] 
sdu_stays = stays_discharge_locs[
    stays_discharge_locs['unitdischargelocation'].isin(sdu_locs)] 
non_readmission_stays = stays_discharge_locs[
    stays_discharge_locs['unitdischargelocation'].isin(non_readmission_locs)] 
exception_stays = stays_discharge_locs[
    stays_discharge_locs['unitdischargelocation'].isin(exception_locs)] 
defer_stays = stays_discharge_locs[
    stays_discharge_locs['unitdischargelocation'].isin(defer_locs)] 
#Count how many were deaths.
#stays_discharge_locs[stays_discharge_locs['unitdischargelocation']=='Death']

#%% Get all potentially relevant ICU stays, with needed info. 
all_rel_stays = pat[['patientunitstayid','patienthealthsystemstayid',
                     'hospitaldischargestatus','hospitaldischargeoffset',
                     'unitdischargelocation','unitvisitnumber',
                     'hospitaladmitoffset','unitdischargeoffset','unittype',
                     'unitadmitsource','unitstaytype']]
all_rel_stays = all_rel_stays[all_rel_stays['patienthealthsystemstayid'].isin(
    first_surg_icu_stays['patienthealthsystemstayid'])]
all_rel_stays.sort_values(['patienthealthsystemstayid','unitvisitnumber'],
                          inplace=True)
all_rel_stays.reset_index(inplace=True,drop=True)

#Remove unit stays that were actually SDU, not ICU.
all_rel_stays = all_rel_stays[
    all_rel_stays['unitstaytype']!='stepdown/other']

#Find readmission discharge location stays that actually had readmissions. 
#Check if there's another ICU stay in that same health system stay (True for readmission)
all_rel_stays['next_patienthealthsystemstayid'] = \
    all_rel_stays[['patienthealthsystemstayid']].shift(periods=-1)
all_rel_stays['next_healthsystemstayid_same?'] = \
    all_rel_stays['patienthealthsystemstayid'] == \
        all_rel_stays['next_patienthealthsystemstayid']
#Find time after hospital admit that the this ICU admit occurred.
all_rel_stays['unit_admit_offset_from_hosp_admit'] = \
    0 - all_rel_stays['hospitaladmitoffset']
#Find time after hospital admit that ICU discharge occurred.
all_rel_stays['unit_disch_offset_from_hosp_admit'] = \
    all_rel_stays['unitdischargeoffset'] - all_rel_stays['hospitaladmitoffset']
#Find time after hospital admit that the next ICU admit occurred. 
all_rel_stays['next_admit_offset_from_hosp_admit'] = \
    all_rel_stays[['unit_admit_offset_from_hosp_admit']].shift(periods=-1)
#Check if the next unit visit's admission is within 48 hours of this unit stay's discharge. (True for readmissions)
all_rel_stays['time_from_this_disch_to_next_admit'] = \
    all_rel_stays['next_admit_offset_from_hosp_admit'] - \
        all_rel_stays['unit_disch_offset_from_hosp_admit']
all_rel_stays['next_unit_admit_in_48_hrs?'] = \
    all_rel_stays['time_from_this_disch_to_next_admit'] <= 4320 #2880 if you want to do 48 hours. 4320 for 72 hours.
#Identify stays that had an ICU readmission from our definition.
all_rel_stays['readmission?'] = \
    (all_rel_stays['next_healthsystemstayid_same?'] & \
         all_rel_stays['next_unit_admit_in_48_hrs?'])

#Remove some weird unit types/admit sources later on.
all_rel_stays['next_unittype'] = all_rel_stays[['unittype']].shift(periods=-1)
all_rel_stays['next_admitsource'] = all_rel_stays[['unitadmitsource']].shift(
    periods=-1)
all_rel_stays['next_staytype'] = all_rel_stays[['unitstaytype']].shift(
    periods=-1)

#Get time of discharge relative to ICU discharge. 
all_rel_stays['hosp_disch_from_unit_disch'] = \
    all_rel_stays['hospitaldischargeoffset'] - all_rel_stays['unitdischargeoffset']
#Get if patient died within 72 hours. 
all_rel_stays['death_after_disch'] = (
    (all_rel_stays['hosp_disch_from_unit_disch'] <= 4320) &
    (all_rel_stays['hospitaldischargestatus'] == 'Expired')
    )

#Get label (badly planned discharge, indicated by death or readmission.)
all_rel_stays['bad_disch_plan'] = (
    all_rel_stays['death_after_disch'] | all_rel_stays['readmission?']
    )

#%%Analyze groups of locations for readmissions, death, or either. 

#Analyze a group for bad discharge planning. 
def count_labels(stays):
    stays = stays.merge(all_rel_stays,on='patientunitstayid',how='inner')
    stays = stays[['patientunitstayid','readmission?','death_after_disch',
                   'bad_disch_plan']]
    counts = pd.DataFrame(columns=['readmission','death','bad_disch_plan'])
    #Count up readmissions and non-readmissions. 
    counts['readmission'] = \
        stays.groupby('readmission?').count()['patientunitstayid']
    counts['death'] = \
        stays.groupby('death_after_disch').count()['patientunitstayid']
    counts['bad_disch_plan'] = \
        stays.groupby('bad_disch_plan').count()['patientunitstayid']
    return stays, counts

pot_readmission_stays, pot_readmit_counts = count_labels(pot_readmission_stays)
non_readmission_stays,non_readmit_counts = count_labels(non_readmission_stays)
exception_stays, exception_counts = count_labels(exception_stays)

#%% Combine the relevant groups together. 
final_dataset = pd.concat([pot_readmission_stays,non_readmission_stays])

#%% Handle CMO/DNR and related patients.
#Remove CMO or augmentation of care folks, we expect them to die.
cmo = cpg[(cpg['cplitemvalue']=='Comfort measures only') |
          (cpg['cplitemvalue']=='No augmentation of care')]
cmo = cmo['patientunitstayid']
final_dataset = final_dataset[~final_dataset['patientunitstayid'].isin(cmo)]
#removes 106. 

#Remove DNR/no CPR folks who died after ICU discharge within 72 hours.
#Find those who were marked DNR or no CPR.
dnr = cpg[(cpg['cplitemvalue']=='Do not resuscitate') | 
          (cpg['cplitemvalue']=='No CPR')]
dnr = dnr['patientunitstayid']
dnr = final_dataset['patientunitstayid'].isin(dnr)
#Find those who died within 72 hours.
dead = final_dataset['death_after_disch']==True

#Remove if both were true. 
final_dataset = final_dataset[~(dead & dnr)]

#%% Remove those still with 0 readmission time.

readmit_times = all_rel_stays[
    all_rel_stays['patientunitstayid'].isin(
        final_dataset['patientunitstayid'])]
readmit_times = readmit_times[readmit_times['readmission?']==True]

#30 of these have 0 as their time from discharge to next admit. 
surg_zero_readmit_time = \
    readmit_times[readmit_times\
                        ['time_from_this_disch_to_next_admit']==0]
zero_times = readmit_times[readmit_times[
    'time_from_this_disch_to_next_admit']==0][['patientunitstayid']]

#Drop the stays.
final_dataset = final_dataset[~final_dataset['patientunitstayid'].isin(
    zero_times['patientunitstayid'])]

#%% Drop where readmissions had admit source of ED, other ICU, or Direct Admit. 

drop_locs = ['Other ICU','Direct Admit','ED']
readmit_disch_locs = all_rel_stays[all_rel_stays[
    'patientunitstayid'].isin(final_dataset['patientunitstayid'])]
readmit_disch_locs = readmit_disch_locs[
    readmit_disch_locs['readmission?']==True]
readmit_disch_locs = readmit_disch_locs[readmit_disch_locs['next_admitsource']
                                        .isin(drop_locs)]
#Drop the stays.
final_dataset = final_dataset[~final_dataset['patientunitstayid'].isin(
    readmit_disch_locs['patientunitstayid'])]

#%% Remove stays that were missing any PTS signals. 

# Combine BP data together and remove patients with no BP data. 
for col in ['systolic','diastolic','mean']:
    #Just get invasivead non invasive BP data. 
    inv = pd.read_csv(pts_path.joinpath(
        'all_pre-processed_systemic' + col + '.csv'))
    noninv = pd.read_csv(pts_path.joinpath(
        'all_pre-processed_noninvasive' + col + '.csv'))
    # Rename columns for combination.
    inv.rename(columns={'systemic' + col:'all' + col},inplace=True)
    noninv.rename(columns={'noninvasive' + col:'all' + col},inplace=True)
    #Combine it. 
    bp_data = pd.concat([inv,noninv])
    # Drop nans. 
    bp_data.dropna(inplace=True)
    # Just get the stay IDs that had data.
    bp_data = bp_data['patientunitstayid'].drop_duplicates()
    # Drop stays from the dataset that did not have the PTS signal.
    stays_with_data = final_dataset['patientunitstayid'].isin(bp_data)
    final_dataset = final_dataset[stays_with_data]

# Set up columns to check. 
per_cols = ['sao2','heartrate','respiration']

for per_col in per_cols:
    # Just get this column.
    data = pd.read_csv(pts_path.joinpath('all_pre-processed_' + per_col + '.csv'))
    # Drop rows with nans. 
    data.dropna(inplace=True)
    # Just get the stay IDs that had data.
    data = data['patientunitstayid'].drop_duplicates()
    # Drop stays from the dataset that did not have the PTS signal.
    stays_with_data = final_dataset['patientunitstayid'].isin(data)
    final_dataset = final_dataset[stays_with_data]
    
#%% Remove patients with less than 50% coverage of their LOS or last 24h for each PTS signal.
# Do it for whole LOS.
stays_with_prop50 = stays_prop[(stays_prop.iloc[:,1] > 0.5) &
                             (stays_prop.iloc[:,2] > 0.5) &
                             (stays_prop.iloc[:,3] > 0.5) & 
                             (stays_prop.iloc[:,4] > 0.5) &
                             (stays_prop.iloc[:,5] > 0.5) &
                             (stays_prop.iloc[:,6] > 0.5)]
stays_with_prop50 = stays_with_prop50['patientunitstayid']
final_dataset = final_dataset[
    final_dataset['patientunitstayid'].isin(stays_with_prop50)]

# Do it for just last 24h of data.
stays_with_prop50_24 = stays_prop24[(stays_prop24.iloc[:,1] > 0.5) &
                                    (stays_prop24.iloc[:,2] > 0.5) &
                                    (stays_prop24.iloc[:,3] > 0.5) & 
                                    (stays_prop24.iloc[:,4] > 0.5) &
                                    (stays_prop24.iloc[:,5] > 0.5) &
                                    (stays_prop24.iloc[:,6] > 0.5)]
stays_with_prop50_24 = stays_with_prop50_24['patientunitstayid']
final_dataset = final_dataset[
    final_dataset['patientunitstayid'].isin(stays_with_prop50_24)]
    
#%%Save off data set with labels. 
final_dataset.sort_values('patientunitstayid',inplace=True)
#Make label columns ints. 
for col in ['readmission?','death_after_disch','bad_disch_plan']:
    final_dataset[col] = final_dataset[col].astype(int)
final_dataset.to_csv('ICU_readmissions_dataset.csv',index=False)
#final_dataset.to_csv('ICU_readmissions_dataset_w_12hr_less_stays.csv',index=False)

final_readmit_count = final_dataset.groupby('readmission?').count()
final_dead_count = final_dataset.groupby('death_after_disch').count()
final_label_count = final_dataset.groupby('bad_disch_plan').count()

#%%Look at discharge location, next admit source, next stay type of all surgical
#readmissions.
surg_readmit = all_rel_stays[all_rel_stays['readmission?']==True]
surg_readmit = surg_readmit[surg_readmit['patientunitstayid'].isin(
    final_dataset['patientunitstayid'])]

surg_readmit_disch_locs = \
surg_readmit[['patientunitstayid','unitdischargelocation']]\
        .groupby('unitdischargelocation').count()
surg_readmit_disch_locs['prop'] = \
    np.round(surg_readmit_disch_locs['patientunitstayid']/ \
        surg_readmit_disch_locs['patientunitstayid'].sum(),3)
surg_readmit_disch_locs.sort_values('patientunitstayid',
                                     ascending=False,inplace=True)

surg_readmit_admitsource = \
surg_readmit[['patientunitstayid','next_admitsource']]\
        .groupby('next_admitsource').count()
surg_readmit_admitsource['prop'] = \
    np.round(surg_readmit_admitsource['patientunitstayid']/ \
        surg_readmit_admitsource['patientunitstayid'].sum(),3)
surg_readmit_admitsource.sort_values('patientunitstayid',
                                     ascending=False,inplace=True)

surg_readmit_next_staytype = \
surg_readmit[['patientunitstayid','next_staytype']]\
        .groupby('next_staytype').count()
surg_readmit_next_staytype['prop'] = \
    np.round(surg_readmit_next_staytype['patientunitstayid']/ \
        surg_readmit_next_staytype['patientunitstayid'].sum(),3)
surg_readmit_next_staytype.sort_values('patientunitstayid',
                                     ascending=False,inplace=True)

#%% Get distribution of re-admit times. 
readmit_times_distr = all_rel_stays[all_rel_stays['patientunitstayid'].isin(final_dataset['patientunitstayid'])]
readmit_times_distr = readmit_times_distr[readmit_times_distr['readmission?']==True]

#Get the ones with less than 12 hours LOS. 
short_los_readmit_time = \
    readmit_times_distr[readmit_times_distr['unitdischargeoffset']<=720]

readmit_times_distr = short_los_readmit_time['time_from_this_disch_to_next_admit']/60
plt.figure()
plt.title('Readmit times of LOS<12 hour stays')
plt.xlabel('Hours')
readmit_times_distr.plot.hist(grid=True, bins=30, rwidth=0.9)



calc_time = time() - start