# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 10:25:04 2020

This code isolates all the labs that occurred during relevant ICU stays.
It also combines the Total CO2, bicarbonate, and HCO3 data.
It calls the function in LabsDuringStay.py.
After this, run the relevant lab feature extraction code.

@author: Kirby
"""
import pandas as pd
import LabsDuringStay as lds
from pathlib import Path
from time import time

start = time()
filepath = Path(__file__).parent

all_lab_names = pd.read_csv("RawLabsList.csv")
all_lab_names_list = all_lab_names.values.tolist()
all_lab_names_list = [item for sublist in all_lab_names_list for item in sublist]
for lab_name in all_lab_names_list:
    labs_before_del = lds.labs_before_delirium(lab_name)
    labs_before_del.to_csv(filepath.joinpath("AllLabsDuringStay",lab_name + ".csv"),index=False)

#%% Combine the 3 different sources of Bicarbonate data (HCO3, Total CO2, bicarbonate)
bicarb = pd.read_csv(filepath.joinpath("AllLabsDuringStay","bicarbonate.csv"))
hco3 = pd.read_csv(filepath.joinpath("AllLabsDuringStay","HCO3.csv"))
total_CO2 = pd.read_csv(filepath.joinpath("AllLabsDuringStay","Total CO2.csv"))
all_data = pd.concat([bicarb,hco3,total_CO2])
all_data.sort_values(['patientunitstayid','labresultoffset'],inplace=True)
all_data.to_csv(filepath.joinpath("AllLabsDuringStay","bicarbonate_totalCO2_HCO3.csv"),index=False)

calc_time = time() - start