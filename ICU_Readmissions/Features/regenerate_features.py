# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:56:25 2021

Reruns all feature code to regenerate features, excluding PTS. 

Usually needs to be done after making a new data set. 

@author: Kirby
"""
from pathlib import Path
file_path = Path(__file__).parent
import importlib.util
import os

# List out folders and files to run. 
folders_and_files = [('Static','StaticFeatures.py'),
                     ('Comorbidity','Elixhauser.py'),
                     ('Dialysis','DialysisFeature.py'),
                     ('History','HistoryFeatures.py'),
                     ('IntakeOutput','UrineTransfusions.py'),
                     ('Labs','LastLabFeatures.py'),
                     ('Medications','Create HICL Drug Name Legend.py'),
                     ('Medications','AllDrugFeatures.py'),
                     ('NurseCharting','LastGCS.py'),
                     ('NurseCharting','LastRASS.py'),
                     ('NurseCharting','LastTemperature.py'),
                     ('Sepsis','SuspectedSepsis.py'),
                     ('Sepsis','InfectionAndSepsisDynamic.py'),
                     ('Ventilation','MVDurationDynamic.py')
                     ]

# Use this to run just part of it. 
# folders_and_files = [('IntakeOutput','UrineTransfusions.py'),
#                      ('Labs','LastLabFeatures.py'),
#                      ('Medications','Create HICL Drug Name Legend.py'),
#                      ('Medications','AllDrugFeatures.py'),
#                      ('NurseCharting','LastGCS.py'),
#                      ('NurseCharting','LastRASS.py'),
#                      ('NurseCharting','LastTemperature.py'),
#                      ('Sepsis','SuspectedSepsis.py'),
#                      ('Sepsis','InfectionAndSepsisDynamic.py'),
#                      ('Ventilation','MVDurationDynamic.py')
#                      ]

for folder_and_file in folders_and_files:
    # Get file path of script. 
    folder = folder_and_file[0]
    file = folder_and_file[1]
    curr_path = file_path.joinpath(folder,file)
    # Change working directory.
    os.chdir(curr_path.parent)
    # Run file.
    spec = importlib.util.spec_from_file_location("module.name", curr_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)
    foo.full_script()
