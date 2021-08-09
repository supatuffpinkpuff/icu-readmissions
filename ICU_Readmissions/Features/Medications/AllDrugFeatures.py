# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:45:34 2020

# This file uses a list of paths to csvs, with a list of drug names to search for. 
# Then spits out relative medication features for each one.
Can modify hours to get different time amounts before delirium onset.

Runtime: ~15 min

@author: Kirby
"""
def full_script():
    #%% Setup
    import pandas as pd
    import DrugFeaturesFunction as df
    import os.path
    import glob
    import time as time
    from pathlib import Path
    
    file_path = Path(__file__)
    parent_path = file_path.parent
    cohort_path = file_path.parent.parent.parent.joinpath('Cohort')
        
    start = time.time()
    
    #%% Inputs
    #define all the paths. 
    drugPathList = glob.glob(str(parent_path.joinpath('DrugNameLists','*')))
    
    treatmentPathList = glob.glob(str(parent_path.joinpath('TreatmentStrings','*')))
    
    comp = pd.read_csv(cohort_path.joinpath('ICU_readmissions_dataset.csv'))
    
    #%% error checking
    if len(drugPathList)!=len(treatmentPathList):
        raise NameError('you goofin fam, the lists of paths are different lengths')
    
    for path in drugPathList:
        if os.path.isfile(path) == False:
            raise NameError(path+" is not a valid file path")
            
    for path in treatmentPathList:
        if os.path.isfile(path) == False:
            raise NameError(path+" is not a valid file path")
    
    #%% Generate features.
    for i in range(1,len(drugPathList)):
        comp = pd.concat(
            objs=[comp,df.DrugFeature(drugPathList[i],treatmentPathList[i])]
            ,axis = 1)
        
    comp.to_csv('AllDrugFeatures.csv',index=False)
    
    calc_time = time.time() - start
    
if __name__ == '__main__':
    full_script()