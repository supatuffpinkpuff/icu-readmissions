# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:44:53 2020

Pulls whether history was marked for a patient, for each different history
option in eICU's pasthistory table.

runtime: 2 seconds.

@author: Kirby
"""

def full_script():
    #%%
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from time import time
    
    start = time()
    
    filepath = Path(__file__)
    dataset_path = filepath.parent.parent.parent.joinpath('Cohort')
    eicu_path = filepath.parent.parent.parent.parent.joinpath('eicu')
    
    #%% read in lists of history paths, and names of lists
    paths = pd.read_csv("HistoryFeatureLists.csv")
    pathlistlist = paths.values.astype(str).tolist()
    
    names = pd.read_csv("HistoryListNames.csv")
    nameslist = names.values.astype(str).tolist()
    nameslist = [item for sublist in nameslist for item in sublist]
    
    # import in all history data
    hist = pd.read_csv(eicu_path.joinpath("pastHistory.csv"),
                       usecols=['patientunitstayid','pasthistorypath'])
    
    # only keep data with relevant patient unit stay ids
    comp = pd.read_csv(dataset_path.joinpath('ICU_readmissions_dataset.csv'))
    compHist = hist[hist['patientunitstayid'].isin(comp['patientunitstayid'])]
    
    # for each path list, check if there are rows for it. If there are, mark it as such. 
    features = comp.copy()
    for counter in range(0,len(nameslist)):
        # keep rows with relevant paths
        tempHist = compHist[compHist['pasthistorypath'].isin(
            pathlistlist[counter])]
        tempHist = tempHist.drop(columns=['pasthistorypath'])
        tempHist = tempHist.drop_duplicates()
        tempHistList = tempHist.values.astype(str).tolist()
        tempHistList = [item for sublist in tempHistList for item in sublist]
        features[nameslist[counter]] = features['patientunitstayid'].isin(
            tempHistList).astype(int)
            
    features.to_csv("HistoryFeatures.csv",index=False)
    
    calc = time() - start
    
if __name__ == '__main__':
    full_script()