# -*- coding: utf-8 -*-
"""
Created on July 31 17:55:30 2021

Generates thresholds to use to get pruned features paces. 
Removes 0 feature importance features, and then thresholds at 10% increments 
to get pruned features spaces (10%, 20%, 30%... of features kept)

@author: Kirby
"""

def generate_thresholds(filename):
    import pandas as pd
    import numpy as np
    
    # Make thresholds that represent 10% increments of features.
    feat_imp = pd.read_csv(filename).iloc[:,1]
    # Remove features that have 0 feature importance. 
    feat_imp = feat_imp[feat_imp > 0]
    increm = np.floor(len(feat_imp)/10)
    thresh_idx = []
    thresh_idx.append(len(feat_imp)-1)
    for i in range(1,9):
        thresh_idx.append(increm*i)
    thresh_idx.append(0)
    thresholds = feat_imp[thresh_idx].sort_values()
    
    return thresholds
