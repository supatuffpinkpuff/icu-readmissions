# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 09:14:34 2021

Combines best performing non PTS features with best performing PTS features. 

runtime: a few seconds.

@author: Kirby
"""
import numpy as np
import pandas as pd
from pathlib import Path
from time import time
from datetime import datetime
import pickle
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer

start = time()
now = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))

file_path = Path(__file__)
feature_path = file_path.parent.parent.joinpath('Features')
cohort_path = file_path.parent.parent.joinpath('Cohort')

#%% Load in data. 
nonpts_cat = pd.read_csv('categorical_data.csv')
nonpts_num = pd.read_csv('numeric_data.csv')
nonpts_cols = pd.read_csv('column_names.csv')
nonpts_top = pd.read_csv('all_orig_features_rf_Top_Features.csv')

pts_cat = pd.read_csv('categorical_data_12beforedisch.csv')
pts_num = pd.read_csv('numeric_data_12beforedisch.csv')
pts_cols = pd.read_csv('column_names_12beforedisch.csv')
pts_top = pd.read_csv('12beforedisch_rf_Top_Features.csv')

# convert commas in column names to underscores.
for data in [nonpts_cat,nonpts_num,pts_cat,pts_num]:
    for col_name in data.columns:
        data.rename(columns={col_name:col_name.replace(',','_')},inplace=True)

#%% Combine the num and cat data.
nonpts_data = pd.concat([nonpts_cat,nonpts_num],axis=1)
pts_data = pd.concat([pts_cat,pts_num],axis=1)

#%% Get thresholded versions of these data.

# Cut down features for non-pts. 
def cut_down_feats(top,thresh,data,cols):
    top = top[top['feat_importance'] >= thresh].copy()
    top = top['column_name']
    # Strip extra quotations at front and back generated somehow. 
    for i in range(0,len(top)):
        temp = top[i]
        if temp.startswith('"') == True & temp.endswith('"') == True:
            temp = list(temp)
            temp.pop(-1)
            temp.pop(0)
            temp = ''.join(temp)
            top[i] = temp
        # Remove double quotes.
        top[i] = top[i].replace('""','"')
    
    # Cut down columns.
    data = data.loc[:,top]
    cols = cols[cols['0'].isin(top)]
    
    return data,cols,top

nonpts_data,nonpts_cols,nonpts_top = cut_down_feats(nonpts_top,0.000520,nonpts_data,nonpts_cols) 
pts_data,pts_cols,pts_top = cut_down_feats(pts_top,0.0024275,pts_data,pts_cols) 

#%% Combine PTS and non PTS data and save it off. 
all_data = pd.concat([nonpts_data,pts_data],axis=1)
# Split it into numerical and categorical data. 
pts_num_cols = all_data.columns.intersection(pts_num.columns)
nonpts_num_cols = all_data.columns.intersection(nonpts_num.columns)
all_num_cols = pts_num_cols.union(nonpts_num_cols,sort=False)
all_num = all_data[all_num_cols]

pts_cat_cols = all_data.columns.intersection(pts_cat.columns)
nonpts_cat_cols = all_data.columns.intersection(nonpts_cat.columns)
all_cat_cols = pts_cat_cols.union(nonpts_cat_cols,sort=False)
all_cat = all_data[all_cat_cols]

# Save off numerical, categorical, and column names. 
all_num.to_csv('numeric_data_mixed.csv',index=False)
all_cat.to_csv('categorical_data_mixed.csv',index=False)

# Column names.
cols1 = all_num.columns.to_frame(index=False)
cols2 = all_cat.columns.to_frame(index=False)
all_cols = pd.concat([cols1,cols2],axis=0)
all_cols.to_csv('column_names_mixed.csv',index=False)