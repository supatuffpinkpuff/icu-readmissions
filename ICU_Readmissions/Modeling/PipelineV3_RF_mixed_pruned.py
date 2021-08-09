# -*- coding: utf-8 -*-
"""
Created on July 14 17:55:30 2021

Runs an RF model on the mixed features, then runs pruned models.


@author: Kirby
"""

import pandas as pd
from PipelineV3 import run_model
from generate_thresholds import generate_thresholds

# Run the RF model on the original feature space (no PTS)
run_model(model_set = 'rf',
          use_prev_model = False,
          prev_model_file = 'no model being used',
          thresh = 0.00,
          num_data_name = 'numeric_data_mixed.csv',
          cat_data_name = 'categorical_data_mixed.csv',
          col_names_name = 'column_names_mixed.csv',
          output_name = 'mixed_features',
          shapley = False)

# Run RF model on original feature space with pruning.
# Make thresholds that represent 10% increments of features.
thresholds = generate_thresholds('mixed_features_rf_Top_Features.csv')

# Just get a few of them to run. 
thresholds = thresholds.iloc[0:10]

# Try different thresholds.
# Generate names that indicate proportion of useful features kept.
prop_features = [100,80,70,60,50,40,30,20,10]
i = -1
for threshold in thresholds:
    i += 1
    run_model(model_set = 'rf',
              use_prev_model = True,
              prev_model_file = 'mixed_features_rf_model.pkl',
              thresh = threshold,
              num_data_name = 'numeric_data_mixed.csv',
              cat_data_name = 'categorical_data_mixed.csv',
              col_names_name = 'column_names_mixed.csv',
              output_name = 'mixed_features_' + str(prop_features[i]),
              shapley = False)