# -*- coding: utf-8 -*-
"""
Created on July 14 17:55:30 2021

Runs an LR model on the mixed features.

@author: Kirby
"""

import pandas as pd
from PipelineV3 import run_model

# Run the RF model on the original feature space (no PTS)
run_model(model_set = 'lr',
          use_prev_model = False,
          prev_model_file = 'no model being used',
          thresh = 0.00,
          num_data_name = 'numeric_data_mixed.csv',
          cat_data_name = 'categorical_data_mixed.csv',
          col_names_name = 'column_names_mixed.csv',
          output_name = 'mixed_features',
          shapley = False)
