# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 12:57:24 2021



@author: Kirby
"""

#%% Inputs.

prefix_list = ['all_orig_features','1beforedisch','3beforedisch','6beforedisch',
               '12beforedisch',
               '24beforedisch','36beforedisch',
               'alldiastolic_1hrintervals','allmean_1hrintervals',
               'allsystolic_1hrintervals','heartrate_1hrintervals',
               'respiration_1hrintervals','sao2_1hrintervals',
               'mixed_features']

# thresholds_list = [['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02','0.03'],
#                    ['','0.0001','0.001','0.002','0.003','0.005'],
#                    ['','0.0001','0.001','0.002','0.003','0.005'],
#                    ['','0.0001','0.001','0.002','0.003','0.005'],
#                    ['','0.0001','0.001','0.002','0.003','0.005'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02','0.03'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02','0.03','0.05'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02','0.03','0.05'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02'],
#                    ['','0.0001','0.001','0.002','0.003','0.005','0.01','0.02','0.03','0.05'],
#                    ['','0.005','0.01']
#                    ]


thresholds_list = [['','100','80','70','60','50','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','70','60','50','40','30','20','10'],
                   ['','100','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10'],
                   ['','100','80','70','60','50','40','30','20','10']]

#%% Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Read in test ROC, outer loop AUROC, and St Dev

all_data = dict()
for i in range(0,len(prefix_list)):
    
    thresholds = thresholds_list[i]
    prefix = prefix_list[i]
    
    data = pd.DataFrame(columns = ['Outer Loop Mean AUROC',
                                   'Outer Loop AUROC Standard Deviation',
                                   'Test Set AUROC'])
    # Set up feature importance column.
    data.index.names = ['Feature Importance Threshold']
    
    for threshold in thresholds:
        # Get outer loop AUROCS
        if threshold == '':
            file = open(prefix + threshold + '_rf_results_pipelineV3.txt','r')
        else:
            file = open(prefix + '_' + threshold + '_rf_results_pipelineV3.txt','r')
        mean = file.read()
        file.close()
        if threshold == '':
            data.loc[0,'Outer Loop Mean AUROC'] = mean.split(' ',2)[1]
        else:
            data.loc[threshold,'Outer Loop Mean AUROC'] = mean.split(' ',2)[1]
    
        # Get outer loop ST Devs
        if threshold == '':
            file = open(prefix + threshold + '_rf_results_pipelineV3.txt','r')
        else:
            file = open(prefix + '_' + threshold + '_rf_results_pipelineV3.txt','r')
        std = file.read()
        file.close()
        if threshold == '':
            data.loc[0,'Outer Loop AUROC Standard Deviation'] = std.split('(',3)[1].split(')',1)[0]
        else:
            data.loc[threshold,'Outer Loop AUROC Standard Deviation'] = std.split('(',3)[1].split(')',1)[0]
        
        
        # Get Test ROCs
        if threshold == '':
            test_roc = pd.read_csv(prefix + threshold + '_rf_ROC_test.csv')
        else:
            test_roc = pd.read_csv(prefix + '_' + threshold + '_rf_ROC_test.csv')
        test_roc = test_roc.iloc[0,-1]
        test_roc = np.round(test_roc, decimals = 3)
        if threshold == '':
            data.loc[0,'Test Set AUROC'] = test_roc
        else:
            data.loc[threshold,'Test Set AUROC'] = test_roc
    
    all_data[prefix] = data.astype(float)
    

    
#%% Make some plots? 

# Get dict of max AUROCs from each exploratory feature space. 
max_rocs = dict()
for key in all_data.keys():
    perf = all_data.get(key)
    loop_max = perf['Outer Loop Mean AUROC'].max(skipna=True)
    loop_std = perf.loc[perf[perf['Outer Loop Mean AUROC'] == loop_max].index,'Outer Loop AUROC Standard Deviation'][0]
    ci_95 = loop_std*1.96
    test_max = perf['Test Set AUROC'].max(skipna=True)
    max_rocs[key] = [loop_max,loop_std,ci_95,test_max]

# Plot Max AUROCs in bar chart for exploratory feature spaces.
max_rocs = pd.DataFrame.from_dict(max_rocs, orient='index', 
                                  columns=['Outer Loop Mean AUROC',
                                           'Outer Loop Standard Deviation',
                                           'Outer Loop 95 CI',
                                           'Test Set AUROC'])

new_labels = ['Low Freq Features',
              'Last 1 Hr, High Freq',
              'Last 3 Hrs, High Freq',
              'Last 6 Hrs, High Freq',
              'Last 12 Hrs, High Freq',
              'Last 24 Hrs, High Freq',
              'Last 36 Hrs, High Freq',
              'Diastolic BP, High Freq',
              'Mean BP, High Freq',
              'Systolic BP, High Freq',
              'Heart Rate, High Freq',
              'Respiration, High Freq',
              'SaO2, High Freq',
              'Mixed Freq Features']

plt.figure()
plt.title('Best Outer Loop Mean AUROCs of Different Feature Spaces')
plt.ylabel('AUROC')
plt.xlabel('Feature Space')
ax = max_rocs['Outer Loop Mean AUROC'].plot.bar()
ax.set_xticklabels(new_labels)

# Plot of outer loop results and CIs.
plt.figure()
plt.title('Best AUROCs of Different Feature Spaces')
plt.ylabel('AUROC')
plt.xlabel('Feature Space')
plt.errorbar(list(range(1,15)),max_rocs['Outer Loop Mean AUROC'],
            xerr = max_rocs['Outer Loop 95 CI'],
            yerr = max_rocs['Outer Loop 95 CI'],
            linestyle='',
            marker = 'o')
plt.errorbar(list(range(1,15)),max_rocs['Test Set AUROC'],
            linestyle='',
            marker = 'o',
            color = 'r')
plt.legend(['Outer Loop Mean AUROC, with 95% CIs',
            'Test Set AUROC'], bbox_to_anchor=(1.05, 1), loc='upper left')


# Plot test set results. 
plt.figure()
plt.title('Best Test Set AUROCs of Different Feature Spaces')
plt.ylabel('AUROC')
plt.xlabel('Feature Space')
ax = max_rocs['Test Set AUROC'].plot.bar()
ax.set_xticklabels(new_labels)

plt.figure()
plt.title('Best Test Set AUROCs of Different Feature Spaces')
plt.ylabel('AUROC')
plt.xlabel('Feature Space')
ax = plt.errorbar(list(range(1,15)),max_rocs['Test Set AUROC'],
             linestyle='',
             marker = 'o')

# Plot scatter of test/loop rocs by threshold across different features spaces. 
plt.figure()
plt.title('Effect of Feature Importance Thresholds on Performance')
plt.ylabel('AUROC')
plt.xlabel('% of Features Kept')
for key in all_data.keys():
    temp_rocs = all_data.get(key)
    temp_rocs = temp_rocs.iloc[1:temp_rocs.shape[0],:]
    temp_rocs.reset_index(inplace=True)
    plt.plot(temp_rocs['Feature Importance Threshold'].astype(int),temp_rocs['Outer Loop Mean AUROC'])
    # temp_rocs['Outer Loop Mean AUROC'].plot()
plt.legend(new_labels, bbox_to_anchor=(1.05, 1))



# Plot ROCs of non-pts, 12hr after discharge, and the combined pruned model.
# plt.figure()
# plt.title()

# 
