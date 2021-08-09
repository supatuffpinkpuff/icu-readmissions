# -*- coding: utf-8 -*-
"""
Created on Sat Dec 19 17:55:30 2020

Run Create full feature space.py first!

This code takes in a feature space, does pre-processing on it, 
and runs an SVM on it with cross validation.

Run times: 
LR - 
RF - 30 min with 23 processors on MARCC
XGB - 12 hours with 23 processors on MARCC.


@author: Kirby
"""
#%% Pick which model to use.
#Options: 'lr','rf','svm', 'xgb'
model_set = 'rf'

#Use previous model for feature selection?
use_prev_model = False
prev_model_file = '18-05-2021_14-19-40_rfmodel.pkl'
thresh = 0.001

#Specify data file names.
num_data_name = 'numeric_datasao2_1hrintervals.csv'
cat_data_name = 'categorical_datasao2_1hrintervals.csv'
col_names_name = 'column_namessao2_1hrintervals.csv'

#Output file name override. Leave as empty string to use a time stamp.
output_name = '1hr_intervals_sao2'


def run_model(model_set,use_prev_model,prev_model_file,thresh,num_data_name,
              cat_data_name,col_names_name,output_name,shapley):
    #%% Package setup. 
    #Miscellaneous packages. 
    from matplotlib import pyplot as plt
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from time import time
    from datetime import datetime
    import pickle
    import inspect as insp
    import os
    import warnings
    
    #Preprocessing
    from sklearn import preprocessing
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import train_test_split
    
    #Models
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    
    #Evaluation
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import GridSearchCV
    from scipy.stats import uniform
    from scipy.stats import loguniform
    from scipy.stats import randint
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve
    from sklearn.metrics import plot_roc_curve
    from sklearn.feature_selection import SelectFromModel
    import shap
    
    start = time()
    # Get current date/time for file names.
    if output_name == '':
        now = str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))
    else:
        now = output_name
    
    # Get relative file paths.
    filename = insp.getframeinfo(insp.currentframe()).filename
    file_path = os.path.dirname(os.path.abspath(filename))
    wd = Path(file_path)
    print(file_path)
    parent = wd.parent
    cohort_path = parent.joinpath('Cohort')
    feature_path = parent.joinpath('Features')
    
    
    #%% Load in data. 
    
    ids = np.genfromtxt(cohort_path.joinpath('ICU_readmissions_dataset.csv'), 
                          delimiter=',',skip_header=1)
    # ids = np.genfromtxt('ICU_readmissions_dataset.csv', 
    #                      delimiter=',',skip_header=1,)
    #Numeric data.
    num_data = np.genfromtxt(wd.joinpath(num_data_name),
                             delimiter=',',skip_header=1)
    
    #Categorical/binary data, already one-hot encoded.
    cat_data = np.genfromtxt(wd.joinpath(cat_data_name),
                             delimiter=',',skip_header=1)
    
    #Column names. 
    cols = np.genfromtxt(wd.joinpath(col_names_name),
                         delimiter=',',dtype=str,skip_header=1)
    
    y = ids[:,3].astype(bool)
    #For testing.
    print(y)
    
    #%% Split the data 80-20.
    num_train, num_test, cat_train, cat_test, y_train, y_test = train_test_split(
        num_data, cat_data, y, test_size=0.2, random_state=1, shuffle=True)
    
    #%% Put together and standardize numerical stuff. 
    
    scaler = preprocessing.StandardScaler().fit(num_train)
    num_train = scaler.transform(num_train)
    num_test = scaler.transform(num_test)
    
    X_train = np.concatenate([num_train,cat_train],axis=1)
    X_test = np.concatenate([num_test,cat_test],axis=1)
    
    #%% Do imputation. 
    
    imp = SimpleImputer(missing_values=np.nan, strategy='median')
    imp.fit(X_train)
    X_train = imp.transform(X_train)
    X_test = imp.transform(X_test)
    
    #%% Select features to model on, if desired.
    if use_prev_model == True:
        # Cut down features. 
        with open(prev_model_file, 'rb') as p:
            prev_model = pickle.load(p)
        selection = SelectFromModel(prev_model, threshold=thresh, prefit=True)
        X_train = selection.transform(X_train)
        X_test = selection.transform(X_test)
        # Cut down list of columns. 
        cols = selection.transform(cols.reshape(1,len(cols)))
        cols = cols.reshape(cols.shape[1],)
        # Don't run the model if the threshold cuts to less than 2 features. 
        if X_train.shape[1] < 2:
            warnings.warn("Threshold cuts too many features. Drop it.")
            return
    
    #%% Model selection and parameters to optimize over.
    
    #Random Forest
    if model_set == 'rf':
        model = RandomForestClassifier(random_state=1)
        params = {'n_estimators':randint(10,1000),
             'max_features':randint(1,X_train.shape[1]),
             'max_depth':randint(1,30),
             'min_samples_leaf':randint(1,100),
             'min_samples_split':randint(2,100)}
    #Logistic Regression. l2 is best penalty.
    elif model_set == 'lr':
        model = LogisticRegression(solver='saga',max_iter=10000,verbose=1)
        params = {'C':loguniform(a=10**-3,b=10**2),
                  'penalty':['l2']}
    #Support Vector Classifier with "rbf" kernel
    elif model_set == 'svm':
        model = SVC(kernel="rbf")
        params = {"C": uniform(loc=0,scale=100),
                  "gamma": uniform(loc=0,scale=1)}
    #XGBoost
    elif model_set == 'xgb':
        model = XGBClassifier(use_label_encoder=False)
        params = {'eta':uniform(loc=0,scale=1),
                  'gamma':randint(0,100),
                  'max_depth':randint(1,X_train.shape[1]),
                  'n_estimators':randint(10,1000)}
    
    
    #%% Nested cross validation.
    f = open(now + '_' + model_set + '_results_pipelineV3.txt','a')
    
    # configure the cross-validation procedure
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    
    # define search
    # search = GridSearchCV(model, params, scoring='roc_auc', n_jobs=3, cv=cv_inner, 
    #                       verbose=0, refit=True)
    search = RandomizedSearchCV(model, params, n_iter=30, scoring='roc_auc', 
                                n_jobs=-2, cv=cv_inner, verbose=1, refit=True, 
                                random_state=1)
    # configure the cross-validation procedure
    cv_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
    # execute the nested cross-validation
    scores = cross_val_score(search, X_train, y_train, scoring='roc_auc', 
                             cv=cv_outer, n_jobs=-2, verbose=1)
    # report performance
    print('AUROC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    f.write('AUROC: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
    
    #%% Get best model, Train it on the full dataset, and evaluate on test dataset. 
    search = RandomizedSearchCV(model, params, n_iter=30,scoring='roc_auc', 
                                cv=cv_inner, n_jobs=-2,verbose=1,refit=True, 
                                random_state=1)
    # execute search
    result = search.fit(X_train, y_train)
    # get the best performing model fit on the whole training set
    best_model = result.best_estimator_
    print(best_model.get_params())
    f.write('\n' + str(best_model.get_params()))
    classifier = best_model.fit(X_train,y_train)
    y_test_pred = classifier.predict_proba(X_test)[:,1]
    fpr,tpr,thresholds = roc_curve(y_test,y_test_pred)
    roc_score = roc_auc_score(y_test,y_test_pred)
    roc = pd.DataFrame(data={'fpr':fpr,
                             'tpr':tpr,
                             'roc':roc_score})
    roc.to_csv(now + '_' + model_set + '_ROC_test.csv',index=False)
    plt.figure()
    plt.plot(fpr,tpr)
    plt.plot([0,1],[0,1],color='red',linestyle='dashed')
    plt.title('Test Set ROC Curve')
    plt.legend(['AUROC='+str(np.round_(roc_score,decimals=3))])
    plt.savefig(now + '_' + model_set + '_Test.png',bbox_inches='tight')
    plt.show()
    
    with open(now + '_' + model_set + '_model.pkl', "wb") as m:
        pickle.dump(classifier, m)
        
    #%% If shapley == True, get a Shapley summary plot too.
    if shapley == True:
        ex = shap.Explainer(classifier, X_train)
        shap_values = ex.shap_values(X_train,check_additivity=False)
        # shap_obj = ex(X_train,check_additivity=False)
        fig = plt.figure()
        shap.summary_plot(shap_values, X_train)
        #shap.plots.beeswarm(shap_values)
        fig.savefig(now + '_' + model_set + '_shapley.png', bbox_inches='tight')
    #%% Get feature importance. 
    
    #Random Forest
    if model_set == 'rf':
        rf_top_feat = classifier.feature_importances_
        top_feats = pd.DataFrame(data={'column_name':cols,
                                       'feat_importance':rf_top_feat})
        top_feats.sort_values('feat_importance',ascending=False,inplace=True)
        top_feats.iloc[0:20,:].sort_values(
            'feat_importance',ascending=True).plot.barh(
                x='column_name',y='feat_importance')
        top_feats.to_csv(now + '_' + model_set + '_Top_Features.csv',index=False)
        plt.title('RF Feature Importance')
        plt.savefig(now + '_' + model_set + '_Top_Features.png',
                    bbox_inches='tight')
        plt.show()
    #Logistic Regression. l2 is best penalty.
    elif model_set == 'lr':
        lr_top_feat = classifier.coef_[0]
        top_feats = pd.DataFrame(data={'column_name':cols,
                                       'coefficient':lr_top_feat})
        top_feats['abs_val_coefficient'] = top_feats['coefficient'].abs()
        top_feats.sort_values('abs_val_coefficient',ascending=False,inplace=True)
        top_feats.iloc[0:20,:].sort_values(
            'abs_val_coefficient',ascending=True).plot.barh(
                x='column_name',y='coefficient')
        top_feats.to_csv(now + '_' + model_set + '_Top_Features.csv',index=False)
        plt.title('LR Feature Importance')
        plt.savefig(now + '_' + model_set + '_Top_Features.png',
                    bbox_inches='tight')
        plt.show()
    elif model_set == 'xgb':
        rf_top_feat = classifier.feature_importances_
        top_feats = pd.DataFrame(data={'column_name':cols,
                                       'feat_importance':rf_top_feat})
        top_feats.sort_values('feat_importance',ascending=False,inplace=True)
        top_feats.iloc[0:20,:].sort_values(
            'feat_importance',ascending=True).plot.barh(
                x='column_name',y='feat_importance')
        top_feats.to_csv(now + '_' + model_set + '_Top_Features.csv',index=False)
        plt.title('XGB Feature Importance')
        plt.savefig(now + '_' + model_set + '_Top_Features.png',
                    bbox_inches='tight')
        plt.show()
    
    calc = time() - start
    f.write('\nCalculation time: ' + str(calc/60) + ' min.')
    f.close()

if __name__ == '__main__':
    run_model(model_set,use_prev_model,prev_model_file,thresh,num_data_name,
              cat_data_name,col_names_name,output_name)