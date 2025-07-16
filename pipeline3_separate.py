from radiomics_utils_final import prep_OPNCB, run_OPNCB, data_path, process_test_set, kruskal_wallis, features
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import warnings
import os
import re
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# %% 
''' EDIT '''
# where are the input files located?
data_path = data_path
# file containing unharmonized data after being passed through filter_data.py
features_file = 'raw_data_no_unfollowed.csv'
# file containing header data
headers_file = 'all_headers.csv'
# names of subgroups
groups = ['LCS','cancer','wedge']
# which acquisition parameters are being harmonized over?
batch_list = ['Manufacturer',
              'FocalSpots',
              'CE',
              'KVP']
# which data type is isolated in a given split?
isolate = 'patient'
# incremental percentage of augmentation dataset to include
# multiples of 5 only (e.g. 5%, 100%)
run_increments = [5,10]

''' END EDIT'''
# %% set up
# this script is for separate harmonization
orig_tag = 'separate'
aug_increment = 5
# read in feature data
features_df = pd.read_csv(data_path+features_file, index_col=0).T
# read in acquisition parameter data
headers = pd.read_csv(data_path+headers_file, index_col=0)
# ensure that acquisition parameter data corresponds to feature data, 
# and in the correct order
headers = headers.loc[list(features_df.index)]
# gets application dataset
LCS = features_df.loc[features_df['group']=='LCS']
# gets augmentation dataset: none of these patients/PNs were ever in the LCS group
augmentation = features_df.loc[
    (features_df['group'] != 'LCS')
]

# %% iteration setup
# trial number
i = 0
for cv in range(10): # loops through trials
    # set up application dataset split
    sgkf = StratifiedGroupKFold(n_splits=5, random_state = cv+10, shuffle = True)
    # set up augmentation dataset split, for ablation test
# %% 
    # 5-fold CV for this trial
    for X_train_ind, X_test_ind in sgkf.split(LCS[features], list(LCS['cancer']),list(LCS[isolate])):
        # gets training and test sets
        train_set, test_set = LCS.iloc[X_train_ind], LCS.iloc[X_test_ind]
        # ablation test for this CV
        augmentation_i = augmentation.loc[~augmentation[isolate].isin(test_set.index)]
        
        skf_aug = StratifiedKFold(n_splits=int(100/aug_increment), shuffle=True, random_state=cv+10)
        # store the augmentation splits
        aug_splits = []
        for _, test_index in skf_aug.split(augmentation_i[features], list(augmentation_i['cancer'])):
            split = augmentation_i.iloc[test_index]
            aug_splits.append(split)
        
        for aug_ind in range(20):
            
            # adds a set fraction of augmentation scans to the training set
            train_set = pd.concat([train_set,aug_splits[aug_ind]],axis='index')
            
            if aug_increment*(aug_ind+1) not in run_increments:
                continue
            
            # labels filenames to save
            tag = orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug'
            # makes output folder for this trial
            out_path = data_path+tag+'_'+str(i)+'/'
            
            harmonized= False
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            elif os.path.exists(out_path):
                harmonized=True
            
            # sets up data to be stored from this trial
            metrics_df = {'test_AUC':[],
                          'test_accuracy':[],
                          'test_weighted_accuracy':[],
                          'test_prevalence':[],
                          'test_FPR':[],
                          'test_FNR':[],
                          'test_specificity':[],
                          'test_sensitivity':[],
                          'num_usable_features':[],
                          'selected_features':[],
                          'alpha':[],
                          'test_N':[],
                          'test_PN_N':[],
                          'test_patient_N':[],
                          'train_N':[],
                          'train_PN_N':[],
                          'train_patient_N':[],
                          'train_prevalence':[],
                          }
            
        # %%
            # splits training and test sets into subgroups
            # specific to separate harmonization
            # loops through subgroups
            
            if not harmonized:
                
                bad_try = 0 # use this to communicate separate harmonization loop errors to augmentation loop
                
                total_scans = []
                
                
                for g in groups:
                    # get subgroup from training set of this split
                    g_train_set = train_set.groupby('group').get_group(g)
                    
                    # for use in file names
                    subtag = tag+'_'+g+'_cv'+str(i)
                    train_tag = subtag+'_train'
                    
                    # save training set to csv
                    g_train_set.to_csv(out_path+train_tag+'.csv')
                    
                    # filters and formats training set
                    try:
                        dat, scans, le_dict, group_col, cancer_col, covars, covars_p = prep_OPNCB(out_path, out_path, out_path+train_tag+'.csv', data_path+headers_file, batch_list, train_tag)
                    except TypeError: # if harmonization will fail
                        bad_try = 1
                        break # exit harmonization loop (don't skip harmonizing just one group)
                    
                    total_scans += scans
                    
                    # read in which parameters are present in the training set
                    valid_headers = pd.read_csv(out_path+train_tag+'_batch_info.csv', index_col=0)
                    
                    # get test set of this split
                    if g == 'LCS':
                        test_tag = subtag+'_test'
                        g_test_set = test_set.groupby('group').get_group(g)
                        
                        # gets headers for this test set
                        test_headers = headers.loc[g_test_set.index] 
                        
                        # filters test set
                        
                        # initialize list of scans to exclude
                        exclude = []
                        # loops through patients
                        for p in list(test_headers.index):
                            # loops through acquisition parameters
                            for b in batch_list:
                                if str(test_headers.loc[p,b]) not in list(valid_headers['Batch']):
                                    # if patients in the test set have parameters not in the training set, 
                                    # those are excluded
                                    exclude.append(p)
                        final_test_group = [p for p in list(test_headers.index) if p not in exclude]
                        
                        # print(list(set(exclude)))
                        print(len(list(set(exclude))), 'patients excluded from test set')
                        # if all the patients were excluded, move on to the next train-test split
                        if len(list(set(exclude)))==len(test_headers.index):
                            print('Incompatible test set. Trying again')
                            # don't harmonize if it's not going to work for the test set
                            bad_try = 1
                            break # exits group harmonization loop
                        
                    # is AUC defined?
                    if len(set(test_set.loc[final_test_group]['cancer']))!=2:
                        print('Only one class present in test set. Trying again')
                        bad_try = 1
                        break
                    
                    # ComBat shouldn't see uniform acquisition parameters
                    inner_batch_list = list(set(list(valid_headers['Acquisition Parameter'])))
                    
                    #harmonizes training set
                    harmonized_data, estimates = run_OPNCB(dat, scans, out_path, group_col, cancer_col, covars, covars_p, inner_batch_list, train_tag, multi_group=False)
                    
                    if g == 'LCS':
                        # applies estimators to test set
                        harmonized_test_data = process_test_set(test_set, final_test_group, test_headers, out_path, inner_batch_list, estimates, test_tag,train_tag,le_dict)
                
                if bad_try:
                    continue # goes to next increment of augmentation, or the next trial
                
                # initialize list of acquisition-dependent features
                blacklist = []
                # these are the groups that should have a normal distribution
                for g in ['cancer','wedge']:
                    # for use in file names
                    subtag = tag+'_'+g+'_cv'+str(i)
                    train_tag = subtag+'_train'
                    # perform kruskal wallis test
                    g_blacklist = kruskal_wallis(g, out_path, out_path+train_tag+'_batch_info.csv', out_path+train_tag+'_harmonized.csv', data_path+headers_file, tag = train_tag)
                    # add acquisition-dependent features from this subgroup to total list
                    blacklist = blacklist + g_blacklist
                # remove repetitions of features
                blacklist = list(set(blacklist))
                print(len(blacklist),'features failed separate harmonization')
                # get acquisition independent features
                good_features = [f for f in features if f not in blacklist]
                
                # if there are no usable features
                if len(blacklist)==len(features):
                    print('No usable features. Trying again')
                    # fixes output file
                    for k in list(metrics_df.keys()):
                        # only this feature is defined
                        if k== 'num_usable_features':
                            continue
                        # placeholder values
                        metrics_df[k].append(np.nan)
                    
                    # convert to dataframe
                    write_metrics_df = pd.DataFrame(metrics_df)
                    # label this row with the trial number
                    write_metrics_df.index = [i]
                    if os.path.exists(data_path+tag+'_harmonization_pipeline_metrics.csv'):
                        # read in existing data sheet
                        existing_metrics_df = pd.read_csv(data_path+tag+'_harmonization_pipeline_metrics.csv', index_col=0)
                        # add to existing data sheet
                        write_metrics_df = pd.concat([existing_metrics_df, write_metrics_df], axis = 0)
                    # save data sheet
                    write_metrics_df.to_csv(data_path+tag+'_harmonization_pipeline_metrics.csv')
                    
            
        # %%
                # combines outputs of runOpNCB for training data
                harmonized_data = pd.concat([pd.read_csv(out_path+tag+'_'+g+'_cv'+str(i)+'_train_harmonized.csv', index_col=0) for g in groups], axis = 0)
                harmonized_data.sort_index(axis = 'index', inplace = True)
                harmonized_data.to_csv(out_path+tag+'_cv'+str(i)+'_train_harmonized.csv')
                print('Separately harmonized data saved to csv')
            
                # make data file for only acquisition-independent features
                good_harmonized_data=harmonized_data[good_features+['group','cancer']]
                print(good_harmonized_data)
                good_harmonized_data.to_csv(out_path+tag+'_cv'+str(i)+'_usable_harmonized.csv')
                print('Usable data saved to csv')
            
        # %%
            print('LASSOing features')
            if harmonized:
                try:
                    good_harmonized_data = pd.read_csv(out_path+tag+'_cv'+str(i)+'_usable_harmonized.csv', index_col=0)
                except:
                    continue
                good_features = [f for f in good_harmonized_data.columns if f in features]
                harmonized_test_data = pd.read_csv(out_path+orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug_LCS_cv'+str(i)+'_test_harmonized.csv', index_col=0)
                final_test_group = list(harmonized_test_data.index)
                total_scans = list(good_harmonized_data.index)
            # convert training data to numpy array
            
            X = good_harmonized_data[good_features].to_numpy()
            y = good_harmonized_data['cancer'].apply(pd.to_numeric).to_numpy()
            # scale training data before LASSO
            scaler = StandardScaler()
            scaler.fit(X)
            X = scaler.transform(X)
            # initialize LASSO
            a = 0.05
            lasso = Lasso(alpha=a, random_state=42)
            # train LASSO
            lasso.fit(X, y)
            # use LASSO to select features
            sel_ = SelectFromModel(lasso, prefit=True)
            # get feature selection information
            f_eval = list(sel_.get_support())
            
            selected = {}
            # loop through features
            for j in range(len(f_eval)):
                if f_eval[j]:
                    # if this feature was selected
                    if good_features[j] in selected:
                        selected[good_features[j]]+=1 # increase selection count
                    else: 
                        selected[good_features[j]]=1 # this was the first time this feature was selected
            # list of LASSO-selected features
            used_f = list(selected.keys())
            print(used_f)
            # add metrics to data sheet
            metrics_df['alpha'].append(a)
            metrics_df['selected_features'].append(used_f)
             
         # %%
            
            harmonized_test_data.reset_index(inplace=True)
            # convert training set to numpy array
            Xf = good_harmonized_data[used_f].apply(pd.to_numeric).to_numpy()   
            # convert test set to numpy array
            X_test = harmonized_test_data[used_f].apply(pd.to_numeric).to_numpy()  
            y_test = harmonized_test_data['cancer'].apply(pd.to_numeric).to_numpy()
            
            # scale features before using SVM
            scaler = StandardScaler()
            # scale training set
            Xf = scaler.fit_transform(Xf)
            # scale test set using same scale as training set
            X_test = scaler.transform(X_test)
        
            # initialize SVM
            svm_model = SVC(kernel='linear', probability = True, random_state=42, C=1)
            # train SVM on training set
            print('Fitting SVM model')
            svm_model.fit(Xf, y)
            
            # classify test set
            print('Predicting')
            y_pred = svm_model.predict(X_test)
        
            g_accuracy = accuracy_score(y_test,y_pred)
            # add accuracy metric to data sheet
            metrics_df['test_accuracy'].append(g_accuracy)
            
            y_scores = svm_model.predict_proba(X_test)[:, 1]  # get probability for class 1
            g_AUC = roc_auc_score(y_test, y_scores)
            
            # gets single values for fnr, fpr
            g_tn, g_fp, g_fn, g_tp = confusion_matrix(y_test, y_pred).ravel()
            # false positive rate
            g_fpr = g_fp/(g_fp+g_tn)
            # false negative rate
            g_fnr = g_fn/(g_fn+g_tp)
            
            # compute weighted accuracy
            weights = compute_sample_weight(class_weight='balanced', y=y_test)
            g_weighted_accuracy = accuracy_score(y_test,y_pred, sample_weight=weights)
            
            # add metrics to data sheet
            metrics_df['num_usable_features'].append(len(good_features))
            
            metrics_df['test_FNR'].append(g_fnr)
            metrics_df['test_FPR'].append(g_fpr)
            metrics_df['test_sensitivity'].append(1-g_fnr)
            metrics_df['test_specificity'].append(1-g_fpr)
            metrics_df['test_AUC'].append(g_AUC)
            metrics_df['test_prevalence'].append(np.mean(y_test))
            metrics_df['test_weighted_accuracy'].append(g_weighted_accuracy)
            
            metrics_df['test_N'] = len(final_test_group)
            metrics_df['test_PN_N'] = len(set([n.split('PRE')[0] for n in final_test_group]))
            metrics_df['test_patient_N'] = len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in final_test_group]))
            metrics_df['train_N'] = len(total_scans)
            metrics_df['train_PN_N'] = len(set([n.split('PRE')[0] for n in total_scans]))
            metrics_df['train_patient_N'] = len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in total_scans]))
            metrics_df['train_prevalence'].append(np.mean(y))
            
        # %%
            write_metrics_df = pd.DataFrame(metrics_df)
            write_metrics_df.index = [i]
    
            if os.path.exists(data_path+tag+'_harmonization_pipeline_metrics.csv'):
                # read in existing data
                existing_metrics_df = pd.read_csv(data_path+tag+'_harmonization_pipeline_metrics.csv', index_col=0)
                # add metrics from this trial to existing data
                write_metrics_df = pd.concat([existing_metrics_df, write_metrics_df], axis = 0)
            # save data sheet with metrics from this trial
            write_metrics_df.to_csv(data_path+tag+'_harmonization_pipeline_metrics.csv')
        
        i += 1
print('\a')
        

 
