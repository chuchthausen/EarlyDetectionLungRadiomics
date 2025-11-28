from radiomics_utils_final import prep_OPNCB, run_OPNCB, data_path, process_test_set, kruskal_wallis, features, fill_nans, update_metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, f1_score, make_scorer, precision_recall_curve, auc, average_precision_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import os
import re
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# %% User settings
''' USER EDIT '''
# where are the input files located?
data_path = data_path
# file containing unharmonized data 
features_file = 'raw_data_no_unfollowed.csv'
# file containing header data
headers_file = 'all_headers.csv'
# suffix of file to write with result summary
# final filenames will begin with harmonization, augmentation, and trial
out_suffix = '_harmonization_L1SVC_metrics.csv'
# names of datasets
groups = ['LCS','cancer','wedge']
# which acquisition parameters are being harmonized over?
batch_list = ['Manufacturer',
              'FocalSpots',
              'CE',
              'KVP']
# covariate or collective harmonization?
orig_tag = 'covariate'
# which data type is isolated in a given split?
isolate = 'patient'
# include augmentation data?
aug = True
# augment the training set at what percentage increment?
# multiples of 5 only (e.g. 5%, 75%)
run_increments = [100]
# perform hyperparameter tuning?
hp_tune = False
# manual hyperparameter input
default_C = 0.1

''' END USER EDIT'''

# %% general set up

# are datasets being distinguished during the harmonization?
if 'covariate' in orig_tag:
    multi_group = True
elif 'collective' in orig_tag:
    multi_group = False

if not aug:
    aug_increment = 0

# read in feature data
features_df = pd.read_csv(data_path+features_file, index_col=0).T
# read in acquisition parameter data
headers = pd.read_csv(data_path+headers_file, index_col=0)

# ensure that acquisition parameter data corresponds to feature data, 
# and in the correct order
headers = headers.loc[list(features_df.index)]
# gets application dataset
LCS = features_df.loc[features_df['group']=='LCS']
# gets augmentation dataset
augmentation = features_df.loc[
    (features_df['group'] != 'LCS')
]

# %% Cross-validation setup
# trial number
i = 0
for cv in range(10): 
    # set up application dataset split
    sgkf = StratifiedGroupKFold(n_splits=5, random_state = cv+10, shuffle = True)
    # set up augmentation dataset split, for learning curve
    # 5-fold CV for this trial
    # splits will be balanced by class as well as possible
    for X_train_ind, X_test_ind in sgkf.split(LCS[features], list(LCS['cancer']),list(LCS[isolate])):
        print('Trial',i)
        # gets training and test sets
        train_set, test_set = LCS.iloc[X_train_ind], LCS.iloc[X_test_ind]
        
        if len(set(test_set['cancer']))!=2:
            print('Only one class in the test set after KFold split. Model cannot be trained')
            with open(data_path+orig_tag+'_'+str(i)+"_error.txt", "a") as f:
                f.write("Only one class remains in the test set after KFold split.\n")
            i+=1
            continue
        
# %% Set up training set augmentation
        
        # gets augmentation data in 4 increments for this CV
        if aug:
            # no patients in the test set will appear in the augmentation data
            augmentation_i = augmentation.loc[~augmentation[isolate].isin(test_set.index)]
            # split augmentation data into random increments
            # increments will be balanced by class
            aug_increment = 5 
            skf_aug = StratifiedKFold(n_splits=int(100/aug_increment), shuffle=True, random_state=cv+10)
            # store the augmentation splits
            aug_splits = []
            for _, test_index in skf_aug.split(augmentation_i[features], list(augmentation_i['cancer'])):
                split = augmentation_i.iloc[test_index]
                aug_splits.append(split)
        
        # cumulative augmentation in increments of 5%
        for aug_ind in range(20):
            # labels filenames to save
            tag = orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug'
            
            if aug:
                # adds a set fraction of augmentation scans to the training set
                train_set = pd.concat([train_set,aug_splits[aug_ind]],axis='index')
                # only calculate for user-defined augmentation fraction
                if aug_increment*(aug_ind+1) not in run_increments:
                    continue
                
# %% Set up results to be saved
            # makes output folder for this trial
            out_path = data_path+tag+'_'+str(i)+'/'
            
            harmonized= False
            
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            elif os.path.exists(out_path):
                harmonized=True
            
            # results to save
            metrics_df = {'test_AUC':[],
                          'test_AUC_CE':[],
                          'test_AUC_nonCE':[],
                          'test_AUC_GE':[],
                          'test_AUC_SIEMENS':[],
                          'test_pr_AUC':[],
                          'test_pr_AUC_CE':[],
                          'test_pr_AUC_nonCE':[],
                          'test_pr_AUC_GE':[],
                          'test_pr_AUC_SIEMENS':[],
                          'test_precision':[],
                          'test_recall':[],
                          'test_f1':[],
                          'test_accuracy':[],
                          'test_weighted_accuracy':[],
                          'test_prevalence':[],
                          'spec_90sens':[],
                          'sens_30spec':[],
                          'sens_50spec':[],
                          'sens_85spec':[],
                          'sens_90spec':[],
                          'test_specificity':[],
                          'test_sensitivity':[],
                          'num_usable_features':[],
                          'num_sel':[],
                          'sel_f':[],
                          'C_clf':[],
                          'test_N':[],
                          'test_PN_N':[],
                          'test_excluded_N':[],
                          'test_patient_N':[],
                          'train_N':[],
                          'train_PN_N':[],
                          'train_excluded_N':[],
                          'train_patient_N':[],
                          'train_prevalence': []
                          }
            
        # %% Harmonize
            if not harmonized: # don't re-do harmonization if it's already done
            
                # for use in file names
                subtag = tag+'_cv'+str(i)
                train_tag = subtag+'_train'
                
                # save training set to csv
                train_set.to_csv(out_path+train_tag+'.csv')
                
                try:
                    dat, total_scans, le_dict, group_col, cancer_col, covars = prep_OPNCB(out_path, out_path, out_path+train_tag+'.csv', data_path+headers_file, batch_list, train_tag)
                except TypeError: # if harmonization will fail
                    with open(out_path+"error.txt", "a") as f:
                        f.write("Training set with these parameters cannot be harmonized.\n")
                    continue # go to the next augmentation increment
                    
                # read in which parameters are present in the training set
                valid_headers = pd.read_csv(out_path+train_tag+'_batch_info.csv')
                
                # label setup
                test_tag = subtag+'_test'
                
                # gets headers for this test set
                test_headers = headers.loc[test_set.index] 
                 
                # filters test set
                
                # initialize list of test scans to exclude
                # this function of the code never had to be used in our primary analysis
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
                
                print(len(list(set(exclude))), 'patients excluded from test set')
                metrics_df['test_excluded_N'].append(len(list(set(exclude))))
                
                # if all the patients were excluded, move on to the next train-test split
                if len(list(set(exclude)))==len(test_headers.index):
                    print('Incompatible test set. Trying again')
                    with open(out_path+"error.txt", "a") as f:
                        f.write("Test set parameters are incompatible with training set parameters.\n")
                    # don't harmonize if it's not going to work for the test set
                    metrics_df = fill_nans(metrics_df)
                    update_metrics(metrics_df,i,tag+out_suffix)
                    if not aug:
                        break
                    else:
                        continue # try again in the next augmentation increment
                
                # is AUC defined?
                if len(set(test_set.loc[final_test_group]['cancer']))!=2:
                    print('Only one class remains in the test set after parameter-based exclusion. Model cannot be trained')
                    with open(out_path+"error.txt", "a") as f:
                        f.write("Only one class remains in the test set after parameter-based exclusion.\n")
                    metrics_df = fill_nans(metrics_df)
                    update_metrics(metrics_df,i,tag+out_suffix)
                    if not aug:
                        break
                    else:
                        continue # try again in the next augmentation increment
                    
                # after exclusion, what still needs to be harmonized?
                inner_batch_list = list(set(list(valid_headers['Acquisition Parameter'])))
                    
                # harmonizes training set
                harmonized_data, estimates = run_OPNCB(dat, total_scans, out_path, group_col, cancer_col, covars, inner_batch_list, train_tag, multi_group=multi_group)
                
                # applies estimators to test set
                harmonized_test_data = process_test_set(test_set, final_test_group, test_headers, out_path, inner_batch_list, estimates, test_tag,train_tag,le_dict)
                
                n_ex_train = len(train_set.index)-len(harmonized_data.index)
                metrics_df['train_excluded_N'].append(n_ex_train)
                
        # %% Read in pre-harmonized data
            if harmonized:
                # does the harmonized test data exist in the folder?
                try:
                    harmonized_test_data = pd.read_csv(out_path+orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug_cv'+str(i)+'_test_harmonized.csv', index_col=0)
                except FileNotFoundError:
                    print('Test data was not harmonized. Trying again')
                    if not aug:
                        break
                    else:
                        continue
                # check if test scans were excluded
                n_ex_test = len(test_set.index)-len(harmonized_test_data.index)
                metrics_df['test_excluded_N'].append(n_ex_test)
                
                # does the harmonized training data exist in the folder?
                try:
                    harmonized_data = pd.read_csv(out_path+tag+'_cv'+str(i)+'_train'+'_harmonized.csv', index_col=0)
                except FileNotFoundError:
                    print('Training data was not harmonized. Trying again')
                    metrics_df = fill_nans(metrics_df)
                    update_metrics(metrics_df,i,tag+out_suffix)
                    if not aug:
                        break
                    else:
                        continue
                # check for excluded training scans
                n_ex_train = len(train_set.index)-len(harmonized_data.index)
                metrics_df['train_excluded_N'].append(n_ex_train)
                
        # %%  Acquisition dependence test                

            # initialize list of acquisition-dependent features
            blacklist = []
            if aug:
                for g in ['cancer','wedge']: # these are the groups that should have a normal distribution
                    subtag = tag+'_cv'+str(i)
                    train_tag = subtag+'_train'
                    g_blacklist = kruskal_wallis(g, out_path, out_path+train_tag+'_batch_info.csv', out_path+train_tag+'_harmonized.csv', data_path+headers_file, tag = tag+'_'+g+'_cv'+str(i)+'_train')
                    # add acquisition-dependent features from this subgroup to total list
                    blacklist = blacklist + g_blacklist
                    
            # remove redundant features
            blacklist = list(set(blacklist))
            print(len(blacklist),'features failed harmonization')
            
            # if there are no usable features
            if len(blacklist)==len(features):
                print('No usable features. Trying again')
                with open(out_path+"error.txt", "a") as f:
                    f.write("No usable features.\n")
                metrics_df = fill_nans(metrics_df)
                update_metrics(metrics_df,i,tag+out_suffix)
                if not aug:
                    break
                else:
                    continue
            # get acquisition independent features
            good_features = [f for f in features if f not in blacklist]
            
            # handle formatting
            if 0 in harmonized_data.columns:
                harmonized_data = harmonized_data.set_index(0,drop=True)
            
            # make data file for only acquisition-independent features
            good_harmonized_data=harmonized_data[good_features+['group','cancer']]
            #print(good_harmonized_data)
            good_harmonized_data.to_csv(out_path+tag+'_cv'+str(i)+'_usable_harmonized.csv')
            print('Usable data saved to csv')
            
# %% Classification 
            final_test_group = list(harmonized_test_data.index)
            total_scans = list(good_harmonized_data.index)   
            
            good_harmonized_data = good_harmonized_data.reset_index()
            
            # convert training data to numpy array
            Xf = good_harmonized_data[good_features].apply(pd.to_numeric).to_numpy()
            y = good_harmonized_data['cancer'].apply(pd.to_numeric).to_numpy()
            
            harmonized_test_data.reset_index(inplace=True)
            # convert test set to numpy array
            X_test = harmonized_test_data[good_features].apply(pd.to_numeric).to_numpy()
            y_test = harmonized_test_data['cancer'].apply(pd.to_numeric).to_numpy()
            
            # scale features before using SVM
            scaler = StandardScaler()
            # scale training set
            Xf = scaler.fit_transform(Xf)
            # scale test set
            X_test = scaler.transform(X_test)
            
            if hp_tune:
                # initialize SVC for classification
                svm_clf = LinearSVC(penalty='l1', dual=False, class_weight='balanced', random_state=42)
                # hyperparameter tuning
                param_grid = {'C': np.logspace(-3, 2, 10)}
                
                pr_auc_scorer = make_scorer(average_precision_score)

                search_clf = GridSearchCV(estimator=svm_clf, param_grid=param_grid, cv=5, scoring=pr_auc_scorer)
                search_clf.fit(Xf, y)
                
                C_clf = search_clf.best_params_['C']
                print(f"Best C for classification: {C_clf}")
            else:
                C_clf = default_C
            
            # initialize SVM for classification
            svm_clf = LinearSVC(penalty='l1', C = C_clf, dual=False, class_weight='balanced', random_state=42) 
            # train SVM
            svm_clf.fit(Xf,y)
            # predict
            y_scores = svm_clf.decision_function(X_test)
            # calculate precision and recall
            precision, recall, _ = precision_recall_curve(y_test, y_scores)
            # feature selection 
            metrics_df['C_clf'].append(C_clf)
            sel = SelectFromModel(svm_clf, prefit=True)
            # Extract the selected feature indices
            sel_ind = sel.get_support(indices=True)
            sel_f = [good_features[i] for i in sel_ind]
            # save data
            metrics_df['num_sel'].append(len(sel_f))
            metrics_df['sel_f'].append(sel_f)
            # classify test set
            print('Predicting')
            y_pred = svm_clf.predict(X_test)
 
        # %% metrics
        
            g_accuracy = accuracy_score(y_test,y_pred)
            # add accuracy metric to data sheet
            metrics_df['test_accuracy'].append(g_accuracy)
            
            g_AUC = roc_auc_score(y_test, y_scores)
            g_pr_AUC = auc(recall, precision)
            print(g_pr_AUC)

            fpr, tpr, thresholds = roc_curve(y_test, y_scores)

            spec_90sens = 1 - np.interp(0.91, tpr, fpr)
            sens_30spec = np.interp((1-0.3), fpr, tpr)
            sens_50spec = np.interp((1-0.5), fpr, tpr)
            sens_85spec = np.interp((1-0.85), fpr, tpr)
            sens_90spec = np.interp((1-0.9), fpr, tpr)
            
            g_tn, g_fp, g_fn, g_tp = confusion_matrix(y_test, y_pred).ravel()
            # false positive rate
            g_fpr = g_fp/(g_fp+g_tn)
            # false negative rate
            g_fnr = g_fn/(g_fn+g_tp)
            
            # compute weighted accuracy
            weights = compute_sample_weight(class_weight='balanced', y=y_test)
            g_weighted_accuracy = accuracy_score(y_test,y_pred, sample_weight=weights)
            
            manufacturer = headers.loc[final_test_group,'Manufacturer']
            manufacturer = manufacturer.reset_index()
            
            
# %% # per subgroup performance
            
            GE_ind = manufacturer.loc[manufacturer['Manufacturer'] == 'GE MEDICAL SYSTEMS'].index
            SIEMENS_ind = manufacturer.loc[manufacturer['Manufacturer']=='SIEMENS'].index
            
            CE = headers.loc[final_test_group,'CE']
            CE = CE.reset_index()
            
            CE_ind = CE.loc[CE['CE']==1].index
            nonCE_ind = CE.loc[CE['CE']==0].index
            
            try:
                CE_AUC = roc_auc_score(y_test[CE_ind], y_scores[CE_ind])
                CE_precision, CE_recall, _ = precision_recall_curve(y_test[CE_ind], y_scores[CE_ind])
                CE_pr_AUC = auc(CE_recall, CE_precision)
            except ValueError:
                CE_AUC = np.nan
                CE_pr_AUC = np.nan
            try:
                nonCE_AUC = roc_auc_score(y_test[nonCE_ind], y_scores[nonCE_ind])
                nonCE_precision, nonCE_recall, _ = precision_recall_curve(y_test[nonCE_ind], y_scores[nonCE_ind])
                nonCE_pr_AUC = auc(nonCE_recall, nonCE_precision)
            except ValueError:
                nonCE_AUC = np.nan
                nonCE_pr_AUC = np.nan
            try:
                GE_AUC = roc_auc_score(y_test[GE_ind], y_scores[GE_ind])
                GE_precision, GE_recall, _ = precision_recall_curve(y_test[GE_ind], y_scores[GE_ind])
                GE_pr_AUC = auc(GE_recall, GE_precision)
            except:
                GE_AUC = np.nan
                GE_pr_AUC = np.nan
            
            try:
                SIEMENS_AUC = roc_auc_score(y_test[SIEMENS_ind], y_scores[SIEMENS_ind])
                SIEMENS_precision, SIEMENS_recall, _ = precision_recall_curve(y_test[SIEMENS_ind], y_scores[SIEMENS_ind])
                SIEMENS_pr_AUC = auc(SIEMENS_recall, SIEMENS_precision)
            except:
                SIEMENS_AUC = np.nan
                SIEMENS_pr_AUC = np.nan
                
# %% add metrics to data sheet

            metrics_df['num_usable_features'].append(len(good_features))
            
            metrics_df['test_AUC_CE'].append(CE_AUC)
            metrics_df['test_AUC_nonCE'].append(nonCE_AUC)
            metrics_df['test_AUC_GE'].append(GE_AUC)
            metrics_df['test_AUC_SIEMENS'].append(SIEMENS_AUC)
            metrics_df['test_pr_AUC_CE'].append(CE_pr_AUC)
            metrics_df['test_pr_AUC_nonCE'].append(nonCE_pr_AUC)
            metrics_df['test_pr_AUC_GE'].append(GE_pr_AUC)
            metrics_df['test_pr_AUC_SIEMENS'].append(SIEMENS_pr_AUC)
            metrics_df['spec_90sens'].append(spec_90sens)
            metrics_df['sens_30spec'].append(sens_30spec)
            metrics_df['sens_50spec'].append(sens_50spec)
            metrics_df['sens_85spec'].append(sens_85spec)
            metrics_df['sens_90spec'].append(sens_90spec)
            
            metrics_df['test_sensitivity'].append(1-g_fnr)
            metrics_df['test_specificity'].append(1-g_fpr)
            metrics_df['test_AUC'].append(g_AUC)
            metrics_df['test_pr_AUC'].append(g_pr_AUC)
            metrics_df['test_precision'].append(precision_score(y_test, y_pred))
            metrics_df['test_recall'].append(recall_score(y_test, y_pred))
            metrics_df['test_f1'].append(f1_score(y_test, y_pred))
            metrics_df['test_weighted_accuracy'].append(g_weighted_accuracy)
            
            metrics_df['test_N'].append(len(final_test_group))
            metrics_df['test_PN_N'].append(len(set([n.split('PRE')[0] for n in final_test_group])))
            metrics_df['test_patient_N'].append(len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in final_test_group])))
            metrics_df['test_prevalence'].append(np.mean(y_test))
            
            metrics_df['train_N'].append(len(total_scans))
            metrics_df['train_PN_N'].append(len(set([n.split('PRE')[0] for n in total_scans])))
            metrics_df['train_patient_N'] = len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in total_scans]))
            metrics_df['train_prevalence'].append(np.mean(y))
            
            update_metrics(metrics_df,i,tag+out_suffix)
            if not aug:
                break
        i += 1