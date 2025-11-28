from radiomics_utils_final import prep_OPNCB, run_OPNCB, data_path, process_test_set, kruskal_wallis, features, fill_nans, update_metrics
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, roc_auc_score, confusion_matrix, f1_score, make_scorer, precision_recall_curve, auc, average_precision_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import warnings
import os
import re
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# %% 
''' EDIT '''
# where are the input files located?
data_path = data_path
# file containing unharmonized data
features_file = 'raw_data_no_unfollowed.csv'
# file containing header data
headers_file = 'all_headers.csv'
# suffix of file to write with result summary
# final filenames will begin with harmonization, augmentation, and trial
out_suffix = '_harmonization_L1SVC_metrics.csv'
# names of subgroups
groups = ['wedge','LCS','cancer']
# which acquisition parameters are being harmonized over?
batch_list = ['Manufacturer',
              'FocalSpots',
              'CE',
              'KVP']
# which data type is isolated in a given split?
isolate = 'patient'

# incremental percentage of augmentation dataset to include
# multiples of 5 only (e.g. 5%, 100%)
run_increments = [100]
# tune hyperparameter?
hp_tune = False
# manual input of hyperparameter
default_C = 0.1

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
# gets augmentation dataset
augmentation = features_df.loc[
    (features_df['group'] != 'LCS')
]

# %% Cross-validation setup
# trial number
i = 0

for cv in range(10): # loops through trials
    sgkf = StratifiedGroupKFold(n_splits=5, random_state = cv+10, shuffle = True)
    # set up augmentation dataset split, for ablation test
    # 5-fold CV for this trial
    for X_train_ind, X_test_ind in sgkf.split(LCS[features], list(LCS['cancer']),list(LCS[isolate])):
        print(i)
        # gets training and test sets
        train_set, test_set = LCS.iloc[X_train_ind], LCS.iloc[X_test_ind]
        
        if len(set(test_set['cancer']))!=2:
            print('Only one class in the test set after KFold split. Model cannot be trained')
            with open(data_path+orig_tag+'_'+str(i)+"_error.txt", "a") as f:
                f.write("Only one class remains in the test set after KFold split.\n")
            i+=1
            continue
        
# %% Set up training set augmentation
        # make sure a patient that appears in the test set is not in the augmentation data for this trial
        augmentation_i = augmentation.loc[~augmentation[isolate].isin(test_set.index)]
        # initialize splitter
        skf_aug = StratifiedKFold(n_splits=int(100/aug_increment), shuffle=True, random_state=cv+10)
        # store the augmentation splits
        aug_splits = []
        for _, test_index in skf_aug.split(augmentation_i[features], list(augmentation_i['cancer'])):
            split = augmentation_i.iloc[test_index]
            aug_splits.append(split)
        
        for aug_ind in range(20):
            
            # adds a set fraction of augmentation scans to the training set
            train_set = pd.concat([train_set,aug_splits[aug_ind]],axis='index')
            # only calculate predefined augmentation fraction
            if aug_increment*(aug_ind+1) not in run_increments:
                continue
            
# %% Set up results to be saved

            # labels filenames
            tag = orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug'
            # makes output folder for this trial
            out_path = data_path+tag+'_'+str(i)+'/'
            
            # check for what has already been saved
            harmonized= False
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            elif os.path.exists(out_path):
                harmonized=True
            
            # sets up data to be stored from this trial
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
        
            # splits training and test sets into subgroups
            # specific to separate harmonization
            # loops through subgroups
            
            if not harmonized:
                bad_try = 0 # use this to communicate separate harmonization loop errors to augmentation loop
                
                total_scans = []
                gskip = {'LCS':0, 'wedge':0,'cancer':0}
                
                for g in groups:
                    
                    print('Harmonizing',g,'trial',i)
                    # get subgroup from training set of this split
                    g_train_set = train_set.groupby('group').get_group(g)
                    
                    # for use in file names
                    subtag = tag+'_'+g+'_cv'+str(i)
                    train_tag = subtag+'_train'
                    
                    # save training set to csv
                    g_train_set.to_csv(out_path+train_tag+'.csv')
                    
                    # filters and formats training set
                    try:
                        dat, scans, le_dict, group_col, cancer_col, covars = prep_OPNCB(out_path, out_path, out_path+train_tag+'.csv', data_path+headers_file, batch_list, train_tag)
                        
                        if le_dict == 0:
                            # data was uniform after exclusion and did not need to be harmonized
                            gskip[g] = 1
                    except TypeError: # if harmonization will fail
                        bad_try = 1
                        with open(out_path+g+"_error.txt", "a") as f:
                            f.write("Training set with these scans cannot be harmonized.Too few scans or too rare acquisition parameters.\n")
                        break # exit harmonization loop (don't skip harmonizing just one group)
                    
                    total_scans += scans
                    
                    # read in which parameters are present in the training set
                    valid_headers = pd.read_csv(out_path+train_tag+'_batch_info.csv')
                    
                    # %% Get test set
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
                        
                        print(len(list(set(exclude))), 'patients excluded from test set')
                        metrics_df['test_excluded_N'].append(len(list(set(exclude))))
                        
                        # if all the patients were excluded, move on to the next train-test split
                        if len(list(set(exclude)))==len(test_headers.index):
                            print('Incompatible test set. Trying again')
                            # don't harmonize if it's not going to work for the test set
                            bad_try = 1
                            with open(out_path+g+"_error.txt", "a") as f:
                                f.write("Test set parameters are incompatible with training set parameters.\n")
                            
                            metrics_df = fill_nans(metrics_df)
                            update_metrics(metrics_df,i,tag+out_suffix)
                            break 
                
                        # is AUC defined?
                        if len(set(test_set.loc[final_test_group]['cancer']))!=2:
                            print('Only one class remains in the test set after parameter-based exclusion. Model cannot be trained')
                            with open(out_path+"error.txt", "a") as f:
                                f.write("Only one class remains in the test set after parameter-based exclusion.\n")
                            metrics_df = fill_nans(metrics_df)
                            update_metrics(metrics_df,i,tag+out_suffix)
                            bad_try = 1
                            break
                        
                    
                    # %% Apply ComBat
                    # ComBat shouldn't see uniform acquisition parameters
                    inner_batch_list = list(set(list(valid_headers['Acquisition Parameter'])))
                    
                    # if data is not uniform
                    if not gskip[g]:
                        #harmonizes training set
                        harmonized_data, estimates = run_OPNCB(dat, scans, out_path, group_col, cancer_col, covars, inner_batch_list, train_tag, multi_group=False)
                    
                    if g == 'LCS':
                        # applies estimators to test set
                        harmonized_test_data = process_test_set(test_set, final_test_group, test_headers, out_path, inner_batch_list, estimates, test_tag,train_tag,le_dict)
                        
                # combines outputs of runOpNCB for training data
                if bad_try:
                    continue # goes to next increment of augmentation, or the next trial
                    
                harmonized_data = pd.concat([pd.read_csv(out_path+tag+'_'+g+'_cv'+str(i)+'_train_harmonized.csv', index_col=0) for g in groups], axis = 0)
                harmonized_data.sort_index(axis = 'index', inplace = True)
                harmonized_data.to_csv(out_path+tag+'_cv'+str(i)+'_train_harmonized.csv')
                print('Separately harmonized data saved to csv')
                n_ex_train = len(train_set.index)-len(harmonized_data.index)
                metrics_df['train_excluded_N'].append(n_ex_train)
                
# %% Read in pre-harmonized data
            elif harmonized:
                try:
                    harmonized_data = pd.read_csv(out_path+tag+'_cv'+str(i)+'_train_harmonized.csv', index_col=0)
                except FileNotFoundError:
                    print('Training data was not harmonized. Trying again')
                    metrics_df = fill_nans(metrics_df)
                    update_metrics(metrics_df,i,tag+out_suffix)
                    continue
                n_ex_train = len(train_set.index)-len(harmonized_data.index)
                metrics_df['train_excluded_N'].append(n_ex_train)
                try:
                    harmonized_test_data = pd.read_csv(out_path+orig_tag+'_'+str(aug_increment*(aug_ind+1))+'aug_LCS_cv'+str(i)+'_test_harmonized.csv', index_col=0)
                except FileNotFoundError:
                    print('Test data was not harmonized. Trying again')
                    continue
                n_ex_test = len(test_set.index)-len(harmonized_test_data.index)
                metrics_df['test_excluded_N'].append(n_ex_test)
                
# %% Feature acquisition-dependence test

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
            # if there are no usable features
            if len(blacklist)==len(features):
                print('No usable features. Trying again')
                with open(out_path+"error.txt", "a") as f:
                    f.write("No usable features.\n")
                metrics_df = fill_nans(metrics_df)
                update_metrics(metrics_df,i,tag+out_suffix)
            
            # get acquisition independent features
            good_features = [f for f in features if f not in blacklist]
            
            if 0 in harmonized_data.columns:
                harmonized_data = harmonized_data.set_index(0,drop=True)
            
            # make data file for only acquisition-independent features
            good_harmonized_data=harmonized_data[good_features+['group','cancer']]
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
            
            # initialize SVM
            svm_clf = LinearSVC(penalty='l1', C = C_clf, dual=False, class_weight='balanced', random_state=42) 
            # train SVM
            svm_clf.fit(Xf,y)
            
            y_scores = svm_clf.decision_function(X_test)
            
            precision, recall, _ = precision_recall_curve(y_test, y_scores)

            metrics_df['C_clf'].append(C_clf)
            sel = SelectFromModel(svm_clf, prefit=True)
            # Extract the selected feature indices
            sel_ind = sel.get_support(indices=True)
            sel_f = [good_features[i] for i in sel_ind]
            
            if len(sel_f)==0:
                print('No features selected. Cannot train model.')
                with open(out_path+"error.txt", "a") as f:
                    f.write("No features selected.\n")
                metrics_df = fill_nans(metrics_df)
                update_metrics(metrics_df,i,tag+out_suffix)
                continue
            
            metrics_df['num_sel'].append(len(sel_f))
            metrics_df['sel_f'].append(sel_f)

            # classify test set
            print('Predicting')
            y_pred = svm_clf.predict(X_test)
        
        # %% Overall performance
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
            
            
            # %% Performance on subgroups
            
            manufacturer = headers.loc[final_test_group,'Manufacturer']
            manufacturer = manufacturer.reset_index()
            
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
            except ValueError:
                GE_AUC = np.nan
                GE_pr_AUC = np.nan
            
            try:
                SIEMENS_AUC = roc_auc_score(y_test[SIEMENS_ind], y_scores[SIEMENS_ind])
                SIEMENS_precision, SIEMENS_recall, _ = precision_recall_curve(y_test[SIEMENS_ind], y_scores[SIEMENS_ind])
                SIEMENS_pr_AUC = auc(SIEMENS_recall, SIEMENS_precision)
            except ValueError:
                SIEMENS_AUC = np.nan
                SIEMENS_pr_AUC = np.nan
                
            # %% add metrics to data sheet

            metrics_df['num_usable_features'].append(len(good_features))
            
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
            
            metrics_df['test_AUC_CE'].append(CE_AUC)
            metrics_df['test_AUC_nonCE'].append(nonCE_AUC)
            metrics_df['test_AUC_GE'].append(GE_AUC)
            metrics_df['test_AUC_SIEMENS'].append(SIEMENS_AUC)
            metrics_df['test_pr_AUC_CE'].append(CE_pr_AUC)
            metrics_df['test_pr_AUC_nonCE'].append(nonCE_pr_AUC)
            metrics_df['test_pr_AUC_GE'].append(GE_pr_AUC)
            metrics_df['test_pr_AUC_SIEMENS'].append(SIEMENS_pr_AUC)
            
            metrics_df['test_N'].append(len(final_test_group))
            metrics_df['test_PN_N'].append(len(set([n.split('PRE')[0] for n in final_test_group])))
            metrics_df['test_patient_N'].append(len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in final_test_group])))
            metrics_df['test_prevalence'].append(np.mean(y_test))
            
            metrics_df['train_N'].append(len(total_scans))
            metrics_df['train_PN_N'].append(len(set([n.split('PRE')[0] for n in total_scans])))
            metrics_df['train_patient_N'] = len(set([re.sub(r'\d+$', '', n.split('PRE')[0]) for n in total_scans]))
            metrics_df['train_prevalence'].append(np.mean(y))
            
            update_metrics(metrics_df,i,tag+out_suffix)
        i += 1