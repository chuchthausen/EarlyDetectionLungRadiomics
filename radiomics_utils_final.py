#radiomics_utils 
#author: Claire Huchthausen
#2023

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import OPNestedComBat as nested
import numpy as np
import scipy
import shutil
from neuroCombat import neuroCombatFromTraining
from statsmodels.stats.multitest import multipletests
import os
from scipy import stats
import sys

# folder where your data are saved
data_path = 'C:/Users/cjhuc/OneDrive/Lung radiomics/Reduced-noise_reanalysis/'

# folder where your graphs are saved
graphs_path = 'C:/Users/cjhuc/OneDrive/Lung radiomics/pipeline3_graphs/' 

features = ['original_shape_Elongation',
'original_shape_Flatness',
'original_shape_LeastAxisLength',
'original_shape_MajorAxisLength',
'original_shape_Maximum2DDiameterColumn',
'original_shape_Maximum2DDiameterRow',
'original_shape_Maximum2DDiameterSlice',
'original_shape_Maximum3DDiameter',
'original_shape_MeshVolume',
'original_shape_MinorAxisLength',
'original_shape_Sphericity',
'original_shape_SurfaceArea',
'original_shape_SurfaceVolumeRatio',
'original_shape_VoxelVolume',
'original_firstorder_10Percentile',
'original_firstorder_90Percentile',
'original_firstorder_Energy',
'original_firstorder_Entropy',
'original_firstorder_InterquartileRange',
'original_firstorder_Kurtosis',
'original_firstorder_Maximum',
'original_firstorder_MeanAbsoluteDeviation',
'original_firstorder_Mean',
'original_firstorder_Median',
'original_firstorder_Minimum',
'original_firstorder_Range',
'original_firstorder_RobustMeanAbsoluteDeviation',
'original_firstorder_RootMeanSquared',
'original_firstorder_Skewness',
'original_firstorder_TotalEnergy',
'original_firstorder_Uniformity',
'original_firstorder_Variance',
'original_glcm_Autocorrelation',
'original_glcm_ClusterProminence',
'original_glcm_ClusterShade',
'original_glcm_ClusterTendency',
'original_glcm_Contrast',
'original_glcm_Correlation',
'original_glcm_DifferenceAverage',
'original_glcm_DifferenceEntropy',
'original_glcm_DifferenceVariance',
'original_glcm_Id',
'original_glcm_Idm',
'original_glcm_Idmn',
'original_glcm_Idn',
'original_glcm_Imc1',
'original_glcm_Imc2',
'original_glcm_InverseVariance',
'original_glcm_JointAverage',
'original_glcm_JointEnergy',
'original_glcm_JointEntropy',
'original_glcm_MCC',
'original_glcm_MaximumProbability',
'original_glcm_SumAverage',
'original_glcm_SumEntropy',
'original_glcm_SumSquares',
'original_gldm_DependenceEntropy',
'original_gldm_DependenceNonUniformity',
'original_gldm_DependenceNonUniformityNormalized',
'original_gldm_DependenceVariance',
'original_gldm_GrayLevelNonUniformity',
'original_gldm_GrayLevelVariance',
'original_gldm_HighGrayLevelEmphasis',
'original_gldm_LargeDependenceEmphasis',
'original_gldm_LargeDependenceHighGrayLevelEmphasis',
'original_gldm_LargeDependenceLowGrayLevelEmphasis',
'original_gldm_LowGrayLevelEmphasis',
'original_gldm_SmallDependenceEmphasis',
'original_gldm_SmallDependenceHighGrayLevelEmphasis',
'original_gldm_SmallDependenceLowGrayLevelEmphasis',
'original_glrlm_GrayLevelNonUniformity',
'original_glrlm_GrayLevelNonUniformityNormalized',
'original_glrlm_GrayLevelVariance',
'original_glrlm_HighGrayLevelRunEmphasis',
'original_glrlm_LongRunEmphasis',
'original_glrlm_LongRunHighGrayLevelEmphasis',
'original_glrlm_LongRunLowGrayLevelEmphasis',
'original_glrlm_LowGrayLevelRunEmphasis',
'original_glrlm_RunEntropy',
'original_glrlm_RunLengthNonUniformity',
'original_glrlm_RunLengthNonUniformityNormalized',
'original_glrlm_RunPercentage',
'original_glrlm_RunVariance',
'original_glrlm_ShortRunEmphasis',
'original_glrlm_ShortRunHighGrayLevelEmphasis',
'original_glrlm_ShortRunLowGrayLevelEmphasis',
'original_glszm_GrayLevelNonUniformity',
'original_glszm_GrayLevelNonUniformityNormalized',
'original_glszm_GrayLevelVariance',
'original_glszm_HighGrayLevelZoneEmphasis',
'original_glszm_LargeAreaEmphasis',
'original_glszm_LargeAreaHighGrayLevelEmphasis',
'original_glszm_LargeAreaLowGrayLevelEmphasis',
'original_glszm_LowGrayLevelZoneEmphasis',
'original_glszm_SizeZoneNonUniformity',
'original_glszm_SizeZoneNonUniformityNormalized',
'original_glszm_SmallAreaEmphasis',
'original_glszm_SmallAreaHighGrayLevelEmphasis',
'original_glszm_SmallAreaLowGrayLevelEmphasis',
'original_glszm_ZoneEntropy',
'original_glszm_ZonePercentage',
'original_glszm_ZoneVariance',
'original_ngtdm_Busyness',
'original_ngtdm_Coarseness',
'original_ngtdm_Complexity',
'original_ngtdm_Contrast',
'original_ngtdm_Strength']

nice_features = ['Shape Elongation',
'Shape Flatness',
'Shape Least Axis Length',
'Shape Major Axis Length',
'Shape Maximum 2D Diameter Column',
'Shape Maximum 2D Diameter Row',
'Shape Maximum 2D Diameter Slice',
'Shape Maximum 3D Diameter',
'Shape Mesh Volume',
'Shape Minor Axis Length',
'Shape Sphericity',
'Shape Surface Area',
'Shape Surface Volume Ratio',
'Shape Voxel Volume',
'First Order 10 Percentile',
'First Order 90 Percentile',
'First Order Energy',
'First Order Entropy',
'First Order Interquartile Range',
'First Order Kurtosis',
'First Order Maximum',
'First Order Mean',
'First Order Mean Absolute Deviation',
'First Order Median',
'First Order Minimum',
'First Order Range',
'First Order Robust Mean Absolute Deviation',
'First Order Root Mean Squared',
'First Order Skewness',
'First Order Total Energy',
'First Order Uniformity',
'First Order Variance',
'GLCM Autocorrelation',
'GLCM Cluster Prominence',
'GLCM Cluster Shade',
'GLCM Cluster Tendency',
'GLCM Contrast',
'GLCM Correlation',
'GLCM Difference Average',
'GLCM Difference Entropy',
'GLCM Difference Variance',
'GLCM Inverse Difference',
'GLCM Inverse Difference Moment',
'GLCM Inverse Difference Moment Normalized',
'GLCM Inverse Difference Normalized',
'GLCM Informational Measure of Correlation 1',
'GLCM Informational Measure of Correlation 2',
'GLCM Inverse Variance',
'GLCM Joint Average',
'GLCM Joint Energy',
'GLCM Joint Entropy',
'GLCM Maximum Correlation Correlation',
'GLCM Maximum Probability',
'GLCM Sum Average',
'GLCM Sum Entropy',
'GLCM Sum Squares',
'GLDM Dependence Entropy',
'GLDM Dependence Non-uniformity',
'GLDM Dependence Non-uniformity Normalized',
'GLDM Dependence Variance',
'GLDM Gray Level Non-uniformity',
'GLDM Gray Level Variance',
'GLDM High Gray Level Emphasis',
'GLDM Large Dependence Emphasis',
'GLDM Large Dependence High Gray Level Emphasis',
'GLDM Large Dependence Low Gray Level Emphasis',
'GLDM Low Gray Level Emphasis',
'GLDM Small Dependence Emphasis',
'GLDM Small Dependence High Gray Level Emphasis',
'GLDM Small Dependence Low Gray Level Emphasis',
'GLRLM Gray Level Non-uniformity',
'GLRLM Gray Level Non-uniformity Normalized',
'GLRLM Gray Level Variance',
'GLRLM High Gray Level Run Emphasis',
'GLRLM Long Run Emphasis',
'GLRLM Long Run High Gray Level Emphasis',
'GLRLM Long Run Low Gray Level Emphasis',
'GLRLM Low Gray Level Run Emphasis',
'GLRLM Run Entropy',
'GLRLM Run Length Non-uniformity',
'GLRLM Run Length Non-uniformity Normalized',
'GLRLM Run Percentage',
'GLRLM Run Variance',
'GLRLM Short Run Emphasis',
'GLRLM Short Run High Gray Level Emphasis',
'GLRLM Short Run Low Gray Level Emphasis',
'GLSZM Gray Level Non-uniformity',
'GLSZM Gray Level Non-uniformity Normalized',
'GLSZM Gray Level Variance',
'GLSZM High Gray Level Zone Emphasis',
'GLSZM Large Area Emphasis',
'GLSZM Large Area High Gray Level Emphasis',
'GLSZM Large Area Low Gray Level Emphasis',
'GLSZM Low Gray Level Zone Emphasis',
'GLSZM Size Zone Non-uniformity',
'GLSZM Size Zone Non-uniformity Normalized',
'GLSZM Small Area Emphasis',
'GLSZM Small Area High Gray Level Emphasis',
'GLSZM Small Area Low Gray Level Emphasis',
'GLSZM Zone Entropy',
'GLSZM Zone Percentage',
'GLSZM Zone Variance',
'NGTDM Busyness',
'NGTDM Coarseness',
'NGTDM Complexity',
'NGTDM Contrast',
'NGTDM Strength']

"""
find_batches

inputs:
    headers - dataframe, containing header data, expects index = scans, columns = acquisition parameters
    feature_data - dataframe, containing feature_data, expects index = features, columns = scans
    batch_list - list, the acquisition parameters (columns in headers) that will be harmonized
    blacklist - list, scans to exclude (filled in recursive loop)
    blacklist_df - dataframe, why scans are excluded (filled in recursive loop)
outputs:
    headers - updated dataframe containing only headers of harmonizable training scans
    feature_data - updated dataframe containing only data for harmonizable training scans
    blacklist_df - dataframe of scans that were excluded and why
    batch_info - dataframe with non-uniform acquisition parameters of interest, the specific protocols contained in this training set, number of scans with each protocol
    a list of uniform acquisition parameters (do not need harmonization)
"""

def find_batches(headers, feature_data, batch_list, blacklist=None, blacklist_df=None):
    import pandas as pd

    if blacklist is None:
        blacklist = []
    if blacklist_df is None:
        blacklist_df = {}

    headers = headers.copy()
    feature_data = feature_data.copy()

    while True:
        batches_info = []  # list of (parameter, value, count)
        uniform = []
        temp_blacklist = {}
        any_excluded = False # reset flag - function will iterate if True

        #Identify valid batches and uniform parameters
        for col in batch_list:
            # get instances of an acquisition parameter
            vals = pd.unique(headers[col])
            # setup
            param_batches = []
            count_usable_batches = 0
            # loop through instances
            for val in vals:
                # how many times does it appear?
                count = (headers[col] == val).sum()
                # ComBat works for >3 scans
                if count >= 3:
                    param_batches.append(val) # so this might be a valid batch, save for later reference
                    batches_info.append((col, val, count))
                    count_usable_batches += 1 # only valid before scan exclusion occurs
            if count_usable_batches < 2: # only one instance
                uniform.append(col)

            # Mark scans with rare values for exclusion
            for idx in headers.index:
                if headers.loc[idx, col] not in param_batches:
                    temp_blacklist[idx] = col
                    any_excluded = True # any change in this iteration?

        # If no scans were excluded in this iteration, stop
        if not any_excluded:
            break

        # Drop the scans excluded in this iteration
        drop_indices = list(temp_blacklist.keys())
        headers.drop(index=drop_indices, inplace=True)
        feature_data.drop(columns=drop_indices, inplace=True)

        # Update cumulative blacklist
        blacklist.extend(drop_indices)
        blacklist_df.update(temp_blacklist)
        # Now go back to the beginning of the loop
        # repeat the process since excluding patients might mess up other parameter batches

    return headers, feature_data, blacklist_df, batches_info, uniform

def CI_poisson(data, confidence_level=0.95):
    data = [d for d in data if not np.isnan(d)]
    x = np.sum(data)         # total number of significant results
    n = len(data)            # number of trials
    mean = x / n             # mean number per trial
    alpha = 1 - confidence_level

    # Exact Poisson CI for the total count
    lower = 0.5 * stats.chi2.ppf(alpha / 2, 2 * x)
    upper = 0.5 * stats.chi2.ppf(1 - alpha / 2, 2 * (x + 1))

    # Convert to per-trial mean CI
    ci_lower = lower / n
    ci_upper = upper / n

    return mean, ci_lower, ci_upper, n

def CI(data, confidence_level=0.95):
    mean = np.nanmean(data)
    std_dev = np.std(data, ddof=1)  # Sample standard deviation (use ddof=1 for sample)
    n = len(data.dropna())

    alpha = 1 - confidence_level
    dof = n - 1  # degrees of freedom

    # t-distribution critical value
    t_critical = stats.t.ppf(1 - alpha/2, dof)
    
    # Standard error of the mean
    se = std_dev / np.sqrt(n)
    
    # Margin of error
    margin_of_error = t_critical * se
    
    # Confidence interval
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return mean, ci_lower, ci_upper, n


"""
prep_OPNCB

Automates batch selection process for the training set. 
Also requires the output of dicom_header.py with manually added contrast info, containing the headers 
    of at least the scans being harmonized (code will disregard any additional information)
inputs:
    data_path: str, path to folder containing input csv files
    out_path: str, path to folder where output will be saved
    features_path: str, full path to file containing radiomic feature data (output of get_features.py)
    headers_path: str, full path to file containing header info (output of dicom_header.py with added contrast info)
    batch_list: list of parameters to harmonize over
    tag: str, labels output files
outputs:
    dat - dataframe containing ordered feature data, with scans in the columns (numerical index)
    scans - list, contains the ordered scan labels
    le_dict - dict, contains the labels encoded for specific acquisition protocols
    group_col - list, contains group labels of ordered scans
    cancer_col - list, contains diagnosis labels of ordered scans
    headers - dataframe containing ordered header information for the scans in dat, with scans in the rows (numerical index)
"""
def prep_OPNCB(data_path, out_path, features_path, headers_path, batch_list, tag):

    # Load data and standardize formatting
    feature_data = pd.read_csv(features_path, index_col=0).T
    headers = pd.read_csv(headers_path)[['Timestep'] + batch_list]
    
    headers['Timestep'] = headers['Timestep'].str.upper()
    feature_data.columns = feature_data.columns.str.upper()

    # Keep only scans present in both headers and feature data
    headers = headers[headers['Timestep'].isin(feature_data.columns)].copy()
    feature_data = feature_data.loc[:, headers['Timestep']]

    print(len(headers['Timestep']),len(feature_data.columns))

    headers.set_index('Timestep', inplace=True)

    # Find batches
    headers, feature_data, blacklist_df, batch_info, uniform = find_batches(
        headers, feature_data, batch_list
    )

    print(f"{len(blacklist_df)} patients excluded")

    # Save batch info and exclusions
    if batch_info:
        batches_info = pd.DataFrame(batch_info, columns=['Acquisition Parameter', 'Batch', 'Patients'])
    else:
        batches_info = pd.DataFrame(columns=['Acquisition Parameter', 'Batch', 'Patients'])
    

    pd.DataFrame.from_dict(blacklist_df, orient='index', columns=['Excluded Parameter']).to_csv(
        f"{out_path}{tag}_excluded.csv", index=True
    )

    # Align scans and feature data
    scans = list(headers.index)
    feature_data = feature_data[scans]

    # Extract group and cancer labels
    group_col = list(feature_data.loc['group'])
    cancer_col = list(feature_data.loc['cancer'])

    # Keep only radiomic features
    dat = feature_data.loc[features]

    if len(scans)==0:
        print('Too few scans or all scans excluded')
        print()
        return 0

    # Handle fully uniform case
    if len(uniform) == len(batch_list):
        print("Data is fully uniform; no harmonization needed.")
        write_df = dat.T
        write_df['group']=group_col
        write_df['cancer']=cancer_col
        write_df.to_csv(f"{out_path}{tag}_harmonized.csv")
        batches_info.to_csv(f"{out_path}{tag}_batch_info.csv", index=False)
        print(batches_info)
        return dat, scans, 0, group_col, cancer_col, headers

    # batch info passed to ComBat cannot contain uniform columns
    headers = headers.drop(columns=uniform, errors='ignore')
    # drop uniform acquisition params from batch info too (only one instance per parameter)
    batches_info = batches_info[batches_info['Acquisition Parameter'].duplicated(keep=False)].copy()
    batches_info.to_csv(f"{out_path}{tag}_batch_info.csv", index=False)
    print(batches_info)
    
    # Label encode batch parameters that are not uniform
    le_dict = {}
    for covar in headers.columns:
        if covar not in uniform:
            col_data = headers[covar].astype(str).tolist()
            le = LabelEncoder()
            le.fit(col_data)
            headers[covar] = le.transform(col_data)
            le_dict[covar] = le

    print(f"{len(blacklist_df)} patients excluded: {list(blacklist_df.keys())}")

    # format for the OPNComBat algorithm
    headers = headers.reset_index(drop = True)
    headers = headers[[b for b in batch_list if b in headers.columns]]

    # all outputs are ordered to align with each other
    return dat, scans, le_dict, group_col, cancer_col, headers

"""
run_OPNCB
Runs the optimized nested ComBat function and saves output data appropriately.

inputs:
    dat - dataframe, containing ordered feature data, with scans in the columns (numerical index), output of prep_OPNCB
    scans - list, contains the ordered scan labels, output of prep_OPNCB
    out_path - str, path where data will be saved
    group_col - list, contains group labels of ordered scans, output of prep_OPNCB
    cancer_col - list, contains diagnosis labels of ordered scans, output of prep_OPNCB
    headers - dataframe containing ordered and label-encoded header information for the scans in dat, with scans in the rows (numerical index), output of prep_OPNCB
    batch_list - list, acquisition parameters (appearing in the columns of headers) that will be corrected for in harmonization
    tag - str, label to include in output filenames
    multi_group - bool, set to True if using a categorical covariate for group
outputs:
    write_df - dataframe, harmonized feature data, scans as a column along the index, with columns for group and diagnosis info
    final_estimates - dict, input batches as keys, values are arrays corresponding to the ComBat estimators trained for the protocols
"""

def run_OPNCB(dat, scans, out_path, group_col, cancer_col, headers, batch_list, tag, multi_group=False, publish = True):
    preserve = []
    # use a covariate
    if multi_group:
        preserve = ['group']
    headers['group'] = group_col

    print('Harmonizing')
    output_df, final_estimates = nested.OPNestedComBat(dat, headers, batch_list, data_path, categorical_cols = preserve, return_estimates=True)
    # rename this output file so it won't be overwritten in the next loop
    shutil.move(data_path+'order.txt',out_path+tag+'_order.txt')

    write_df = pd.concat([pd.DataFrame(scans), output_df], axis=1) # write results fo file
    write_df.insert(1, 'group', group_col)
    write_df.insert(2,'cancer',cancer_col)
    if publish:
        write_df.to_csv(out_path+tag+'_harmonized.csv', index=False)
        print('Harmonized data saved to csv')

    return write_df, final_estimates

"""
get_batchinfo

input:
    groupct_file - str, name of csv file containing batch info from harmonization (created by run_OpNCB)
returns: 
    params - dict, parameters with associated protocols to be used in harmonization
"""
def get_batchinfo(groupct_path):
    batch_info = pd.read_csv(groupct_path)
    #print(batch_info)
    batch_info.replace('[1.6, 1.6]', 1.6, inplace = True) # replaces problematic str
    params = {p:[] for p in set(list(batch_info.loc[:,'Acquisition Parameter']))} # gets list of parameters used in harmonization
    for i in batch_info.index:
        p = batch_info.loc[i,'Acquisition Parameter']
        var = batch_info.loc[i,'Batch']
        try:
            var = pd.to_numeric(var)
        except:
            pass
        params[p].append(var)
    return(params)

"""
kruskal_wallis

Performs the kruskal-wallis test and determines which features are acquisition-dependent

inputs:
    group_nm - str, short name of group you are testing in the column 'group' in feature data file
    out_path - str, path where output will be stored
    groupct_file - str, name of csv file containing batch info from harmonization (created by run_OpNCB)
    harm_path - str, full path to csv file containing harmonized data (created by run_OpNCB)
    headers_path - str, full path to file containing header info (output of dicom_header.py)
    tag - str, label to be included in output filenames
    publish - bool, set to true if you want to write output files
    multitest_a - alpha hyperparameter to use in Benjamini-Hochberg multiple testing correction
returns: 
    blacklist - list, names of features that are acquisition-dependent according to this protocol
"""
def kruskal_wallis(group_nm, out_path, groupct_file, harm_path, headers_path, tag = '', publish=True, multitest_a=0.05):
    # read in feature data
    harm_data = pd.read_csv(harm_path, index_col = 0)
    # make sure only data from the desired group is included in the test
    harm_data = harm_data.groupby(['group']).get_group(group_nm)
    headers = pd.read_csv(headers_path)
    # properly formats headers
    headers['Timestep'] = [h.upper() for h in list(headers.Timestep)]
    headers.replace('[1.6, 1.6]', 1.6, inplace = True)
    for h in headers:
        try:
            headers[h] = pd.to_numeric(headers[h])
        except:
            pass
    headers = headers.set_index('Timestep', drop=True)
    
    # reads in batch information
    params = get_batchinfo(groupct_file)
    if all([len(params[k])==1 for k in params.keys()]):
        print('Acquisition parameters are uniform, KW not needed')
        return []
    
    # which scans are used?
    valid =  [] 
    for i in range(len(list(params.values()))):
        valid+= list(params.values())[i]

    # adds acquisition parameters to dataframe
    for p in params:
        # for each scan
        for s in harm_data.index:
            # if header information is missing, don't use this scan
            if s not in headers.index:
                harm_data.drop(s, axis = 'index')
                continue
            # make sure this scan has a protocol that can be harmonized over
            if not headers.loc[s,p] in valid:
                harm_data.drop(s, axis = 'index')
                continue 
            harm_data.loc[s,p]=headers.loc[s,p] 
    harm_data.reset_index(inplace = True, drop = True)

    df = pd.DataFrame()
    df['features'] = features
    df.set_index('features', drop = True, inplace = True)

    # which features are dependent on acquisition?
    blacklist = []
    exclude = pd.DataFrame(columns = ['feature', 'parameter'])
    exclude.set_index('feature', inplace = True, drop = True)
    # test all the chosen parameters with variability
    for p in params:
        # set up
        temp = harm_data.copy()
        batches = {}
        # go through each protocol for this param
        for var in params[p]:
            # get scans with this protocol
            batch_df = temp.loc[temp[p]==var]
            batch_df.reset_index(inplace = True, drop = True)
            if not batch_df.empty:
                # store scans separated by protocol
                batches[var] = batch_df
        if len(batches)==1: # an acquisition param may be uniform within a subgroup
            continue
        # test each feature distribution
        
        feature_pval = []
        for feature in features:
            # get feature distributions grouped by a specific protocol
            arrays = [np.asarray(batches[var][feature], dtype=object) for var in batches]
            # compare all the distributions
            H, pval = scipy.stats.kruskal(*arrays, nan_policy='raise')
            # save p-val
            feature_pval.append(pval)
        
        sig_ind = np.where(np.array(feature_pval) < 0.05)[0]
        
        feature_pval = multipletests(feature_pval, alpha=multitest_a, method='fdr_bh')[1]
        
        if len(sig_ind)>0:
            if len(np.where(np.array(feature_pval[sig_ind]) < 0.05)[0])==0:
                print(group_nm,p)
                print('All values have been corrected into nonsignificance')
            
        # Adding acquisition parameters to the exclusion dataframe
        for i, feature in enumerate(features):
            df.loc[feature, p] = feature_pval[i]
            if feature_pval[i] <= 0.05:
                # Feature is dependent on a specific protocol
                if feature not in blacklist:
                    blacklist.append(feature)
        
                # Add or update the 'parameter' column in the exclude dataframe
                if feature not in exclude.index:
                    exclude.loc[feature, 'parameter'] = p  # Initialize the parameter if it's a new feature
                else:
                    # If the parameter is not already listed, append it
                    current_params = exclude.loc[feature, 'parameter']
                    if p not in current_params:
                        exclude.loc[feature, 'parameter'] = current_params + ', ' + p
    if publish:
        df.to_csv(out_path+tag+'_kruskal_wallis.csv')
        #print('Results of Kruskal-Wallis test saved to csv')
        exclude.to_csv(out_path+tag+'_kw_badfeat.csv')
        #print('Features with a significant Kruskal-Wallis test saved to csv')
    return blacklist

"""
process_test_set

applies the estimators trained in run_OPNCB to the test set
relies on the *order.txt file output from run_OPNCB being in directory out_path

inputs:
    test_set - dataframe, containing the feature data, with index = scans, columns = features
    final_test_group - list, the scans to be included in the test set
    test_headers - dataframe, containing the header data (samples not in final_test_group will be excluded)
    out_path - str, name of the directory where data is being read and written
    batch_list - list, parameters being controlled for in harmonization
    estimates - dict, the trained estimators, output from run_OPNCB
    test_tag - str, included in file names
    train_tag - str, included in file names
    le_dict - dict, label encoding used for training set, output of run_OPNCB
returns:
    output_test - the dataframe containing the test set harmonized by the trained estimators
"""

def process_test_set(test_set, final_test_group, test_headers, out_path, batch_list, estimates, test_tag,train_tag, le_dict):
    # select data and headers for test set
    # also gets correct order
    test_set = test_set.loc[final_test_group]
    test_headers = test_headers.loc[final_test_group]
    
    # save non-numerical data
    y = list(test_set['cancer'])
    g = list(test_set['group'])
    
    test_set = test_set[features] # removes diagnostic features
    test_set = test_set.apply(pd.to_numeric)
    
    test_set.to_csv(out_path+test_tag+'.csv')

    # reads the order in which the training set was harmonized
    permutation = []
    with open(out_path+train_tag+'_order.txt', 'r') as file:
        for line in file:
            permutation.append(line.strip())
    # apply estimators in the same order as the training set
    dat = test_set.copy().T
    for i,param in enumerate(permutation):
        print(param)
        subest = estimates[i]
        test_batch_info = list(test_headers[param])
        if param in le_dict.keys():
            test_batch_info = le_dict[param].transform(test_batch_info) # need to label encode to match training set
        # harmonize
        output = neuroCombatFromTraining(dat,test_batch_info,subest)
        dat = output['data'] # the corrected version will pass to the next iteration
        # to be corrected iteratively
    output_test = dat.T
    # put back the non-numerical data
    output_test['cancer']=y
    output_test['group']=g
    # save
    output_test.to_csv(out_path+test_tag+'_harmonized.csv')
    #print(final_test_group)
    return output_test

def fill_nans(metrics_df):
    tgt_len = 1
    for k in metrics_df.keys():
        if len(metrics_df[k])<tgt_len:
            metrics_df[k].append(np.nan)
    return metrics_df

def update_metrics(metrics_df,i, out_file):
    write_metrics_df = pd.DataFrame(metrics_df)
    write_metrics_df.index = [i]

    if os.path.exists(data_path+out_file):
        # read in existing data
        existing_metrics_df = pd.read_csv(data_path+out_file, index_col=0)
        # add metrics from this trial to existing data
        write_metrics_df = pd.concat([existing_metrics_df, write_metrics_df], axis = 0)
    # save data sheet with metrics from this trial
    write_metrics_df.to_csv(data_path+out_file)
        
def median_IQR(data):
    data = [d for d in data if not np.isnan(d)]
    med = np.median(data)
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    return med, IQR, len(data)