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

# folder where your data are saved
data_path = 'C:/Users/cjhuc/OneDrive/Lung radiomics/pipeline3_7-14_corrected/' 
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
    headers - dataframe containing header data, expects index = scans, columns = acquisition parameters
    feature_data - dataframe containing feature_data, expects index = features, columns = scans
    batch_list - list of the acquisition parameters (columns in headers) that will be harmonized
    blacklist - scans to exclude (filled in recursive loop)
    blacklist_df - why scans are excluded (filled in recursive loop)
outputs:
    headers - updated dataframe containing only headers of harmonizable training scans
    feature_data - updated dataframe containing only data for harmonizable training scans
    blacklist_df - scans that were excluded and why
    batch_info - dataframe with non-uniform acquisition parameters of interest, the specific protocols contained in this training set, number of scans with each protocol
    a list of uniform acquisition parameters (do not need harmonization)
"""

def find_batches(headers, feature_data, batch_list, blacklist=[], blacklist_df={}):
    # setup
    batches = []
    batch_info = []
    temp_blacklist = []
    temp_blacklist_df = {}
    uniform = []
    # loop through chosen parameters
    for col in batch_list:
        #print(col)
        # what are the specific protocols for this parameter in this data?
        vals = pd.unique(headers[col]) 
        # set up
        temp_batch_info = []
        count_usable_batches = 0
        # loop through specific protocols
        for val in vals:
            # how many scans with this protocol?
            count = (headers[col] == val).sum()
            if count < 3:
                #print('Harmonization will fail for',col,val)
                continue # scans in this group will be excluded since OPNCB fails for less than 3 scans
            else:
                # scans in this group will move on to harmonization
                count_usable_batches += 1
                batches.append(val) # members of groups with more than 2 members will not be excluded
                temp_batch_info.append((col,val,count)) # wait to add to the harmonization information
        if count_usable_batches < 2:
            # no variability in this parameter after exclusion
            print(col,'is uniform')
            uniform.append(col) # nothing to harmonize
        else:
            batch_info += temp_batch_info # only add the batch info to the harmonization sheet if there is something to be harmonized
            
    # exclude scans with rare parameters
    for i in headers.index:
        for col in batch_list: # for each parameter
            if headers.loc[i,col] not in batches: # check if scan was a member of a usable batch
                #print(i,'failed',col)
                temp_blacklist.append(i) 
                temp_blacklist_df[i]=col # saves exclusion information for later reference
    headers.drop(index=temp_blacklist, inplace =True) # exclude scan from header data
    feature_data.drop(columns=temp_blacklist, inplace = True) # exclude scan from feature data
    
    # update exclusion information to be passed to the next iteration
    blacklist += temp_blacklist
    blacklist_df.update(temp_blacklist_df)
    if len(temp_blacklist)>0:
        # repeat the process to ensure that excluding patients didn't mess up other parameter batches
        return find_batches(headers,feature_data,batch_list, blacklist=blacklist,blacklist_df=blacklist_df) 
    else:
        # if the last repeat didn't change anything, batch selection and exclusion list is finalized
        print('Found batches')
        return headers, feature_data, blacklist_df, batch_info, list(set(uniform))
    
"""
prep_OPNCB

Automates batch selection process for the training set. 
Also requires the output of dicom_header.py with manually added contrast info, containing the headers 
    of at least the scans being harmonized (code will disregard any additional information)
inputs:
    data_path: path to folder containing input csv files
    features_file: str, name of file containing radiomic feature data (output of get_features.py)
    headers_file: str, name of file containing header info (output of dicom_header.py with added contrast info)
    groupct_file: str, name of file to which to write final batch information (end with .csv)
    output_file: str, name of file to which to write harmonized data (end with .csv)
"""
def prep_OPNCB(data_path, out_path, features_path, headers_path, batch_list, tag):
    
    # gets training data
    feature_data = pd.read_csv(features_path, index_col=0)
    headers = pd.read_csv(headers_path)[['Timestep','group']+batch_list]
    
    # standardizes 
    feature_data.index = feature_data.index.str.upper() 
    headers['Timestep'] = [scan.upper() for scan in headers['Timestep']]
    
    feature_data = feature_data.T
    
    # removes scans not in this training set
    ctr = 0
    for i in range(len(headers)):
        if headers.loc[i,'Timestep'] not in feature_data:
            headers.drop(i, axis='index', inplace=True)
            ctr += 1
    #print('dropping',ctr)
    for i in feature_data.columns:
        if i not in list(headers.Timestep):
            print('Excluded from dataset:',i)
            feature_data.drop(i, axis='columns', inplace=True)

    headers.set_index('Timestep', inplace=True)
    
    # finds scans to exclude (too few samples to be part of a batch)
    print('Finding batches')
    headers, feature_data, blacklist_df, batch_info, uniform = find_batches(headers, feature_data, batch_list)
    ex = len(blacklist_df.keys())
    print(ex,'patients excluded')
    
    batch_list = [col for col in batch_list if col not in uniform]
    
    batches_info = pd.DataFrame(data=batch_info, columns = ['Acquisition Parameter','Batch','Patients'])
    print(batches_info)
    batches_info.to_csv(out_path+tag+'_batch_info.csv')
    print('Harmonization parameters saved to csv')
    
    blacklist_df = pd.DataFrame.from_dict(blacklist_df, orient = 'index')
    #print(blacklist_df)
    blacklist_df.to_csv(out_path+tag+'_excluded.csv')
    print('Excluded scans saved to csv')
    
    # formats headers
    headers.sort_values('Timestep', inplace=True) # alphabetizes
    headers.reset_index(inplace=True)
    headers_p = headers.Timestep
    headers.drop('Timestep', axis='columns', inplace=True)
    
    # formats feature data
    feature_data.sort_index(axis='columns', inplace=True) #alphabetizes
    scans = list(feature_data.columns)
    group_col = list(feature_data.loc['group',:])
    cancer_col = list(feature_data.loc['cancer',:])
    #vols = get_vol_col(feature_data,scans)
    feature_data = feature_data.set_axis(range(len(scans)), axis ='columns', copy=False)
    
    #uses only radiomic features, not diagnostics
    dat_list = []
    for feature in features:
        dat_list.append(feature_data.loc[feature,:])
    dat = pd.DataFrame(dat_list)
    
    # label encodes batch data
    le_dict ={}
    for covar in headers.columns:
        col = list(headers.loc[:,covar])
        try:
            if (type(col[0])==str):
                le = LabelEncoder()
                le=le.fit(col)
                le_dict[covar]=le
                headers[covar]=le.transform(col)
        except:
            print('No data remains to harmonize.')
            print('\a')
            return 0
    if len(uniform)==4:
        print('Data is uniform after exclusion, no data to harmonize')
        write_df = pd.concat([pd.DataFrame(scans), feature_data.T], axis=1)
        write_df.to_csv(out_path+tag+'_uniform.csv', index=False)
        return 0
    
    print(ex, 'patients excluded:', list(set(list(blacklist_df.index))))
    
    return dat, scans, le_dict, group_col, cancer_col, headers, headers_p
    
def run_OPNCB(dat, scans, out_path, group_col, cancer_col, headers, headers_p, batch_list, tag, multi_group=False):
    preserve = []
    # use a covariate
    if multi_group:
        preserve = ['group']
                  
    print('Harmonizing')
    output_df, final_estimates = nested.OPNestedComBat(dat, headers, batch_list, data_path, categorical_cols = preserve, return_estimates=True)
    shutil.move(data_path+'order.txt',out_path+tag+'_order.txt')

    write_df = pd.concat([pd.DataFrame(scans), output_df], axis=1) # write results fo file
    write_df.insert(1, 'group', group_col)
    write_df.insert(2,'cancer',cancer_col)
    write_df.to_csv(out_path+tag+'_harmonized.csv', index=False)
    print('Harmonized data saved to csv')

    return write_df, final_estimates

"""
get_batchinfo

groupct_file: str, csv file containing batch info from harmonization (created by run_OpNCB)
returns: list of parameters used in harmonization
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

Performs the kruskal-wallis test and determines which features must be excluded 
due to unsuccessful harmonization

group_nm: str, short name of group you are testing
groupct_file: str, csv file containing batch info from harmonization (created by run_OpNCB)
harm_file: str, csv file containing harmonized data (created by run_OpNCB)
headers_file: str, name of file containing header info (output of dicom_header.py)

returns: 
    csv file containing numerical results of kruskal-wallis test on all batch groups
    csv file containing list of features with a significant kruskal-wallis test
"""
def kruskal_wallis(group_nm, out_path, groupct_file, harm_path, headers_path, tag = '', publish=True):
    # read in feature data
    harm_data = pd.read_csv(harm_path, index_col = 0)
    # make sure only data from the desired group is included in the test
    harm_data = harm_data.groupby(['group']).get_group((group_nm,))
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
    # test all the chosen parameters with variability
    for p in params:
        # set up
        exclude = pd.DataFrame(columns = ['feature', 'parameter'])
        exclude.set_index('feature', inplace = True, drop = True)
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
        for feature in features:
            # get feature distributions grouped by a specific protocol
            arrays = [np.asarray(batches[var][feature], dtype=object) for var in batches]
            # compare all the distributions
            H, pval = scipy.stats.kruskal(*arrays, nan_policy='raise')
            # save p-val
            df.loc[feature, p+' p'] = pval
            if pval<=0.05:
                # this feature is dependent on specific protocols when considering this parameter
                if feature not in blacklist:
                    blacklist.append(feature)
                             
        # records which features are dependent overall for this parameter
        for feature in df.index:
            for col in df.columns:
                p = col.split(' p')[0]
                pval = df.loc[feature, col]
                if pval<=0.05:
                    # records which parameters a given feature is dependent on
                    if not feature in exclude.index:
                        exclude.loc[feature, 'parameter']=p
                    else:
                        if not exclude.loc[feature, 'parameter'].__contains__(p):
                            exclude.loc[feature, 'parameter'] += ', '+p
    if publish:
        df.to_csv(out_path+tag+'_kruskal_wallis.csv')
        print('Results of Kruskal-Wallis test saved to csv')
        exclude.to_csv(out_path+tag+'_kw_badfeat.csv')
        print('Features with a significant Kruskal-Wallis test saved to csv')
    return blacklist

"""
process_test_set

applies the estimators trained in run_OPNCB to the test set
relies on the *order.txt file output from run_OPNCB being in directory out_path

inputs:
    test_set - the dataframe containing the feature data, with index = scans, columns = features
    final_test_group - the scans to be included in the test set
    test_headers - the dataframe containing the header data (samples not in final_test_group will be excluded)
    out_path - the directory where data is being read and written
    batch_list - parameters being controlled for in harmonization
    estimates - the trained estimators, output from run_OPNCB
    test_tag - included in file names
    train_tag - included in file names
    le_dict - label encoding used for training set, output of run_OPNCB
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
        dat = output['data']
    output_test = dat.T
    # put back the non-numerical data
    output_test['cancer']=y
    output_test['group']=g
    # save
    output_test.to_csv(out_path+test_tag+'_harmonized.csv')
    #print(final_test_group)
    return output_test