import numpy as np
import pandas as pd
import re
import glob
from scipy.stats import kurtosis, skew, pearsonr
from numpy import nan
from scipy.spatial import distance
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
from sklearn import metrics

def resample_with_replacement(x_idx, X, y):

    # Get array of indices for resampled points
    idx_samples = np.random.choice(x_idx, size=len(x_idx), replace=True)
    # Sample from x and y according to sample_idx
    X_sampled = X[idx_samples, :]
    y_sampled = y[idx_samples]

    return X_sampled, y_sampled

def extract_stat_per_hemis(strm_median_P):
    
    tmp1 = np.mean(strm_median_P, axis=1)
    tmp2 = np.std(strm_median_P, axis=1)
    tmp3 = kurtosis(strm_median_P, axis = 1, fisher = True, bias = True, nan_policy = 'propagate')
    tmp4 = skew(strm_median_P, axis = 1, bias=True)
    feature = np.vstack((tmp1,tmp2,tmp3,tmp4)).T

    return feature

def extract_stat_dif_of_hemis(strm_median_P, strm_median_H):
    
    # consider the normalized dMRI tract profile
    #feature_CST = (strm_median_P-strm_median_H)/(strm_median_P+strm_median_H)
    # consider both ipsilesional and contalesional dMRI tract profile
    feature_CST = (strm_median_P-strm_median_H)
    tmp1 = np.mean(feature_CST, axis=1)
    tmp2 = np.std(feature_CST, axis=1)
    tmp3 = kurtosis(feature_CST, axis = 1, fisher = True, bias = True, nan_policy = 'propagate')
    tmp4 = skew(feature_CST, axis = 1, bias=True)
    feature = np.vstack((tmp1,tmp2,tmp3,tmp4)).T
    
    return feature
    
def histogram_feature(x_profile_all):
    
    x_profile_metric = x_profile_all.reshape(x_profile_all.shape[0],100,5)
    ad_p_stats = extract_stat_per_hemis(x_profile_metric[:,:,0])
    adc_p_stats = extract_stat_per_hemis(x_profile_metric[:,:,1])
    fa_p_stats = extract_stat_per_hemis(x_profile_metric[:,:,2])
    fd_p_stats = extract_stat_per_hemis(x_profile_metric[:,:,3])
    rd_p_stats = extract_stat_per_hemis(x_profile_metric[:,:,4])
    
    return np.hstack((ad_p_stats, 
                      adc_p_stats, 
                      fa_p_stats, 
                      fd_p_stats, 
                      rd_p_stats))

def median_imputation(profile_P_tra, profile_P_tes):
    
    imp_median = SimpleImputer(missing_values=np.nan, strategy='median')
    profile_P_tra_imp = imp_median.fit_transform(profile_P_tra)
    profile_P_tes_imp = imp_median.transform(profile_P_tes)
    X_train_hist = histogram_feature(profile_P_tra_imp)
    X_test_hist = histogram_feature(profile_P_tes_imp)    
    return X_train_hist, X_test_hist

def KNN_imputation(K, profile_P_tr):
    
    imp_knn = KNNImputer(n_neighbors=K)
    profile_P_tra_imp = imp_knn.fit_transform(profile_P_tra)
    profile_P_tes_imp = imp_knn.transform(profile_P_tes)
    X_train_hist = histogram_feature(profile_P_tra_imp)
    X_test_hist = histogram_feature(profile_P_tes_imp)
    return X_train_hist, X_test_hist

def median_profile(data_raw_P):
    # data_raw_P shape: number_of_streamlines*number_of_points_along_tractogram(5000*100)
    return np.median(data_raw_P, axis=0).T
    
def weighted_mean_profile(data_raw_P):
    
    return np.mean(data_raw_P, axis=0).T

def pca_correction(pca, profile):
    # Try to correct pca values
    r, p = pearsonr(pca, profile)
    if r<0:
        pca = -pca
    return pca

def mahalanobis(x=None, mu=None, covmat=None):
    # Compute the Mahalanobis Distance between each row of x and the data  
    # x    : vector or matrix of data with, say, p columns
    # data : ndarray of the distribution from which Mahalanobis distance of each observation of x is to be computed
    # cov  : covariance matrix (p x p) of the distribution. If None, will be computed from data
    if (covmat.all==None):
        covmat = np.cov((x),rowvar=0)
        print('cov shape:', covmat.shape)
    
    x_minus_mu = x - mu
    inv_covmat = np.linalg.inv(covmat)
    left_term = np.dot(x_minus_mu, inv_covmat)
    mahal = np.dot(left_term, x_minus_mu.T)
    
    return mahal

def cal_mahal_profile(data):
    
    dist = np.zeros(shape=(data.shape[0],1))
    
    for obs in range(data.shape[0]):
        
        dist[obs] = mahalanobis(x=data[obs,:], 
                                mu=np.mean(data,axis=0), 
                                covmat=np.cov(data, rowvar=0))
        eplsilon = 0.00000001  # make sure we won't face zero devision
        mahal_profile = ((1/(dist+eplsilon))*data).sum(axis=0) 
    return mahal_profile

def Clustering_streamlines(data):
    # this function will weighing up the streamlines by computing the mahalanobis distance of each
    # node with the mean one/ try to consider representative streamlines
    # data shape = (5000,100,116)
    D = np.zeros(shape=(data.shape[2],data.shape[0])) 
    for i in range(data.shape[2]):
        cov = np.cov((data[:,:,i]),rowvar=0)
        inv_covmat = np.linalg.inv(cov)
        mu = np.mean(data[:,:,i], axis=0)
        for j in range(data.shape[0]):
            D[i,j] = mahalanobis(data[j,:,i],mu,inv_covmat) 
    return D

def mahal_features(adc_data_P,fa_data_P,fd_data_P,rd_data_P,ad_data_P, adc_data_H,fa_data_H,fd_data_H,rd_data_H,ad_data_H):
   
    # Try to create features of all streamline distances from mean profile     
    adc_mahalD_P = Clustering_streamlines(adc_data_P)
    adc_mahalD_H = Clustering_streamlines(adc_data_H)
    
    fa_mahalD_P = Clustering_streamlines(fa_data_P)
    fa_mahalD_H = Clustering_streamlines(fa_data_H)
    
    fd_mahalD_P = Clustering_streamlines(fd_data_P)
    fd_mahalD_H = Clustering_streamlines(fd_data_H)
       
    adc_mahal_P = np.vstack((np.mean(adc_mahalD_P, axis=1),
                              np.std(adc_mahalD_P, axis=1), 
                              kurtosis(adc_mahalD_P, axis=1, fisher=True, bias=True),
                              skew(adc_mahalD_P, axis=1, bias=True)))
    adc_mahal_H = np.vstack((np.mean(adc_mahalD_H, axis=1),
                              np.std(adc_mahalD_H, axis=1), 
                              kurtosis(adc_mahalD_H, axis=1, fisher=True, bias=True),
                              skew(adc_mahalD_H, axis=1, bias=True)))    
    fa_mahal_P = np.vstack((np.mean(fa_mahalD_P,axis=1),
                              np.std(fa_mahalD_P, axis=1), 
                              kurtosis(fa_mahalD_P, axis=1, fisher=True, bias=True),
                              skew(fa_mahalD_P, axis=1, bias=True)))    
    fa_mahal_H = np.vstack((np.mean(fa_mahalD_H, axis=1),
                              np.std(fa_mahalD_H, axis=1), 
                              kurtosis(fa_mahalD_H, axis=1, fisher=True, bias=True),
                              skew(fa_mahalD_H, axis=1, bias=True)))
    fd_mahal_P = np.vstack((np.mean(fd_mahalD_P, axis=1),
                              np.std(fd_mahalD_P, axis=1), 
                              kurtosis(fd_mahalD_P, axis=1, fisher=True, bias=True),
                              skew(fd_mahalD_P, axis=1, bias=True)))
    fd_mahal_H = np.vstack((np.mean(fd_mahalD_H, axis=1),
                              np.std(fd_mahalD_H, axis=1), 
                              kurtosis(fd_mahalD_H, axis=1, fisher=True, bias=True),
                              skew(fd_mahalD_H, axis=1, bias=True)))        
    
    adc = np.abs(adc_mahal_P-adc_mahal_H)
    fa = np.abs(fa_mahal_P-fa_mahal_H)
    fd = np.abs(fd_mahal_P-fd_mahal_H)
    feature_all = np.vstack((fa,adc,fd))
    
    return feature_all, adc_mahalD_P, adc_mahalD_H,fa_mahalD_P, fa_mahalD_H, fd_mahalD_P, fd_mahalD_H 

def length_tract(path):
    
    paths = sorted(glob.glob(path+'*.txt'), key=lambda x:float(re.findall("(\d+)",x)[0]))
    list_of_dfs = [pd.read_csv(path, skip_blank_lines=True, header = None, na_values = ['no info', ';']) for path in paths] 
    length = np.zeros(shape=(116,5000))
    for j in range(0, len(list_of_dfs),1):  
        df = list_of_dfs[j]    
        length[j] = df[0][:]
    return length

def extract_value_H_P(adc_data, fa_data, fd_data, rd_data, ad_data, PathSide, subjN):
    
    adc_data_P = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    fa_data_P = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    fd_data_P = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    rd_data_P = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    ad_data_P = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    
    adc_data_H = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    fa_data_H = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    fd_data_H = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    rd_data_H = np.zeros(shape=(5000,100,subjN), dtype = np.float64)
    ad_data_H = np.zeros(shape=(5000,100,subjN), dtype = np.float64)    
    pl = 0
    pr = 0
    for n in range(0,subjN):  
        # Pathology placed at left hemisphere

        if (PathSide[n]==0):
            adc_data_P[:,:,n] = adc_data[:,:,n*2]  
            #adc_data_PL[pl] = adc_data[:,:,n*2]
            fa_data_P[:,:,n] = fa_data[:,:,n*2]
            fd_data_P[:,:,n] = fd_data[:,:,n*2]
            ad_data_P[:,:,n] = ad_data[:,:,n*2]
            rd_data_P[:,:,n] = rd_data[:,:,n*2]
            #length_P[n,:] = length[n*2,:]
            adc_data_H[:,:,n] = adc_data[:,:,n*2+1]  
            fa_data_H[:,:,n] = fa_data[:,:,n*2+1]
            fd_data_H[:,:,n] = fd_data[:,:,n*2+1]
            ad_data_H[:,:,n] = ad_data[:,:,n*2+1]
            rd_data_H[:,:,n] = rd_data[:,:,n*2+1]            
            #length_H[n,:]= length[n*2+1,:]
            
            pl = pl +1
        # Pathology placed at right hemisphere    
        else:
            adc_data_P[:,:,n] = adc_data[:,:,n*2+1]
            fa_data_P[:,:,n] = fa_data[:,:,n*2+1]
            fd_data_P[:,:,n] = fd_data[:,:,n*2+1]
            ad_data_P[:,:,n] = ad_data[:,:,n*2+1]
            rd_data_P[:,:,n] = rd_data[:,:,n*2+1]
            #length_P[n,:] = length[n*2+1,:]
            
            adc_data_H[:,:,n] = adc_data[:,:,n*2]  
            fa_data_H[:,:,n] = fa_data[:,:,n*2]
            fd_data_H[:,:,n] = fd_data[:,:,n*2]
            ad_data_H[:,:,n] = ad_data[:,:,n*2]
            rd_data_H[:,:,n] = rd_data[:,:,n*2]
            #length_H[n,:] = length[n*2,:]
            pr = pr +1
        
    return adc_data_P,fa_data_P,fd_data_P, ad_data_P, rd_data_P, adc_data_H,fa_data_H,fd_data_H, ad_data_H, rd_data_H

def sub_divide_hemispheres(data,PathSide, subjN):
    P_at_right = PathSide.sum()
    P_at_left = subjN - P_at_right
    left = data[:,:,::2]
    right = data[:,:,1:][:,:,::2]
    sub_data_left = dict()
    sub_data_right = dict()
    sub_data_left['H'] = np.zeros(shape = (5000,100,P_at_right), dtype = np.float64)
    sub_data_left['P'] = np.zeros(shape = (5000,100,P_at_left), dtype = np.float64)
    sub_data_right['H'] = np.zeros(shape = (5000,100,P_at_left), dtype = np.float64)
    sub_data_right['P'] = np.zeros(shape = (5000,100,P_at_right), dtype = np.float64)
    l = 0
    r = 0
    for n in range(0,subjN):  
        # Pathology placed at left hemisphere l=0
        if (PathSide[n]==0):
            sub_data_right['H'][:,:,l] = right[:,:,n]
            sub_data_left['P'][:,:,l] = left[:,:,n]
            l=l+1
            
        else:
            sub_data_left['H'][:,:,r] = left[:,:,n]
            sub_data_right['P'][:,:,r] = right[:,:,n]
            r=r+1      
            
    return sub_data_left, sub_data_right

def loadrawdata_R_new(strr, subN):
    paths = sorted(glob.glob('/Users/boshra/Desktop/Boshra/all_patients/'+strr+'/*.csv'), 
                   key=lambda x:float(re.findall("(\d+)",x)[1]))   
    
    for i in range(subN):
        paths[i*2:i*2+2] = sorted(paths[i*2:i*2+2])
    list_of_dfs = [pd.read_csv(path, skip_blank_lines=True, header = None, na_values = ['no info', ';']) for path in paths]
        
    DATA = np.empty(shape = (5000,100,subN*2), dtype = np.float64)
    DATA[:,:,:] = np.NaN
    for j in range(0,subN*2):  
        #print(j)
        df = list_of_dfs[j]
        #df = df.replace(0,nan)
        df = df.reset_index(drop=True)        
        for i in range(1,len(df[0])):            
            listt = [x for x in df[0][i].split(' ') if x]
            tt = np.asarray(listt, dtype = np.float32)
            DATA[i-1,:,j] = tt
    return DATA

def check_data(data):
    
    for i in range(data.shape[2]):
        print(i,':',len(np.argwhere(np.isnan(data[:,:,i]))))
    return
