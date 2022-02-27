import numpy as np
import pandas as pd
from scipy import interp
from utils import loadrawdata_R_new, extract_value_H_P

def read_raw_data(path, pathologic_side, subjN):
    
    ad_data = loadrawdata_R_new(path, 'ad',subjN)
    adc_data = loadrawdata_R_new(path, 'adc',subjN)
    fa_data = loadrawdata_R_new(path, 'fa',subjN)
    fd_data = loadrawdata_R_new(path, 'fd',subjN)
    rd_data = loadrawdata_R_new(path, 'rd',subjN)
    
    return extract_value_H_P(adc_data, fa_data, fd_data,rd_data, ad_data, pathologic_side, subjN)

def profiling(pathologic_side, subjN, path, method='Median'):
    
    adc_data_P,fa_data_P,fd_data_P, ad_data_P, rd_data_P, adc_data_H, fa_data_H,fd_data_H, ad_data_H, rd_data_H = read_raw_data(path, pathologic_side, subjN)
    
    if method == 'Median':
        
        # Median-based profiling
        strm_median_P = np.median(fa_data_P, axis=0).T; 
        strm_median_H = np.median(fa_data_H, axis=0).T; 
        strm_median_P_fd = np.median(fd_data_P, axis=0).T; 
        strm_median_H_fd = np.median(fd_data_H, axis=0).T; 
        strm_median_P_adc = np.median(adc_data_P, axis=0).T; 
        strm_median_H_adc = np.median(adc_data_H, axis=0).T; 
        strm_median_P_rd = np.median(rd_data_P, axis=0).T; 
        strm_median_H_rd = np.median(rd_data_H, axis=0).T;
        strm_median_P_ad = np.median(ad_data_P, axis=0).T; 
        strm_median_H_ad = np.median(ad_data_H, axis=0).T;
        
        return  strm_median_P, strm_median_H, strm_median_P_fd, strm_median_H_fd, strm_median_P_adc, strm_median_H_adc, strm_median_P_rd, strm_median_H_rd, strm_median_P_ad, strm_median_H_ad
    
    elif method == 'Mahalanobis':
        # weighted-mean profiling based on mahalanobis distance
        strm_mahal_P = mahal_profile(fa_data_P)
        strm_mahal_H = mahal_profile(fa_data_H)
        strm_mahal_P_fd = mahal_profile(fd_data_P)
        strm_mahal_H_fd = mahal_profile(fd_data_H)
        strm_mahal_P_adc = mahal_profile(adc_data_P)
        strm_mahal_H_adc = mahal_profile(adc_data_H) 
        strm_mahal_P_rd = mahal_profile(rd_data_P) 
        strm_mahal_H_rd = mahal_profile(rd_data_H)
        strm_mahal_P_ad = mahal_profile(ad_data_P) 
        strm_mahal_H_ad = mahal_profile(ad_data_H)
        
        return  strm_mahal_P, strm_mahal_H, strm_mahal_P_fd, strm_mahal_H_fd, strm_mahal_P_adc, strm_mahal_H_adc, strm_mahal_P_rd, strm_mahal_H_rd, strm_mahal_P_ad, strm_mahal_H_ad
    else:
        raise AttributeError('Please select Median or Mahalanobis')
                
def dataframe_dMRI_profile( strm_P,
                            strm_H,
                            strm_P_fd,
                            strm_H_fd,
                            strm_P_adc,
                            strm_H_adc,
                            strm_P_rd,
                            strm_H_rd,
                            strm_P_ad,
                            strm_H_ad,
                            data):

    new_data_sep = pd.DataFrame(columns=['Sub_ID', 'pathologic_side','pathology_location','age', 
                                         'gender','MT_P','MT_H','RMT_P','RMT_H','Pathology','motor_status',
                                         'FAP','FAH','ADCP', 'ADCH','ADP','ADH','RDP','RDH','FDP','FDH','loc'])
    new_data_sep.astype({'loc': 'int32'})
    for i in range(strm_median_P.shape[0]):
        for j in range(100):
            new_data_sep = new_data_sep.append(data.iloc[j])
        new_data_sep.iloc[i*100:(i+1)*100,11] = strm_P[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,12] = strm_H[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,13] = strm_P_adc[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,14] = strm_H_adc[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,15] = strm_P_ad[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,16] = strm_H_ad[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,17] = strm_P_rd[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,18] = strm_H_rd[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,19] = strm_P_fd[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,20] = strm_H_fd[i,:]
        new_data_sep.iloc[i*100:(i+1)*100,21] = np.int64(np.arange(0,100))
    new_data_sep.reset_index()
    #new_data_sep.to_csv('/Users/boshra/Desktop/my_table_patients.csv') 
    return new_data_sep

            
    
    
