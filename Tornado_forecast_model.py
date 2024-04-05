#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd
from   statsmodels.distributions.empirical_distribution import ECDF
from   sklearn import linear_model
from   datetime import datetime
from sklearn import linear_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
regr          = linear_model.LinearRegression()


class tornado_model():
    def  __init__(self,tornado_data = np.ndarray, predictor = np.ndarray, init_year = int):
        self.tornado              = tornado_data   
        self.predictor            = predictor     # predictor index, initial month, year X lead
        self.init_year            = init_year
        
        
        # tornado data has dimension of time X 1
        # sst data has dimension of time X lat X lon 
        
    def _quantile_mapping(self,input_variable):    
        
        ecdf                      = ECDF(input_variable)
        percentile                = ecdf(input_variable)

        return percentile
    
    def _min_max_scaling(self,input_variable,axis):
        input_min        = np.min(input_variable,axis=axis)
        scaled_input     = input_variable-input_min

        return scaled_input
    
    def _find_next_target_month(self,dates, month):
        # the input should have a format of dd/mm/yyyy
        string_input_with_date = dates
        current_year           = datetime.strptime(string_input_with_date, "%d/%m/%Y")
        string_input_with_date = ''+str(current_year.day)+'/'+str(month).zfill(2)+'/'+str(current_year.year)+''
        first_year             = datetime.strptime(string_input_with_date, "%d/%m/%Y")
        string_input_with_date = ''+str(current_year.day)+'/'+str(month).zfill(2)+'/'+str(current_year.year+1)+''
        second_year            = datetime.strptime(string_input_with_date, "%d/%m/%Y")
        
        if current_year<=first_year:
            return first_year.year
        else:
            return second_year.year
    
    def _forecast(self, leave_one_out=True, normalize=True):
        pcs_obs_sst                 = self.predictor 
        dim                         = np.shape(self.tornado) 
        dim_sst                     = np.shape(self.predictor)
        self.tornado                = np.reshape(self.tornado,[dim[0],int(np.size(self.tornado)/dim[0])])
        tornado_percentile          = np.zeros(np.shape(self.tornado))
        for i in range(int(np.size(self.tornado)/dim[0])):
            tornado_percentile[:,i] = self._quantile_mapping(self.tornado[:,i])
        tornado_dates               = pd.date_range(str(self.init_year )+'-03', periods=dim[0], freq='12M')
        tornado_percentile_min      = np.min(tornado_percentile,axis=0)
        tornado_percentile          = tornado_percentile-tornado_percentile_min

        
        
        coef                        = np.zeros((12,12,dim[0],dim_sst[0],dim[1]))
        predict                     = np.zeros((int(dim_sst[0]),12,12,dim[0],dim[1]))
        for mode in range(1,int(dim_sst[0])+1):
            print('working on mode = '+str(mode))
            for init_month in range(1,13):         # for initialization month 
                dates_1M              = pd.date_range(str(self.init_year )+'-'+str(init_month).zfill(2)+'', periods=dim[0]*12, freq='1M')  # the initialization month  
                
                for month in range(1,13):          # month used as predictors 
                    predictor_posi    = np.squeeze(np.where(dates_1M.month == month))
                    dates_predictor   = dates_1M[predictor_posi]                                                     # the month used as predictor (forecast lead considered)
                    target_init_year  = self._find_next_target_month('15/'+str(dates_predictor[0].month)+'/'+str(dates_predictor[0].year),3)
#                     print('the init time is: year     ='+str(dates_1M[0].year)       +' month='+str(dates_1M[0].month).zfill(2))
#                     print('the predictor time is: year='+str(dates_predictor[0].year)+' month='+str(dates_predictor[0].month).zfill(2))
#                     print('the target time is: year   ='+str(target_init_year)       +' month='+str(3).zfill(2))
                    
                    
                    target_posi       = np.squeeze(np.where(tornado_dates.year >=target_init_year))
                    size_min          = np.min([np.size(target_posi),np.size(predictor_posi)])
                    #print(size_min)
                    series1           = tornado_percentile[target_posi[0:size_min],:]
                    series2           = self.predictor[0:mode,init_month-1,predictor_posi[0:size_min]]
                    
                    #print(target_init_year)
                    #print(np.size(series2))
                    for year in range(target_init_year,self.init_year+dim[0]):
                        #print(year-self.init_year)
                        if leave_one_out: 
                            series4 = np.delete(series2, year-target_init_year,axis=1)
                            series3 = np.delete(series1, year-target_init_year,axis=0)                      
                        else:
                            series4 = series2
                            series3 = series1          
                        # series4 = (series4-np.mean(series4))/np.std(series4)
                        # series3 = (series3-np.mean(series3))/np.std(series3)
                        # print(np.shape(series4))
                        # one      = np.ones((1,np.size(series3)))
                        # series4  = np.append(one, series4, axis=0)
                        regularization = np.std(series4)**2
                        ans      = np.linalg.inv(series4.dot(series4.T)+regularization*0.01*np.eye(series4.shape[0]))\
                                        .dot(series4.dot(series3))
                    
                       
                        coef[init_month-1,month-1,year-self.init_year,0:mode,:] = ans
                        X_val  = np.zeros((mode,1))
                        Y_val  = np.zeros((1,1))
                        X_val[:,0] = np.transpose(series2[0:mode,year-target_init_year])[:]
                        # one    = np.ones((1,np.size(X_val)))
                        # X_val  = np.append(one, X_val, axis=0)
                        predict[mode-1,init_month-1,month-1,year-self.init_year,:] = X_val.T.dot(ans)

                        
                        
   
                        if normalize:
                            Y_train   =  series3
                            X_train   =  series4.T
                            model_ols =  linear_model.LinearRegression()
                            model_ols.fit(X_train,Y_train)
                            coef[init_month-1,month-1,year-self.init_year,0:mode,:] = (model_ols.coef_).T
                            X_val     = np.zeros((mode,1))
                            X_val[:,0] = np.transpose(series2[0:mode,year-target_init_year])[:]
                            predict[mode-1,init_month-1,month-1,year-self.init_year,:] = model_ols.predict(np.transpose(X_val))[0]
                
                    
        return predict, coef,tornado_percentile, tornado_percentile_min
                    #print(np.shape(series2))
#         for mode in range(1,int(np.size(self.tornado)/dim[0])):
#             print('working on mode = '+str(mode))
#             for init_month in range(1,13):         # for initialization month 
#                 dates_1M  = pd.date_range(str(self.init_year )+'-'+str(init_month).zfill(2)+'', periods=dim[0]*12, freq='1M')     
#                 for month in range(1,13):          # month used as predictors 
#                     posi = np.squeeze(np.where((dates_1M.month==month) & (dates_1M.year<=2021)))
            
#                     if (dates_1M[posi[0]].year==self.init_year ) and (month<=3):
#                         init_year = self.init_year 
#                         series1   = tornado_percentile[:,:]
#                         series2   = pcs_obs_sst[0:mode,init_month-1,posi]
#                     elif (dates_1M[posi[0]].year==self.init_year ) and (month>3):
#                         init_year = self.init_year +1
#                         series1   = tornado_percentile[1:,:]
#                         series2   = pcs_obs_sst[0:mode,init_month-1,posi[:-1]]
#                     elif (dates_1M[posi[0]].year==self.init_year +1) and (month<=3):
#                         init_year = self.init_year +1
#                         series1   = tornado_percentile[1:,:]
#                         series2   = pcs_obs_sst[0:mode,init_month-1,posi]
#                     elif (dates_1M[posi[0]].year==self.init_year +1) and (month>3):
#                         init_year = self.init_year +2
#                         series1   = tornado_percentile[2:,:]
#                         series2   = pcs_obs_sst[0:mode,init_month-1,posi[:-1]]
                
#                     for year in range(init_year,2022):
            
#                         series4 = np.delete(series2, year-init_year,axis=1)
#                         series3 = np.delete(series1, year-init_year,axis=0)
                    
                
                    

#                         ans = np.linalg.inv(series4.dot(series4.T)+10**0.9*np.eye(series4.shape[0]))\
#                                        .dot(series4.dot(series3))
                       
                    
#                         coef_SPEAR_sst_region[init_month-1,month-1,year-1992,0:mode,:] = ans
#                         X_val  = np.zeros((mode,1))
#                         Y_val  = np.zeros((1,1))
#                         X_val[:,0] = np.transpose(series2[0:mode,year-init_year])[:]
#                         Y_val_LM_sst_EOF_region[mode-1,init_month-1,month-1,year-1992,:] = X_val.T.dot(ans)

