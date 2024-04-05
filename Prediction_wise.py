#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd

class Prediction_wise():
    def __init__(self,obs=np.ndarray,model=np.ndarray,obs_init=str, obs_final=str, obs_interval=str, model_init=str, model_final=str, model_init_freq=str,total_forecast_lead=int):
        self.obs                 = obs
        self.model               = model
        self.obs_init            = obs_init
        self.obs_final           = obs_final
        self.obs_interval        = obs_interval
        self.model_init          = model_init
        self.model_final         = model_final
        self.model_init_freq     = model_init_freq
        self.total_forecast_lead = total_forecast_lead
        return
    
    def _prediction_eval(self):
        """
        the input should have the following form
        obs        = dates
        reforecast = initial month, forecast dates
        """
        score = np.zeros((12,self.total_forecast_lead))
        # calculate Nino 3.4 scores 
        
        
        for month in range(12):                  
            dates          = pd.date_range(self.model_init,  self.model_final,   freq=self.model_init_freq)   # total length of reanalysis data
            dates          = dates.shift(month, freq="MS")
            dates_obs      = pd.date_range(self.obs_init,    self.obs_final,     freq=self.obs_interval )


            for lead_time in range(self.total_forecast_lead):
                target_dates   = dates.shift(lead_time, freq="MS")
                posi           = []
                          
                # collecting the position index of forecast month of obs  
                for i in range(np.size(target_dates)):
                    temporary_posi = np.where((dates_obs.month==target_dates[i].month) & (dates_obs.year==target_dates[i].year))[0]
                    posi           = np.squeeze(np.append(posi, temporary_posi))
                posi         = posi.astype(int)
                #print(posi)
                min_length   = int(np.min([np.size(posi),np.size(target_dates)]))
                model_series = np.squeeze(self.model[month,0:min_length,lead_time])
                obs_series   = np.squeeze(self.obs[posi[0:min_length]])
                score[month,lead_time] = np.corrcoef(model_series,obs_series)[0][1]
                
        return score

