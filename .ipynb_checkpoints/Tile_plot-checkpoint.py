#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd
import matplotlib.patheffects as path_effects


class making_tile():
    def  __init__(self,data=np.ndarray):
        self.data = data
        
    
    def _tile_plot(self, text=False):
        # this function is used when the input is [initial_month X forecast_lead]
        # re-organize the data for plotting 
        score_map         = np.zeros(np.shape(self.data))
        score_map_anomaly = np.zeros(np.shape(self.data))
        score_map_upper   = np.zeros(np.shape(self.data))
        score_map_lower   = np.zeros(np.shape(self.data))
        target_month_map  = np.zeros(np.shape(self.data))    
        # convert data from init_month X forecast_lead to init_month X target_month
        
        for forecast_lead in range(0,12):
            for init_month in range(1,13):
                if (init_month+forecast_lead)%12 ==0:
                    target_month = 12
                else:
                    target_month = (init_month+forecast_lead)%12
                #print(target_month)
                score_map[init_month-1,target_month-1] = self.data[init_month-1,forecast_lead]
                score_map_anomaly[init_month-1,target_month-1]  = self.data[init_month-1,forecast_lead]-np.mean(self.data[:,forecast_lead])
                #score_map_anomaly[init_month-1,target_month-1] = NINO34_scores[init_month-1,forecast_lead]-np.mean(NINO34_scores[:,forecast_lead])
                score_map_upper[init_month-1,target_month-1] = np.percentile(self.data[:,forecast_lead],66)
                score_map_lower[init_month-1,target_month-1] = np.percentile(self.data[:,forecast_lead],33)
                target_month_map[init_month-1,target_month-1] = target_month
                
        new        = np.concatenate((score_map, score_map,score_map,score_map),axis=0)
        new        = np.concatenate((new, new, new,new),axis=1)
        new2       = np.concatenate((score_map_anomaly, score_map_anomaly,score_map_anomaly,score_map_anomaly),axis=0)
        new2       = np.concatenate((new2, new2, new2,new2),axis=1)

        new3       = np.concatenate((score_map_upper, score_map_upper,score_map_upper,score_map_upper),axis=0)
        new3       = np.concatenate((new3, new3, new3,new3),axis=1)
        new4       = np.concatenate((score_map_lower, score_map_lower,score_map_lower,score_map_lower),axis=0)
        new4       = np.concatenate((new4, new4, new4,new4),axis=1)

        new5       = np.concatenate((target_month_map, target_month_map,target_month_map,target_month_map),axis=0)
        new5       = np.concatenate((new5, new5, new5,new5),axis=1)

        
        
        score_map_extend = (new.T)[11+4:11+4+24,11+4:11+4+12]
        score_map_anomaly_extend = (new2.T)[11+4:11+4+24,11+4:11+4+12]
        score_map_upper  = (new3.T)[11+4:11+4+24,11+4:11+4+12]
        score_map_lower  = (new4.T)[11+4:11+4+24,11+4:11+4+12]
        target_month_map = (new5.T)[11+4:11+4+24,11+4:11+4+12]
        
        
        
        
        for i in range(12):
            for j in range(24):
                if ((j-i)>11 and (j>i)):
                    score_map_extend[j,i] = 0
                    score_map_anomaly_extend[j,i] = 0
                    target_month_map[i,i] = 0
                elif ((i-j)<12 and (i>j)):
                    score_map_extend[j,i] = 0
                    score_map_anomaly_extend[j,i] = 0
                    target_month_map[j,i] = 0
        if text:
            for i in range(12):
                for j in range(24):
                    if ((j-i)>11 and (j>i)):
                        score_map_extend[j,i] = np.nan
                        score_map_anomaly_extend[j,i] = np.nan
                        target_month_map[j,i] = np.nan
                    elif ((i-j)<12 and (i>j)):
                        score_map_extend[j,i] = np.nan
                        score_map_anomaly_extend[j,i] = np.nan
                        target_month_map[j,i] = np.nan

        return score_map_extend, score_map_upper, score_map_lower, score_map_anomaly_extend, target_month_map
    
    
    def _tile_plot_predictor_month(self,lead_time_label):
        # this function is used when the input is [initial_month X predictor_month]
        max_score   = np.zeros((np.max(lead_time_label).astype(int),))
        mean_score  = np.zeros((np.max(lead_time_label).astype(int),))
        std_score   = np.zeros((np.max(lead_time_label).astype(int),))
        init_month  = np.zeros((np.max(lead_time_label).astype(int),))
        lead_time_label[lead_time_label>100]=0
        for lead in range(np.max(lead_time_label).astype(int)):
            #print(lead)
            score_template            = np.zeros((14,14))
            score_template[1:-1,1:-1] = self.data[:,:]
            #print(score_template)
            score_template[0,1:-1]    = (self.data[:,:])[-1,:]
            score_template[-1,1:-1]   = (self.data[:,:])[0,:]
            score_template[:,0]       = score_template[:,-2]
            score_template[:,-1]      = score_template[:,1]
        
            
            posi_x,posi_y      = np.where(lead_time_label==lead)
            max_score[lead,]   = np.max(score_template[posi_x,posi_y])
            mean_score[lead,]  = np.mean(score_template[posi_x,posi_y])
            std_score[lead,]   = np.std(score_template[posi_x,posi_y])
            init_month[lead,]  = posi_x[0]

        
        template   = np.concatenate((score_template[1:13,1:13], score_template[1:13,1:13], score_template[1:13,1:13]), axis=0)
        template   = np.concatenate((template,template,template),axis=1)
        score_map3 = template[3:15,3:15]    

        score_map3_concatenate = np.concatenate((score_map3,score_map3,score_map3),axis=0)
        score_map3_concatenate = np.concatenate((score_map3_concatenate,score_map3_concatenate,score_map3_concatenate),axis=1)
        score_map3_concatenate = score_map3_concatenate.T[0:24,12:24]
        for i in range(12):
            for j in range(24):
                if ((j-i)>11 and (j>i)):
                      score_map3_concatenate[j,i]=np.nan
                elif ((i-j)<12 and (i>j)):
                      score_map3_concatenate[j,i]=np.nan
        
        score_map_extend       = score_map3_concatenate
        
        return score_map_extend, max_score, mean_score, std_score, init_month
    
    def _label(self):
        lead_time_label = np.ones((14,14))*1000

        for init_month in range(1,13):
            if init_month<4:
                for target_month in range(1,13):
                    if (init_month<=target_month) and (target_month<=3):
                        lead_time_label[init_month,target_month] = 3-init_month
                    else:
                        lead_time_label[init_month,target_month] = 3-init_month+12
            elif init_month==4:
                for target_month in range(1,13):
                    lead_time_label[init_month,target_month] = 11
            else:
                for target_month in range(1,13):
                    if (init_month<=target_month) and (target_month>3):
                        lead_time_label[init_month,target_month] = 3-init_month+12
                    elif (target_month<=init_month) and (target_month<=3):
                        lead_time_label[init_month,target_month] = 3-init_month+12
                    else:
                        lead_time_label[init_month,target_month] = 3-init_month+12+12  
        lead_time_label[1:13,1:13]
        template = np.concatenate((lead_time_label[1:13,1:13], lead_time_label[1:13,1:13], lead_time_label[1:13,1:13]), axis=0)
        template = np.concatenate((template,template,template),axis=1)
        lead_time_label2 = template[3:15,3:15]



        # replot the checker board 
        lead_time_label_concatenate = np.concatenate((lead_time_label2,lead_time_label2,lead_time_label2),axis=0)
        lead_time_label_concatenate = np.concatenate((lead_time_label_concatenate,lead_time_label_concatenate,lead_time_label_concatenate),axis=1)
        lead_time_label_concatenate = lead_time_label_concatenate.T[0:24,12:24]

        for i in range(12):
            for j in range(24):
                if ((j-i)>11 and (j>i)):
                      lead_time_label_concatenate[j,i]=np.nan
                elif ((i-j)<12 and (i>j)):
                      lead_time_label_concatenate[j,i]=np.nan
        return lead_time_label, lead_time_label_concatenate

