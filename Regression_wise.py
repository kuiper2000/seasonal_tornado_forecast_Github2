#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd


class Regression_wise():
    def __init__(self, data1: np.ndarray, data2: np.ndarray, dim1: int, dim2: int):
        print("input: data1, data2, dim1 = the dimension of data1 used for regression, dim2 = the dimension of data2 used for regression")
        self.data1     = data1
        self.data2     = data2
        self.dim1      = dim1
        self.dim2      = dim2
        
    def _swap_dim(self, data,last_dim: int):
        """
        swap the dimensition of data and put the dimesion of interest to the last place 
        """
        
        ndim           = data.ndim
        original_order = [i for i in range(ndim)]

        list_total     = ['a','b','c','d','e','f','g']
        list_subset    = list_total[0:np.size(original_order)]


        str1           = str()
        for i in np.arange(np.size(list_subset)):
            str1 += list_subset[i]
        
        index = list_subset[last_dim]
        list_subset[last_dim] = str()
        str2  = str()
        for i in np.arange(np.size(list_subset)):
            str2 += list_subset[i]

        str2 += index
        
        return str1, str2

    def _linear_regression(self):
        
        data1_str1, data1_str2 = self._swap_dim(self.data1, self.dim1)
        data2_str1, data2_str2 = self._swap_dim(self.data2, self.dim2)
        
        self.data1             = np.einsum(data1_str1+'->'+data1_str2, self.data1)
        self.data2             = np.einsum(data2_str1+'->'+data2_str2, self.data2)
        
        data_dim1              = np.shape(self.data1)
        data_dim2              = np.shape(self.data2)
        
        print(data_dim1)
        print(data_dim2)
        
        data_size1             = int(np.size(self.data1))
        data_size2             = int(np.size(self.data2))
        
        self.data1             = np.reshape(self.data1,[int(data_size1/data_dim1[-1]),int(data_dim1[-1]) ])
        self.data2             = np.reshape(self.data2,[int(data_size2/data_dim2[-1]),int(data_dim2[-1]) ])
        #regularization         = np.eye(np.size(self.data1[:,0])) * 10**-3
        
        coef                   = np.linalg.inv(self.data1.dot(self.data1.T)).dot(self.data1.dot(self.data2.T))        
        data_dim3              = np.append([i for i in data_dim1[:-1]],[i for i in data_dim2[:-1]])
        
        coef                   = np.squeeze(np.reshape(coef,data_dim3))
        
        self.data1             = np.reshape(self.data1,data_dim1)
        self.data2             = np.reshape(self.data2,data_dim2)
        self.data1             = np.einsum(data1_str2+'->'+data1_str1, self.data1)
        self.data2             = np.einsum(data2_str2+'->'+data2_str1, self.data2)
        
        return coef
    
    
    def _corrcoef(self):
        data1_str1, data1_str2 = self._swap_dim(self.data1, self.dim1)
        data2_str1, data2_str2 = self._swap_dim(self.data2, self.dim2)
        
        self.data1             = np.einsum(data1_str1+'->'+data1_str2, self.data1)
        self.data2             = np.einsum(data2_str1+'->'+data2_str2, self.data2)
        
        data_dim1              = np.shape(self.data1)
        data_dim2              = np.shape(self.data2)
        
        # print(data_dim1)
        # print(data_dim2)
        
        data_size1             = int(np.size(self.data1))
        data_size2             = int(np.size(self.data2))
        
        self.data1             = np.reshape(self.data1,[int(data_size1/data_dim1[-1]),int(data_dim1[-1]) ])
        self.data2             = np.reshape(self.data2,[int(data_size2/data_dim2[-1]),int(data_dim2[-1]) ])
        
        coef                   = np.zeros((int(data_size1/data_dim1[-1]),int(data_size2/data_dim2[-1])))
        for i in range(int(data_size1/data_dim1[-1])):
            for j in range(int(data_size2/data_dim2[-1])):
                posi      = np.squeeze(np.where((np.abs(self.data1[i,:])>0) & (np.abs(self.data2[j,:])>0)))
                coef[i,j] = np.corrcoef(self.data1[i,posi],self.data2[j,posi])[0][1]
        data_dim3              = np.append([i for i in data_dim1[:-1]],[i for i in data_dim2[:-1]])
        coef                   = np.squeeze(np.reshape(coef,data_dim3))
        
        self.data1             = np.reshape(self.data1,data_dim1)
        self.data2             = np.reshape(self.data2,data_dim2)
        self.data1             = np.einsum(data1_str2+'->'+data1_str1, self.data1)
        self.data2             = np.einsum(data2_str2+'->'+data2_str1, self.data2)
        return coef
    
    def _RPSS(self,calculate_pattern=False):
        
        
        data1_str1, data1_str2 = self._swap_dim(self.data1, self.dim1)
        data2_str1, data2_str2 = self._swap_dim(self.data2, self.dim2)
        print(data2_str1)
        print(data2_str2)
        self.data1             = np.einsum(data1_str1+'->'+data1_str2, self.data1)
        self.data2             = np.einsum(data2_str1+'->'+data2_str2, self.data2)
        
        data_dim1              = np.shape(self.data1)
        data_dim2              = np.shape(self.data2)
        
        # print(data_dim1)
        # print(data_dim2)
        
        data_size1             = int(np.size(self.data1))
        data_size2             = int(np.size(self.data2))
        
        self.data1             = np.reshape(self.data1,[int(data_size1/data_dim1[-1]),int(data_dim1[-1]) ])
        self.data2             = np.reshape(self.data2,[int(data_size2/data_dim2[-1]),int(data_dim2[-1]) ])

        # dimension of interest is rearranged to the last term 
        print(int(data_size1/data_dim1[-1]))
        model_category         = np.zeros((3,int(data_size1/data_dim1[-1]),int(data_size2/data_dim2[-1]),int(data_dim2[-1]) ))
        RPSS                   = np.zeros((int(data_size1/data_dim1[-1]),int(data_size2/data_dim2[-1])))
        
        for i in range(int(data_size1/data_dim1[-1])):
            for j in range(int(data_size2/data_dim2[-1])):
                posi      = np.squeeze(np.where((np.abs(self.data1[i,:])>0) & (np.abs(self.data2[j,:])>0)))
                series1   = self.data1[i,:] # obs
                series2   = self.data2[j,posi] # forecast 
                
                series1          = (series1-np.mean(series1))/np.std(series1)
                series2          = (series2-np.mean(series2))/np.std(series2)
                lower_percentile = np.percentile(series1,33)
                upper_percentile = np.percentile(series1,66)
                obs_category     = np.zeros((3,int(data_dim1[-1])))
                clim_category    = np.ones((3,int(data_dim1[-1])))*1/3
                
                obs_category[0,:][series1<lower_percentile] = 1
                obs_category[2,:][series1>upper_percentile] = 1
                obs_category[1,:][(series1<=upper_percentile) & (series1>=lower_percentile)] = 1
                
                std_range                = np.std(series2-series1[posi]) 
                for k in range(np.size(posi)):
                    s                        = np.random.normal(series2[k], std_range, 1000)
                    model_category[2,i,j,posi[k]] = np.size(np.where(s>upper_percentile))/1000
                    model_category[0,i,j,posi[k]] = np.size(np.where(s<lower_percentile))/1000
                    model_category[1,i,j,posi[k]] = 1-model_category[2,i,j,posi[k]]-model_category[0,i,j,posi[k]]
                                           
                RPS              = sum(sum((model_category[:,i,j,posi]-obs_category[:,posi])**2))
                RPS_clim         = sum(sum((clim_category-obs_category)**2))
                RPSS[i,j]        = 1-RPS/RPS_clim
        
        
        
        data_dim3              = np.append([i for i in data_dim1[:-1]],[i for i in data_dim2[:-1]])
        RPSS                   = np.squeeze(np.reshape(RPSS,data_dim3))
    
        if calculate_pattern:
            print("Calculate Pattern")
        else:
            data_dim3              = np.append([3],[i for i in data_dim2])
            model_category         = np.squeeze(np.reshape(model_category,data_dim3))     

            data_dim3              = np.append([3],[i for i in data_dim1])
            obs_category           = np.squeeze(np.reshape(obs_category,data_dim3))     
        
        
        self.data1             = np.reshape(self.data1,data_dim1)
        self.data2             = np.reshape(self.data2,data_dim2)
        self.data1             = np.einsum(data1_str2+'->'+data1_str1, self.data1)
        self.data2             = np.einsum(data2_str2+'->'+data2_str1, self.data2)
        # print(np.shape(self.data1))
        # print(np.shape(self.data2))
        
        return RPSS, obs_category, model_category
    
    def _attribute_diagram(self,threshold  = 0.3):
        
        
        RPSS, obs_category, model_category = self._RPSS()
        coef                               = self._corrcoef()
        coef                               = np.reshape(coef,[np.size(coef),1])
        posi                               = np.squeeze(np.where(coef[:,0]>threshold))
        
        data1_str1, data1_str2 = self._swap_dim(self.data1, self.dim1)
        data2_str1, data2_str2 = self._swap_dim(self.data2, self.dim2)
        
        self.data1             = np.einsum(data1_str1+'->'+data1_str2, self.data1)
        self.data2             = np.einsum(data2_str1+'->'+data2_str2, self.data2)
        
       
        
        
        data_dim1              = np.shape(self.data1)
        data_dim2              = np.shape(self.data2)
        data_dim3              = np.shape(obs_category)
        data_dim4              = np.shape(model_category)
        
        
        model_category         = np.reshape(model_category,[3,int(np.size(self.data2)/data_dim2[-1]),data_dim2[-1]])
        
        scale                  = np.arange(0,1,0.1)
        pool                   = np.zeros((3,10,data_dim1[-1]))
        hit                    = np.zeros((3,10,data_dim1[-1]))
        self.data2             = np.reshape(self.data2,[int(np.size(self.data2)/data_dim2[-1]),data_dim2[-1]])
       
                
            
    
        for one_out in range(data_dim1[-1]):
            leave_out = np.squeeze(np.where(np.arange(data_dim1[-1])!=one_out))
            for i in range(np.size(posi)):
                for j in range(data_dim1[-1]-1):
                    posi2                     = np.max(np.where(model_category[0,posi[i],leave_out[j]]>=scale))  
                    pool[0,posi2,one_out]     = 1 + pool[0,posi2,one_out] # how many times the forecast are made 
                    if obs_category[0,leave_out[j]] ==1:                  # how many times the forecast hit
                        hit[0,posi2,one_out]  = hit[0,posi2,one_out]+1
            
                    # near normal
                    posi2                     = np.max(np.where(model_category[1,posi[i],leave_out[j]]>=scale))  
                    pool[1,posi2,one_out]     = 1 + pool[1,posi2,one_out]
                    if obs_category[1,leave_out[j]] ==1:
                        hit[1,posi2,one_out]  = hit[1,posi2,one_out]+1
                
                    # above normal
                    posi2                     = np.max(np.where(model_category[2,posi[i],leave_out[j]]>=scale))  
                    pool[2,posi2,one_out]     = 1 + pool[2,posi2,one_out]
                    if obs_category[2,leave_out[j]] ==1:
                        hit[2,posi2,one_out]  = hit[2,posi2,one_out]+1
        return hit, pool, scale
    
    def _calibration_attribute(self):
        hit, pool, scale    = self._attribute_diagram()
        # calibrating the forecast probability
        new_probaiblity     = np.zeros((3,np.size(scale),30))
        reg                 = np.zeros((3,2,30))
        
        
        
        for i in range(30):
            series1            = hit[0,:,i]/pool[0,:,i]
            X_mat              = np.vstack((np.ones(10), series1)).T

            coef_total         = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(scale)
            reg[0,0,i]         = coef_total[1]
            reg[0,1,i]         = coef_total[0]
            new_probaiblity[0,:,i] = series1*reg[0,0,i] +reg[0,1,i] 
    
    
            # near normal
            posi               = np.squeeze(np.where(pool[1,:,i]>0))
            series1            = hit[1,posi,i]/pool[1,posi,i]
            X_mat              = np.vstack((np.ones(np.size(posi)), series1)).T

            coef_total         = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(scale[posi])
            reg[1,0,i]         = coef_total[1]
            reg[1,1,i]         = coef_total[0]
            new_probaiblity[1,:,i]    = scale
            new_probaiblity[1,posi,i] = series1*reg[1,0,i]+reg[1,1,i]
    
    
            # above normal
            series1            = hit[2,:,i]/pool[2,:,i]
            X_mat              = np.vstack((np.ones(10), series1)).T

            coef_total         = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(scale)
            reg[2,0,i]         = coef_total[1]
            reg[2,1,i]         = coef_total[0]
            new_probaiblity[2,:,i] = series1*reg[2,0,i]+reg[2,1,i] 
            
        return new_probaiblity


# In[ ]:




