#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd

class domain_calculation:
    def __init__(self, data: np.ndarray, lat: np.ndarray, lon: np.ndarray, domain: np.ndarray):
        self.data      = data
        self.lat       = lat
        self.lon       = lon
        self.domain    = domain
        
    def _location(self):
        """
        return domain of interest 
        """
        posi_lat = np.squeeze(np.where((self.lat>=self.domain[0]) & (self.lat<=self.domain[1])))[0]
        posi_lon = np.squeeze(np.where((self.lon>=self.domain[2]) & (self.lon<=self.domain[3])))[1]
        return posi_lat, posi_lon
    
    def _domain_average(self,lat_weighted=True):
        posi_lat,posi_lon = self._location()
        max_x      = np.max(posi_lon)
        min_x      = np.min(posi_lon)
        max_y      = np.max(posi_lat)
        min_y      = np.min(posi_lat)
        dim        = np.shape(self.data)
        
        average    = (np.array(self.data)[:,min_y:max_y,min_x:max_x].mean(axis=2)).mean(axis=1)
        
        return average

