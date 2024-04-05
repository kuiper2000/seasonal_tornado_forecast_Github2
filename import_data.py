#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from   netCDF4 import Dataset as NetCDFFile
import geopandas
import descartes
import pandas as pd
from scipy import interpolate

class tornado_data:
    """
    importing data for predicting seasonal tornado activity including tornado CSV,  
    """
    def __init__(self, start_year: int, end_year: int, colab: True):
        self.start_year = start_year
        self.end_year   = end_year
        self.colab      = colab
        #self.G    = None
        #self.G1   = None
        #self.e    = None
        #self.out  = None
        
    def _return_date(self):
        return pd.date_range(str(self.start_year)+'-01-01', str(self.end_year)+'-12-31', freq='M')
    
    def _data_sst(self, obs=True):
        """
        importing sst data
        """
        dates            = self._return_date()
        print('working on SST data')
        if obs:
            # load obs SST 
            if self.colab:
                nc           = NetCDFFile("/content/drive/MyDrive/Colab/2021_research/tornado_data/sst.nc")
            else:
                nc           = NetCDFFile("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/sst.nc")
            sst          = nc.variables['sst'][12*(self.start_year-1979):12*(self.end_year-1979+1),0,:,:]
            sst_anomaly  = np.zeros((np.shape(sst)))
            
            count        = 0
            for year in range(self.start_year,self.end_year+1):    
                for month in range(1,13):
                    posi  = np.squeeze(np.where((dates.year==year) & (dates.month==month)))
                    posi2 = np.squeeze(np.where((dates.month==month)))
                    sst_anomaly[count,:,:]              = np.squeeze(sst[posi,:,:])-\
                                                          sst[posi2,:,:].mean(axis=0)
                    count = count+1
            return sst, sst_anomaly
        
        else:
            # load SPEAR SST data
            sst          = np.zeros((12,(self.end_year-self.start_year+1)*12,360,576)) # initialization month, forecast month (different lead mix), y, x 
            sst_ens      = np.zeros((15,(self.end_year-self.start_year+1)*12,360,576)) # ens, initialization month, forecast month (different lead mix), y, x 

            count = 0
            for init_time in ['01','02','03','04','05','06','07','08','09','10','11','12']:
                print(init_time)
                dates = pd.date_range('1992-'+init_time,  periods=276, freq='1M')  
                posi  = np.squeeze(np.where((dates.year>=self.start_year) & (dates.year<=self.end_year)))
                for ens in range(1,16,1):
                    if self.colab:
                        nc    = NetCDFFile("/content/drive/MyDrive/Colab/2021_research/tornado_data/sst/SST_"+init_time+"_ens%02d.nc" % (ens,))
                    else:
                        nc    = NetCDFFile("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/sst/SST_"+init_time+"_ens%02d.nc" % (ens,))
                    data  = nc.variables['SST'][:,:,:]
                    #print(np.shape(data))
                    sst_ens[ens-1,:,:,:] = data[12*(self.start_year-1992):12*(self.end_year+1-1992),:,:]
                sst[count,:,:,:]     = sst_ens[:,:,:,:].mean(axis=0)
                count                = count+1
            print("return ensemble mean sst")
            print("dim = (init_month,step(month),lat,lon)")
            return sst, sst_ens # the output SST has format of initialization month,  
        
    def _ats(self):
        if self.colab:
            data             = np.load('/content/drive/MyDrive/Colab/2021_research/tornado_data/ats_EOF.npz')
        else:
            data             = np.load('/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/ats_EOF.npz')
        pcs_atm          = data['pcs'][3,:,:]
        z925_regression  = data['z925_regression']
        EOF_obs_combine  = data['EOF_obs_combine']
        pcs_atm          = np.reshape(pcs_atm,[20,12,30,15]) # modes, init_month, year, member

        EOF_obs_combine  = np.reshape(EOF_obs_combine,[20,3,130, 520])
        
        if self.colab:
            data             = np.load('/content/drive/MyDrive/Colab/2021_research/tornado_data/SPEAR_monthly_data_ivt.npz')
        else:
            data             = np.load('/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/SPEAR_monthly_data_ivt.npz')
        ivtu_model_ens   = data['ivtu_model_ens']
        ivtv_model_ens   = data['ivtv_model_ens']
        
        
        pcs_template     = pcs_atm[:,:,0:29,:]
        pcs_template     = np.reshape(pcs_template,[20,12*29*15])

        IVTu_regression  = np.zeros((20,360,576))
        IVTv_regression  = np.zeros((20,360,576))
        ivtu_model_ens   = np.reshape(ivtu_model_ens,[12*29*15,360*576])
        ivtv_model_ens   = np.reshape(ivtv_model_ens,[12*29*15,360*576])


        for mode in range(20):
            series4            = np.squeeze(pcs_template[mode,:])
            X_mat              = np.vstack((np.ones(series4.shape[0]), series4[:])).T
            coef_total         = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(ivtu_model_ens[:,:])
            IVTu_regression[mode,:,:]    = np.reshape(coef_total[1,:],[360,576])
    
            series4            = np.squeeze(pcs_template[mode,:])
            X_mat              = np.vstack((np.ones(series4.shape[0]), series4[:])).T
            coef_total         = np.linalg.inv(X_mat.T.dot(X_mat)).dot(X_mat.T).dot(ivtv_model_ens[:,:])
            IVTv_regression[mode,:,:]    = np.reshape(coef_total[1,:],[360,576])

        del ivtu_model_ens, ivtv_model_ens
        
        return pcs_atm, EOF_obs_combine, IVTu_regression, IVTv_regression, z925_regression

        
    def _tornado(self):
        """
        load tornado data
        """
        if self.colab:
            data                 = np.load('/content/drive/MyDrive/Colab/2021_research/tornado_data/tornado_num_1992_2021_25.npz')
        else:
            data                 = np.load('/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/tornado_num_1992_2021_25.npz')
        tornado              = data['tornado_month']
        return tornado
    
    def _data_landsea_mask(self,mask_type=[False,False,False]):
        """
        load landsea mask
        type 1 = sst, type 2 = tornado, type 3 = atmosphere
        """
        print("type 1 = sst, type 2 = tornado, type 3 = atmosphere, input example = [True,False,False]")
        
        # using sst data to generate land-sea mask for atmospheric variable
        if self.colab:
            nc                                = NetCDFFile("/content/drive/MyDrive/Colab/2021_research/tornado_data/sst.nc")
        else:
            nc                                = NetCDFFile("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/sst.nc")
        sst                               = nc.variables['sst'][12*(self.start_year-1979):12*(self.end_year-1979),0,:,:]
        lon_SST_obs                       = nc.variables['lon'][:]
        lat_SST_obs                       = nc.variables['lat'][:]
        f                                 = interpolate.interp2d(lon_SST_obs,lat_SST_obs,sst[0,:,:],kind='linear')
        
        if mask_type[0]:
            land_sea_mask                     = f(lon_SST_obs,lat_SST_obs)
            land_sea_mask[land_sea_mask<-100] = 0
            land_sea_mask[land_sea_mask!=0]   = 1
            #land_sea_mask                     = 1-land_sea_mask
            return land_sea_mask
        if mask_type[1]:
            lat                               = np.arange(15,72.5,2.5)
            lon                               = np.arange(150,302.5,2.5)     
            land_sea_mask                     = f(lon,lat)
            land_sea_mask[land_sea_mask<-100] = 0
            land_sea_mask[land_sea_mask!=0]   = 1
            land_sea_mask                     = 1-land_sea_mask
            return land_sea_mask
        
        if mask_type[2]:
            lat                               = np.arange(15,70,0.5)
            lon                               = np.arange(150,300,0.5)        
            land_sea_mask                     = f(lon,lat)
            land_sea_mask[land_sea_mask<-100] = 0
            land_sea_mask[land_sea_mask!=0]   = 1
            land_sea_mask                     = 1-land_sea_mask
            return land_sea_mask

    def _lat_lon(self,sst=False,tornado=False, atmosphere=False,IVT=False):
        if self.colab:
            nc           = NetCDFFile("/content/drive/MyDrive/Colab/2021_research/tornado_data/sst.nc")
        else:
            nc           = NetCDFFile("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/sst.nc")
        if sst:
            lat          = nc.variables['lat']
            lon          = nc.variables['lon']
            xx,yy        = np.meshgrid(lon,lat)
        if atmosphere:
            xx,yy        = np.meshgrid(np.arange(100,360,0.5),np.arange(15,80,0.5))

        if tornado:
            lat          = np.arange(15,72.5,2.5)
            lon          = np.arange(150,302.5,2.5)   
            xx,yy        = np.meshgrid(lon,lat)
        if IVT:
            lat          = nc.variables['lat']
            lon          = nc.variables['lon']
            xx,yy        = np.meshgrid(lon,lat)
        return xx, yy
        
    def _maps(self):
        """
        returning the parameters for plotting the maps
        """
        if self.colab:
            states = geopandas.read_file('/content/drive/MyDrive/Colab/2020_research/Seasonal_tornado_forecast/geopandas-tutorial/data/usa-states-census-2014.shp')
            states = states.to_crs("EPSG:4326")
            f = open("/content/drive/MyDrive/Colab/2021_research/tornado_data/lat.bin", "r")
            costal_lat         = np.fromfile(f, np.float32)
            f = open("/content/drive/MyDrive/Colab/2021_research/tornado_data/lon.bin", "r")
            costal_lon         = np.fromfile(f, np.float32)
        else:
            states = geopandas.read_file('/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2020_research/Seasonal_tornado_forecast/geopandas-tutorial/data/usa-states-census-2014.shp')
            states = states.to_crs("EPSG:4326")
            f = open("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/lat.bin", "r")
            costal_lat         = np.fromfile(f, np.float32)
            f = open("/Users/kaichiht/Library/CloudStorage/GoogleDrive-kuiper2000@gmail.com/My Drive/Colab/2021_research/tornado_data/lon.bin", "r")
            costal_lon         = np.fromfile(f, np.float32)

        return states, costal_lat, costal_lon
    
    

