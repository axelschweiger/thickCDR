#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 13 12:56:34 2025

@author: axel
"""
import xarray as xr
import sys
from dask.diagnostics import ProgressBar
from sklearn.neighbors import BallTree, KDTree
sys.path.append('/home/axel/ax_python_lib/im')
import pandas as pd
import datetime as dt


sys.path.append('/home/axel/ax_python_lib/axel_utilities')
def fix_thickCDR(): 
    # I needed to fix the default version created by IDL to remove duplicates for ENVISAT
    # and rename the index from time to cdr_index. I should do that in the IDL script 
    # that creates the data
#    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20250930_V1.0.nc'
    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20190920_V2.0.nc'
    cdr=xr.open_dataset(cdrfile)
#    
 
#    time=time.swap_dims({"time":"cdr_index"})
    cdr=cdr.rename_dims({"time":"cdr_index" })
#    cdr['time']=time.rename_dims({"time":"cdr_index"})
    
    cdr=cdr.assign_coords({"cdr_index": cdr.INDEX.data.astype('int64')})
    cdr=cdr.drop_duplicates('cdr_index',keep='first') # ENVISAT are duplicate
    time=dt.datetime(1900,1,1)+pd.to_timedelta(cdr.FDATE,unit='days')
    time_array=xr.DataArray(time,dims='cdr_index',coords={'cdr_index':cdr.cdr_index})
    cdr['time']=time_array                      
    with ProgressBar():
#        cdr.to_netcdf('/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc')
         cdr.to_netcdf('/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20190920_V3.0.nc')
    return

if __name__ == "__main__":
    fix_thickCDR()
