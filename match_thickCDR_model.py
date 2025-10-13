#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 14:35:33 2021
# extracts a 
@author: axel
"""
import sys
import xarray as xr
import pandas as pd 
import os
import numpy as np
import os
import dask
import logging 
import time 
import argparse
from  pathlib import Path as Path
import datetime as dt
from scipy import spatial
from dask.diagnostics import ProgressBar
from sklearn.neighbors import BallTree, KDTree
sys.path.append('/home/axel/ax_python_lib/im')
from model_maps_arctic import *
import flox
sys.path.append('/home/axel/ax_python_lib/axel_utilities')
import axutil
import pandas as pd

#import xoak

def read_thickCDR(): 
    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'
    cdr=xr.open_dataset(cdrfile)
    
#    cdr=cdr.where((cdr.SOURCE =='AIR-EM') & (cdr.PLATFORM=='2019MOSAIC'),drop=True)
    cdr=cdr.where((cdr.SOURCE =='AIR-EM'),drop=True)
    fdate=cdr.FDATE
  
    date=dt.datetime(1900,1,1)+pd.to_timedelta(fdate,unit='days')
    lat=cdr.LAT.data
    lon=axutil.lonTo180(cdr.LON.data)
    vardata=cdr.THICK_ICE_SNOW[:,0].data # Get mean value... not thickness (ice + snow
    cdr_index=cdr.INDEX.data.astype('int64')
    pos=np.column_stack((lat,lon))
    
    cdr_data={'cdr_index':cdr_index,'date':date,'pos':np.deg2rad(pos),'thick':vardata}
    return cdr_data

def points2pos(points): # helper function to reform that data points. 
    p=list()
    id_list=list()
    time_list=list()
    for id, point in points.items(): 
         plat=point['pos']['lat']
         plon=point['pos']['lon']
         plon=axutil.lonTo180(plon)
         p.append([plat,plon])
         id_list.append(id)
         t=dt.datetime.strptime(point['date'],'%Y%m%d')
         time_list.append(t)
    pointsd=np.array(p)          
    pos=np.deg2rad(pointsd) # array of 2 element arrays
    return pos,id_list,time_list 

def match_points(cdr_data,ds,varnames,ancil_data,dist_thresh=10.):
#    dist_thresh=10. # 10 km dist threshold
    OceanMask=ancil_data.OceanMask.rename({"y_sn":"y","x_ew":"x"})
    omask=OceanMask.data.flatten()

    ngrid_pts=ds.lat.size # number of grid points nx,ny 
    dlon=ds.lon.data
    dlat=ds.lat.data
 
    nx=ds.sizes['x']
    ny=ds.sizes['y']
    nt=ds.sizes['time'] # not currently used. Can subset later
    
    lat=dlat.flatten()
    lon=dlon.flatten()
#   create a linear index into the source array pointing only 
#   to the ocean gridpoints 
    oindex=np.arange(ngrid_pts)
#    oindex=np.array(list(range(0,lat.size)))
#   select only the pts inside the ocean_mask 
    oindex=oindex[OceanMask.data.flatten()==1]  
    lat=lat[omask==1]
    lon=lon[omask==1]
#  Latitude, longitudes needs to be in radians
    lat=np.deg2rad(lat)
    lon=np.deg2rad(axutil.lonTo180(lon))    
# create 2d array necessary for the BallTree routine 
    latlon = np.dstack([lat,lon])[0]   
#   convert the points dta structure into something we can use for looking up 
#   in th tree structure
    
    pos=cdr_data['pos']
    id_list=cdr_data['cdr_index']
    time_list=cdr_data['date']
      
    time_array=np.array(time_list,dtype='datetime64[ns]')
# build the spatial treee using distance on the sphere 
    mytree = BallTree(latlon, metric='haversine')
    dist, idx_valid = mytree.query(pos,k=1) # get cloestest points 
#   above idx_valid refers to the Ocean Grid points. Need to convert
#   back to the full grid                                                 
    idx_src=oindex[idx_valid]
    dist=dist*6371 # convert to km distance

#   set indices for points that aren't in the range to nan
    idx_src=np.where(dist<= dist_thresh,idx_src,-999).flatten()
    odist=dist[dist <=dist_thresh]
#   create an array of the input positions for the points found within range
    good_pts_pos=np.rad2deg(pos[dist.flatten() <= dist_thresh,:])
    glat=good_pts_pos[:,0]  # latitude longitudes of input points for which match has been found
    glon=good_pts_pos[:,1]
# convert linear indexi back to two dimensions 
#    pts_bad= np.where(idx_src == -999)
    pts_good = np.where(idx_src !=-999)
    good_indices= idx_src[pts_good]
#   convert the linear index back to two dimensions of the original grid
    index2d=np.unravel_index(good_indices, (ny,nx))  
#   extract the x and y coordinates for the index2d tuple 
    yindex=index2d[0] # 
    xindex=index2d[1]
    npts=xindex.shape[0]
#   by creating the new dimension "points" I can extract a series of points 
#   from the data array 
    xa=xr.DataArray(xindex.flatten(),dims='points')
    ya=xr.DataArray(yindex.flatten(),dims='points')
#    ta=xr.DataArray([ds.indexes['time'].get_loc(i,method='nearest') for i in time_array],dims='points')        
    ta=xr.DataArray(ds.indexes['time'].get_indexer(time_array,method='nearest'),dims='points')        
    ta=ta[pts_good] # subset time selector to only the points that are within the distance range
#   select the closest data points, by x,y and time coordinates
    h=ds[varnames].isel(x=xa,y=ya,time=ta) # this should be isel rather than sel
    good_ids=[id_list[i] for i in pts_good[0]]
#   this could be useful for slecting below... not used 
    h=h.assign_coords({"points":good_ids}) #questionable. 
#   at the orginal latitude and longitudes and distance to the dataset 
    ar_glat= xr.DataArray(glat,dims='points',attrs={"long_name":"Point Latitude"})
    ar_glon= xr.DataArray(glon,dims='points',attrs={"long_name":"Point Longitude"})
    ar_dist= xr.DataArray(odist,dims='points',attrs={"long_name":"Distance from closest grid point [km]"})
    h['point_lon']=ar_glon
    h['point_lat']=ar_glat
    h['dist']=ar_dist
    print(axutil.lonTo180(h.lon.compute().data))
    print(h.point_lon.data)
    
    print(h.lat.compute().data) # this generates the correct points
    print(h.point_lat.data)     # but subsampling in time gets weired. 
    # the h data set now how the variable sit as sit(time,points), all 
    # times for each point are selected. The problem now is that 
    # we want to select a different time for each point. 
    return h 

def write_points_csv(points):
    import csv 
    csv_file='test_points.csv'
    fieldnames = ['id', 'lat', 'lon','yymmdd']
    baseyear=2000
    month=1
    day=1
    yearInc=0
    with open(csv_file, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for id in points.keys():
            date=str(baseyear+yearInc)+str(month).zfill(2)+str(day).zfill(2)
            print(date)
            data=[id,points[id]['pos']['lat'],
                      points[id]['pos']['lon'],date]
            month=month+1 
            if month==12: 
                month=1
                yearInc=yearInc+1
            writer.writerow(data)
    print("Done writing test_points.csv")
def read_points_csv(): # smaple cvs reader 
    import csv 
    csv_file='test_points.csv'
    with open(csv_file, 'r', encoding='UTF8', newline='') as f:       
        csv_reader = csv.reader(f, delimiter=',')
        line = 0           
        points=dict()
        for row in csv_reader:
            if line == 0:
                print(f'Column names are {", ".join(row)}')
            else:                        
                points[row[0]]=dict(pos={"lat":float(row[1]),"lon":float(row[2])},date=row[3])    
            line += 1             
    return points

def make_points_data(): # generates some random point data for testing 
    import csv 
    ancil_file='/home/axel/idl-lib/axel/jz_model/data/liamas/liamas_static_scalar.nc'    
    ancil_data=xr.open_dataset(ancil_file)
        
    dlon=ancil_data.lon.data.flatten()
    dlat=ancil_data.lat.data.flatten()
    OceanMask=ancil_data.OceanMask.rename({"y_sn":"y","x_ew":"x"})
    omask=OceanMask.data.flatten()
    dlat=dlat[omask==1]
    dlon=dlon[omask==1]
    points=dict()
    for i in range(0,dlon.size,10000): 
        points[i+100]=dict(pos={"lat":dlat[i],"lon":dlon[i]})             
    return points
def write_test_points(): 
    points=make_points_data()
    test=write_points_csv(points)
    print("Done writing test Points")


def test_output_read(output_file,varnames):
    out=xr.open_dataset(output_file)
    pdata=out.points.data
    for n in list(pdata): 
        print(n)
        test=out[varnames].sel(points=n)
        if type(varnames) == str :
            print(test.data)
        else: 
            for var in list(varnames):  # prevent from iterating over letters
                print(test[var].data)
    test.close()
    return out

def plot_sample():     
    output_file='test_match_airem.heff.nc'
    out=xr.open_dataset(output_file)
    varname='heff'
#    varname='sit'
    pdata=out.points.data 
    for n in list(pdata): 
        print(n)
        test=out.sel(points=n)[varname]
        test.plot.line(label=n)
def fix_sea_ice_cdr(): 
    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20250930_V1.0.nc'
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
        cdr.to_netcdf('/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc')
    return
    
def scatter_plot(ds):
    import matplotlib.pyplot as plt
    import numpy as np
    obs_thick=ds.THICK_ICE_SNOW[:,0].data
    mod_thick=ds.sit.data
    fig=plt.scatter(obs_thick,mod_thick)
    plt.xlabel("Observed Thickness")
    plt.ylabel("Modeled Thickness")
    plt.title("PIOMAS 5e vs CDR MOSAIC")
    return fig  

def plot_map():
    from model_maps_arctic import arctic_map_outline
    import cartopy.crs as ccrs
    
    ax,fig,gl=arctic_map_outline()
    ds=xr.open_dataset("matched_cdr.nc")
#    ax.scatter(ds.lon,ds.lat,markersize=20, color='red',transform=ccrs.PlateCarree())   
    ax.scatter(ds.lon,ds.lat, edgecolors='red',facecolors='none',marker='o',transform=ccrs.PlateCarree(),s=2)
    ax.scatter(ds.point_lon,ds.point_lat,marker='+', color='blue',transform=ccrs.PlateCarree(),s=3)
def make_match_db(): 
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    logging.info("Start of extraction")
    t0=time.process_time()   
    ancil_file='/home/axel/idl-lib/axel/jz_model/data/liamas/liamas_static_scalar.nc'
    ancil_data=xr.open_dataset(ancil_file)
    input_dir="/data/axel0/model_results/2e/nc/"
#
    imvarname='heff'
    varnames=['sit'] # need to address naming scheme
    output_file= 'test_match_airem.heff.nc'
    in_pattern=input_dir+"/"+imvarname+".H[1-2][0-9][0-9][0-9].nc"
#    in_pattern=input_dir+"/"+imvarname+".H2000.nc"    
#    in_pattern=input_dir+"/"+imvarname+".H2020.nc"
#   read points data. Reads a file with locations to extract. (points.csv). 

    cdr_data=read_thickCDR()
    with ProgressBar(): 
        ds=xr.open_mfdataset(in_pattern,chunks=({"x":752,"y":240,"time":12*42}))
#        ds=ds.persist()
        ds=ds.chunk({'time':12*42}) # no idea if that helps
        if "Latitude" in list(ds.coords.keys()): # some of the heff files have Latitude/Longitude as coordinates                                                 # rename
            ds=ds.rename({"Longitude":"lon","Latitude":"lat"})
#        ds.load() # loads all of it into memory which works on my computer with this 
#       match the data in points with the data from the closest grid point         
        pmds = match_points(cdr_data,ds,varnames,ancil_data)
#       write the matched data to netcdf
        pmds.to_netcdf(output_file,mode='w')
        pmds.close()
        t1=time.process_time()
        logging.info("Finished writing %s in %5.2f minutes",output_file,(t1-t0)/60.)
#       print_points(mpoints)        
    out=test_output_read(output_file,varnames)
    model_points=out.rename({'points':'cdr_index'})
    out.close()
    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'
    cdr=xr.open_dataset(cdrfile) 
    # this version has a bad time variable, too long. 
    
    # select the points in the observation data that correspond to the 
    # matched points... This is so we don't have to pass all data into 
    # the points matching data base. 
    
    msel=model_points.cdr_index
    mcdr=cdr.sel(cdr_index=msel.cdr_index.data) 
#    out=model_points.rename({'points':'cdr_index'})
#    model_points=model_points.reset_coords(names='time')
    model_points=model_points.rename({'time':'model_time'})
    mcdr=mcdr.rename({'time':'obs_time'})
    obsAndModel=xr.merge([mcdr,model_points])
    model_time=obsAndModel.model_time
    obs_time=obsAndModel.obs_time
    diff_time=(model_time-obs_time).dt.s=days
#    model_time=pd.to_datetime(obsAndModel.model_time)
#    obs_time=pd.to_datetime(obsAndModel.obs_time)
#    .to_pydatetime()
    good_idx=np.where(np.abs(diff_day) < 15)
    obsAndModel=obsAndModel.isel(cdr_index=good_idx)
    obsAndModel=obsAndModel.reset_coords(names=['lat','lon'])
    obsAndModel.to_netcdf("matched_cdr.nc")
    
    obs_thick=obsAndModel.THICK_ICE_SNOW[:,0].data
    mod_thick=obsAndModel.sit.data
    obs_lat=obsAndModel.point_lat
    obs_lon=obsAndModel.point_lon
    mod_lat=obsAndModel.lat
    mod_lon=obsAndModel.lon
    
if __name__ == "__main__":
    make_match_db()
    #plot_map()
    ds=xr.open_dataset("matched_cdr.nc")
    scatter_plot(ds)
#    main()
    
    
    
    