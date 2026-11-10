#!/usr/bin/env python3
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
import glob
import numpy as np
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
import ax_im_utils as im
#import xoak
def preprocess_ObsAndModel(ds): 
    ds=ds.sortby('cdr_index')
    return ds 

def readMerged(model_id,gridName,sources=None,
               input_root="/home/axel/ax_python_lib/thickCDR/extractDir/"): 
    # reads merged cdr and extraction files for sources. 
    # not DONE fixed model and data 
#    model_id='2cfs1'
#    gridName='360X120'
    print("Reading data from ",input_root)
    # one-pass extraction file (extract_all) is used if it exists, otherwise the per-source merged files
    extract_files=sorted(glob.glob(input_root+gridName+"/"+model_id+"/"+model_id+".extract.*.nc"))
    if extract_files:
        logging.info("Reading one-pass extraction %s",extract_files[0])
        obsAndModel=xr.open_dataset(extract_files[0])
        if sources is not None:
            if not isinstance(sources,list): sources=[sources]
            obsAndModel=obsAndModel.isel(cdr_index=np.where(obsAndModel.SOURCE.isin(sources).values)[0])
        return obsAndModel
    files=select_source_files(model_id,gridName,sources=sources,
                              input_root= input_root)
    #infile= input_dir+'1c.merged.AIR-EM.heff.snow.nc'
    obsAndModel=xr.open_mfdataset(files,preprocess=preprocess_ObsAndModel)
    
#    subset=obsAndModel.where(obsAndModel.SOURCE==select_source,drop=True)
#    sources = list(set(subset.SOURCE.data.tolist()))
#    print(sources)   
#    title=model_id +":" + select_source 
    return obsAndModel

def read_thickCDR(source=None,cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'): 
#   reads CDR file. If source is specified a particular data source is extracted.  
    cdr_ds=xr.open_dataset(cdrfile)
    sources=np.unique(cdr_ds.SOURCE.values)
    if not source==None:         
#    cdr=cdr.where((cdr.SOURCE =='AIR-EM') & (cdr.PLATFORM=='2019MOSAIC'),drop=True)
#    cdr=cdr.where((cdr.SOURCE =='AIR-EM') | (cdr.SOURCE=='BGEP') 
#                  | (cdr.SOURCE=='IOS-EBS')
#                  | (cdr.SOURCE=='IOS-CHK'),drop=True)
        cdr_ds=cdr_ds.where(cdr_ds.SOURCE==source,drop=True)    
    fdate=cdr_ds.FDATE
  
    date=dt.datetime(1900,1,1)+pd.to_timedelta(fdate,unit='days')
    lat=cdr_ds.LAT.data
    lon=axutil.lonTo180(cdr_ds.LON.data)
    vardata=cdr_ds.THICK_ICE_SNOW[:,0].data # Get mean value... not thickness (ice + snow
    cdr_index=cdr_ds.INDEX.data.astype('int64')
    pos=np.column_stack((lat,lon))
#   create a dictionary with the data    
    cdr_data={'cdr_index':cdr_index,'date':date,'pos':np.deg2rad(pos),'thick':vardata}
    return cdr_data,cdr_ds

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

def match_points(cdr_data,ds,varnames,ancil_data,dist_thresh=10.,source=None):
#   finds closest points in model data provide in ds and points provided in cdr_data
#   points further away than dist_thresh are not matched.     

    if "y_sn" in list(ds.coords.keys()): # some of the heff files have Latitude/Longitude as coordinates                                                 # rename
        OceanMask=ancil_data.OceanMask.rename({"y_sn":"y","x_ew":"x"})
    else:
        OceanMask=ancil_data.OceanMask
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
    if good_pts_pos.size == 0: 
        logging.warning("No good points within the distance %d for source %s",dist_thresh,source)
        return  -1 
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
#    print(axutil.lonTo180(h.lon.compute().data))
#    print(h.point_lon.data)
    
#    print(h.lat.compute().data) # this generates the correct points
#    print(h.point_lat.data)     # but subsampling in time gets weired. 
    # the h data set now how the variable sit as sit(time,points), all 
    # times for each point are selected. The problem now is that 
    # we want to select a different time for each point. 
    ds.close()
    return h 

def write_points_csv(points):
    # helper program to write sample test data base
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
def ax_draft2thick(draft,snow,month=None,swe=False):
#   converts ice thickness to draft using Rothrock et al. 2008
#   thick= sea ice thickness [m] 
#   draft =sea ice draft in [m]
   rho_ice = 928                # ; [kg/m3] ROn
   rho_water=1027.
   rho_snow = rho_ice/3.0
   rho_snow_t=np.array([308,318,327,328,329,348,383,219,246,271,291,301]) 
#  if the keyword month is set then the snow density is set to vary by
#  nth.
   if isinstance(month,np.ndarray): 
      rho_snow=np.float32(rho_snow_t[month-1])
#  flotation: rho_water*draft = rho_ice*thick + rho_snow*snow_depth (= rho_fw*SWE)
   rho_fw = 1000.
   if swe:
      thick=rho_water/rho_ice * draft - rho_fw/rho_ice * snow
   else:
      thick=rho_water/rho_ice * draft - rho_snow/rho_ice * snow ;
   return thick
def ax_thick2draft(thick,snow,month=None,swe=False):
#   flotation: draft = (rho_ice*thick + rho_snow*snow_depth)/rho_water, the exact inverse of ax_draft2thick.
#   snow is a DEPTH by default: model nc sisnthick is a depth (the converter multiplies Zhang's SWE output by 3,
#   im/ax_im_varinfo.py); swe=True takes snow water equivalent instead.
#   Fixed 2026-11-09: the depth path multiplied the depth by rho_fw/rho_snow as if it were SWE, which made snow weigh
#   9-19x too much (2 m ice, 0.25 m snow: draft 0.6-1.0 m too deep); the swe path used rho_ice for the snow load (~5 mm).
    rho_ice = 928
    rho_water=1027.
    rho_fw = 1000.
    rho_snow = rho_ice/3.0
    rho_snow_t=np.array([308,318,327,328,329,348,383,219,246,271,291,301])
    if isinstance(month,np.ndarray):
        rho_snow=np.float32(rho_snow_t[month-1])
    if swe:
       draft= (rho_ice*thick + rho_fw*snow)/rho_water
    else:
       draft= (rho_ice*thick + rho_snow*snow)/rho_water
    return draft
    
def make_points_data(): # generates some random point data for testing
    # helper function 
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


def read_match_pointDB_file(match_pointDB_file,varnames,verbose=False):
    # reads a netcdf file of point locatiosn/times matched to the corresponding closest model grid cells/times
    out=xr.open_dataset(match_pointDB_file)
    pdata=out.points.data
    if verbose:
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
def draft2thick(obsAndModel):
    # note: no submarine bias correction here, use toThick() for that
    draft_idx=np.where(~np.isnan(obsAndModel.DRAFT[:,0])) # get indices for draft var
    draft=obsAndModel.DRAFT[draft_idx,0]
    month=obsAndModel.obs_time[draft_idx].dt.month.values
    snow=obsAndModel.sisnthick[draft_idx].data
    thick= ax_draft2thick(draft,snow,month=month)   
    obsAndModel.THICK[draft_idx,0]=thick
    return  obsAndModel

def thick2draft(obsAndModel):
# converts ice thickness to draft using Rothrock et al. 2008
#   thick=thickness in meters,draft in meters, snow thickness in m

    idx=np.where(~np.isnan(obsAndModel.THICK[:,0])) # get indices for draft var
    thick=obsAndModel.THICK[idx,0]
    month=obsAndModel.obs_time[idx].dt.month.values
    snow=obsAndModel.sisnthick[idx].data
    draft=ax_thick2draft(thick,snow,month=month)
    obsAndModel.DRAFT[idx,0]=draft
    return obsAndModel

def toThick(obsAndModel):
    # compute thickness, draft and THCKE_ICE_SNOW for sources that don't have them
    # works on a copy: calling it twice on the same data set used to apply the submarine bias twice
    if obsAndModel.attrs.get('toThick_applied',False):
        logging.info("toThick already applied, returning data unchanged")
        return obsAndModel
    obsAndModel=obsAndModel.copy(deep=True)
    obsAndModel.attrs['toThick_applied']=1 # int, so it survives to_netcdf
    month=obsAndModel.obs_time.dt.month.values
    snow=obsAndModel.sisnthick.data
    
    thick_idx=np.where(~np.isnan(obsAndModel.THICK[:,0]))[0]# get indices for sources with thickness
    draft_idx=np.where(~np.isnan(obsAndModel.DRAFT[:,0]))[0] # get indices for sources with draft
    iAs_idx=np.where(~np.isnan(obsAndModel.THICK_ICE_SNOW[:,0]))[0] # get indices for sources with draft
    
    sub_idx=np.where(obsAndModel.SOURCE.isin(['USSUB-DG','USSUB-AN','UKSUB-AN','UKSUB-DG','USSUB-HI']))
   
    if  len(sub_idx[0]) >0 : 
        print(len(sub_idx[0]))
        #old_draft=obsAndModel.isel(cdr_index=sub_idx[0]).DRAFT[:,0]
        logging.info("Removing Submarine Bias")
        sub_bias=-0.29
        print("before bias removal: :",obsAndModel.isel(cdr_index=sub_idx[0]).DRAFT[:,0].mean().data)
        # .loc is needed for assignment
        obsAndModel['DRAFT'].loc[
            dict(cdr_index=obsAndModel.cdr_index[sub_idx],
                 **{'Mean Max Min Sigma':0})]+=sub_bias
       
        print("After bias removal: :",obsAndModel.isel(cdr_index=sub_idx[0]).DRAFT[:,0].mean().data)
       
    draft=obsAndModel.DRAFT[:,0]
    
    thick=obsAndModel.THICK[:,0]
    thick_ice_snow=obsAndModel.THICK_ICE_SNOW[:,0]
    
    draft_tmp=ax_thick2draft(thick,snow,month=month) # nan where no thickness
    thick_tmp=ax_draft2thick(draft,snow,month=month) # nan where no draft
    
#   compute draft where thickness is provided
    obsAndModel.DRAFT[thick_idx,0]=draft_tmp[thick_idx]
#   compute thickness where draft is provided 
    obsAndModel.THICK[draft_idx,0]=thick_tmp[draft_idx]
    
    
#   compute thickness for AIR-EM data     
    obsAndModel.THICK[iAs_idx,0]= \
        obsAndModel.THICK_ICE_SNOW[iAs_idx,0]-snow[iAs_idx]

#   compute draft for AIR-EM data     
    obsAndModel.DRAFT[iAs_idx,0]= \
        ax_thick2draft(thick_ice_snow[iAs_idx]-snow[iAs_idx],snow[iAs_idx],month=month[iAs_idx])
    
    
#   concatenate draft_idx and thick_idx and for those locations compute THICK_SNOW_ICE
#   using either measured or computed thickness 

    c=np.append(draft_idx,thick_idx)
    print(c.size)
    obsAndModel.THICK_ICE_SNOW[c,0]= obsAndModel.THICK[c,0] + snow[c]
    logging.info("Number of missing Draft values : %d",obsAndModel.DRAFT[:,0].isnull().sum().item())
    logging.info("Number of missing THICK values : %d",obsAndModel.THICK[:,0].isnull().sum().item())
    logging.info("Number of missing THICK_ICE_SNOW values : %d",obsAndModel.THICK_ICE_SNOW[:,0].isnull().sum().item())
    
    missing_mask=obsAndModel.THICK[:,0].isnull()
    missing_coords = obsAndModel.where(missing_mask, drop=True).coords

    print("\nCoordinates of Missing Values:")
    for dim, coord in missing_coords.items():
        print(f"{dim}: {coord.values}")
    return obsAndModel

def test_cdr2thickness():
    cdrfile='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'
    cdr=xr.open_dataset(cdrfile)
    cdr=cdr.where(cdr.SOURCE=='BGEP',drop=True)
    draft_idx=np.where(~np.isnan(cdr.DRAFT[:,0] )) # get indices for draft var
    draft=cdr.DRAFT[draft_idx,0]
    month=cdr.time.dt.month.values
    snow=np.full_like(draft,0.1) # default snow thickness of 0.1 m 
    thick= ax_draft2thick(draft,snow,month=month) 
    cdr.THICK[draft_idx,0]=thick
    
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
def scatter_plot(ds,model_name,title=None,max_thick=10.,draft=False):
    # got to figure out whether something is draft, thickness or thickiness and draft
    
    import matplotlib.pyplot as plt
    import matplotlib.lines as mlines
    import matplotlib.transforms as mtransforms
    import statsmodels.api as sm
    from sklearn.metrics import root_mean_squared_error
    from sklearn.metrics import mean_squared_error
    import numpy as np
    if not isinstance(ds,xr.Dataset): # if ds is not a dataset open it as a file
        ds=xr.open_dataset(ds)
    ds.load()
#    obs_thick=ds.THICK_ICE_SNOW[:,0].data
#    mod_thick=ds.sit.data + ds.sisnthick.data
    if draft==False: 
        obs_thick=ds.THICK[:,0].data
        mod_thick=ds.sit.data
    else:
        # this uses draft as a comparison. Model draft computed from snow and assumed densities. 
        month=ds.obs_time.dt.month.values
        snow=ds.sisnthick.data
        obs_thick=ds.DRAFT[:,0].data        
        mod_thick= ax_thick2draft(ds.sit.data,snow,month=month) 
#   obs_thick=ds.DRAFT[:,0].data
   
    good_idx=np.where(np.isfinite(obs_thick) & np.isfinite(mod_thick))[0]
    mod_thick=mod_thick[good_idx]
    obs_thick=obs_thick[good_idx]
    mhas_nan = np.any(np.isnan(mod_thick))
    ohas_nan = np.any(np.isnan(obs_thick))
    print("Model has Nan: " , mhas_nan)
    print("Observations have Nan:", ohas_nan)
    
    model = sm.OLS(mod_thick, sm.add_constant(obs_thick),missing='drop')
    results = model.fit()
    print(results.params)
    print(results.summary())
    fig,ax=plt.subplots()
    ax.scatter(obs_thick,mod_thick)
    ax.set_xlim(0,max_thick)
    ax.set_ylim(0,max_thick)
    plt.xlabel("Observed Thickness [m]")
    plt.ylabel("Modeled Thickness [m]")
    if title==None:
        title=model_name   
    plt.title(title)
    line = mlines.Line2D([0, 1], [0, 1], color='red')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    
    b, m = results.params
    label= f'$y = {m:.1f}x {b:+.1f}$'
    ax.axline(xy1=(0, b), slope=m, label=label,color='black')
    m2err=mean_squared_error(obs_thick,mod_thick)
    rmse=root_mean_squared_error(obs_thick,mod_thick)
    meane=np.mean(obs_thick-mod_thick)
#    ax.legend(fontsize=12)
    annotation_text = (f'Obs-Mod (mean/rms): {meane:.2f}/{rmse:.2f}\n'
                       f'R² = {results.rsquared:.2f}\n'
                       f'y = {m:.2f}x + {b:.2f}\n')
    
    ax.annotate(annotation_text, xy=(0.05, 0.80), xycoords='axes fraction',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="None", lw=1))
    return fig,ax,results
def source_stats(ds,draft=False,ice_snow=False):
    # table of model vs observation statistics by source (+ ALL), after toThick()
    # draft=True compares draft (model draft from sit, snow and month as in scatter_plot)
    # ice_snow=True compares ice+snow thickness (model sit+sisnthick), e.g. for AIR-EM
    if ice_snow:
        obs=ds.THICK_ICE_SNOW[:,0].values
        mod=ds.sit.values+ds.sisnthick.values
    elif draft:
        obs=ds.DRAFT[:,0].values
        mod=ax_thick2draft(ds.sit.values,ds.sisnthick.values,month=ds.obs_time.dt.month.values)
    else:
        obs=ds.THICK[:,0].values
        mod=ds.sit.values
    df=pd.DataFrame({'source':ds.SOURCE.values,'obs':obs,'mod':mod})
    df=df[np.isfinite(df.obs) & np.isfinite(df['mod'])]
    def _stats(d):
        diff=d['mod']-d.obs
        return pd.Series({'n':len(d),'obs_mean':d.obs.mean(),'mod_mean':d['mod'].mean(),
                          'bias(mod-obs)':diff.mean(),'rmse':np.sqrt((diff**2).mean()),
                          'r':d.obs.corr(d['mod']) if len(d)>2 else np.nan})
    table=pd.DataFrame({src:_stats(d) for src,d in df.groupby('source')}).T
    table.loc['ALL']=_stats(df)
    table['n']=table['n'].astype(int)
    return table

def plot_histograms(ds,model_name, title=None): 
    import numpy as np
    import matplotlib.pyplot as plt
    if not isinstance(ds,xr.Dataset): # if ds is not a dataset open it as a file
        ds=xr.open_dataset(ds)
    ds.load()
#    obs_thick=ds.THICK_ICE_SNOW[:,0] 
    obs_thick=ds.THICK[:,0]
    mod_thick=ds.sit.data
    fig,(ax0,ax1)=plt.subplots(nrows=2,ncols=1)
    ax0.hist([obs_thick,mod_thick], bins= 50, range=[0,8], alpha=0.5, label=['Obs', 'Mod'])
    ax0.legend(loc='upper right')
    ax0.set_title('Thickness Distributions\n'+title)
    diff=obs_thick-mod_thick
    ax1.hist(diff, bins= 50, range=[-3,3], alpha=0.5, label=['Obs-Mod'])
    ax1.legend(loc='upper right')
    ax1.set_title('Difference (Obs-Mod) ' + title)
    plt.show()
    return fig,ax0,ax1
def scatter_plot_sns(): 
    import seaborn as sns

#create scatterplot with regression line and confidence interval lines
    sns.regplot(x, y)
    
def select_source_files(model_id,gridName,
                        input_root="/home/axel/ax_python_lib/thickCDR/extractDir/"
                        ,sources=None):
    print(sources)
    file_list=[]
    input_dir=input_root+gridName+"/"+model_id +"/"
#    print(sources)
    if sources==None: 
        pattern=input_dir+model_id+".merged*.nc" 
#        print(pattern)
        files=glob.glob(pattern)
        return files
    if not isinstance(sources,list): 
        sources=[sources]
    for source in sources: 
        print(source)
        pattern=input_dir+model_id+".merged."+source+"*.nc"
        files=glob.glob(pattern)
        file_list=file_list+files
    return file_list


        
def plot_location_map(ds):
    # sample to plot locations on a basic map 
    from model_maps_arctic import arctic_map_outline
    import cartopy.crs as ccrs 
    ax,fig,gl=arctic_map_outline()
    if isinstance(ds,xr.Dataset)==False :
        ds=xr.open_dataset(ds)
    ax.scatter(ds.lon,ds.lat, edgecolors='red',facecolors='none',marker='o',transform=ccrs.PlateCarree(),s=2)
    ax.scatter(ds.point_lon,ds.point_lat,marker='+', color='blue',transform=ccrs.PlateCarree(),s=3)
    plt.show()
def make_input_file_list(model_path,file_varnames=['hiday','snowday']):
    # creates a list of input file names for the model 
    # should modify to filter by year range
    file_list=[]
    for var in file_varnames: 
#        print(var)
        pattern= model_path+var+'.H*.nc'
#        print(pattern)
        f=glob.glob(pattern)
        file_list=file_list+f
#    print(file_list)
    return file_list

def make_match_points_db(cdr_data=None,cdr_file=None,
                         model_path=None,gridName=None,
                         file_varnames=['hiday','snowday'],
                         ncdf_varnames=['sit','sisnthidef'],
                         match_pointDB_file=None,
                         id=None,source=None,dist_thresh=10.,ds=None): 
    # main driver program that creates the matchded data base
    
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    logging.info("Start of extraction")
    t0=time.process_time()   
    sc,vc,_=im.getAncilData(gridName=gridName)
    g=im.gridInfo(gridName) 
    nx=g.nx; ny=g.ny
    nt=42 # not clear if this is necessary.. This will depend on number of years to match. 
    if not isinstance(cdr_data,dict):  
        logging.info("reading CDR data")
        cdr_data,_=read_thickCDR(source) # may want to do some filtering here. 
    if not isinstance(ds,xr.Dataset):   # load the model_data set if this isn't done 
        logging.info("Model Data set will be loaded  %s", source )
        ds=load_model_data(model_path,file_varnames,nx,ny,nt)
    #       match the data in points with the data from the closest grid point         
    with ProgressBar(): 
        logging.info("Starting Matching for :  %s",source)
        pmds = match_points(cdr_data,ds,ncdf_varnames,sc,dist_thresh=dist_thresh,source=source)
        if not isinstance(pmds,xr.Dataset):
            logging.warning("No good points found for source  %s", source )
            return -1 
        else: 
            pmds.attrs['id']=id
#   write the matched data to netcdf
            pmds.to_netcdf(match_pointDB_file,mode='w')
            ds.close()
            t1=time.process_time()
            logging.info("Finished writing %s in %5.2f minutes",match_pointDB_file,(t1-t0)/60.)
    return pmds
def load_model_data(model_path,file_varnames,nx,ny,nt):
   input_files=make_input_file_list(model_path,file_varnames)
   xchunk=nx
   ychunk=ny
   tchunk=12*nt 
   with ProgressBar(): 
       ds=xr.open_mfdataset(input_files,chunks=({"x":xchunk,"y":ychunk,"time":tchunk}))
#        ds=ds.persist()
       ds=ds.chunk({'time':tchunk}) # no idea if that helps, would seem like we are doign this already above
       if "Latitude" in list(ds.coords.keys()): # some of the heff files have Latitude/Longitude as coordinates                                                 # rename
           ds=ds.rename({"Longitude":"lon","Latitude":"lat"})
   if file_varnames[0] in ['heff','snow']: # monthly files are stamped on the 1st, move to mid-month
       ds['time']=ds['time'] + pd.to_timedelta(ds.time.dt.days_in_month.values/2.,unit='D')
       logging.info("Monthly data: model time shifted to mid-month")
   logging.info("Done reading input model_data")
   return ds

def merge_pointDB_withCDR(cdr_ds=None,cdr_file='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc',
                          match_pointDB_file='2e.matchDB_airem.heff.nc',merged_cdr_model_file='2e.heff.airem.merged_cdr.nc',
                          ncdf_varnames=['hiday','snowday'],source=None):
    # merges extracted points back into the thickness CDR data structure so that we have access to all the variables
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    logging.info("Start of merging")
    t0=time.process_time()
    model_points=read_match_pointDB_file(match_pointDB_file,ncdf_varnames)
    model_points=model_points.rename({'points':'cdr_index'})
    
    if not isinstance(cdr_ds,xr.Dataset):
        cdr_ds=xr.open_dataset(cdr_file)
        if not source == None:
            cdr_ds=cdr_ds.where(cdr_ds.SOURCE==source,drop=True)
        logging.info("Reading CDR frome file")    
    # select the points in the observation data that correspond to the 
    # matched points... This is so we don't have to pass all data into 
    # the points matching data base. 

    msel=model_points.cdr_index
    mcdr=cdr_ds.sel(cdr_index=msel.cdr_index.data) 
    mcdr.attrs['id']=model_points.attrs['id']
#    out=model_points.rename({'points':'cdr_index'})
#    model_points=model_points.reset_coords(names='time')
    model_points=model_points.rename({'time':'model_time'})
    mcdr=mcdr.rename({'time':'obs_time'})
    obsAndModel=xr.merge([mcdr,model_points])
    model_time=obsAndModel.model_time
    obs_time=obsAndModel.obs_time
    diff_time=(model_time-obs_time).dt.days

    good_idx=np.where(np.abs(diff_time) <= 15)[0] # tuple with one array per dimension hence select [0]
    #print("number of good elements: ", good_idx.size)
    if good_idx.size > 0 : 
        obsAndModel=obsAndModel.isel(cdr_index=good_idx) # is that really true
        obsAndModel=obsAndModel.reset_coords(names=['lat','lon'])
        obsAndModel.to_netcdf(merged_cdr_model_file)  
        t1=time.process_time()
        logging.info("Finished writing %s in %5.2f minutes",merged_cdr_model_file,(t1-t0)/60.)
    else:
        logging.info("No matched data for %s : ", source)
    model_points.close()
    mcdr.close()
    cdr_ds.close()
    return obsAndModel

# ---------------------------------------------------------------------------
# One-pass extraction: all CDR sources matched against the model in a single
# pass and written to one file per model run
#   <output_path>/<model_id>.extract.<var0>.<var1>.nc
# Same variables as the per-source *.merged.* files, so readMerged/toThick/
# scatter_plot work unchanged.
# Speed: one BallTree query for all points; then for each model year-file
# only the time steps that have observations are read and indexed with numpy
# (no dask point indexing, which was the slow part).
# ---------------------------------------------------------------------------
EXCLUDE_SOURCES=['AWI-WS','USSUB-HI']

def model_file_times(files,monthly=False):
    # time axis of each yearly model file, monthly files shifted to mid-month
    times=[];file_id=[];local_t=[]
    for i,f in enumerate(files):
        with xr.open_dataset(f) as ds:
            t=pd.DatetimeIndex(ds.time.values)
        if monthly:
            t=t+pd.to_timedelta(t.days_in_month/2.,unit='D')
        times.append(t.values);file_id.append(np.full(t.size,i));local_t.append(np.arange(t.size))
    times=np.concatenate(times);file_id=np.concatenate(file_id);local_t=np.concatenate(local_t)
    order=np.argsort(times,kind='stable')
    return pd.DatetimeIndex(times[order]),file_id[order],local_t[order]

def extract_points(files,ncvar,obs_time,iy,ix,monthly=False,max_days=15):
    # model values at grid cells (iy,ix) at the model time closest to obs_time
    # returns values, model_time, valid (obs within max_days of a model time), variable attrs
    mtimes,file_id,local_t=model_file_times(files,monthly=monthly)
    k=mtimes.get_indexer(pd.DatetimeIndex(obs_time),method='nearest')
    model_time=mtimes.values[k]
    # same criterion as merge_pointDB_withCDR: abs(diff.dt.days) <= max_days
    diff_days=pd.TimedeltaIndex(model_time-obs_time).days
    valid=(k>=0) & (np.abs(diff_days)<=max_days)
    values=np.full(obs_time.size,np.nan,dtype='float32')
    fk=file_id[k];tk=local_t[k]
    attrs={}
    for i in np.unique(fk[valid]):
        sel=np.where(valid & (fk==i))[0]
        tsel,tinv=np.unique(tk[sel],return_inverse=True)
        with xr.open_dataset(files[i]) as ds:
            data=ds[ncvar].isel(time=tsel).values # only time steps with observations
            attrs=dict(ds[ncvar].attrs)
        values[sel]=data[tinv,iy[sel],ix[sel]]
    attrs.pop('coordinates',None)
    return values,model_time,valid,attrs

def extract_all(model_id,model_path,gridName,file_varnames,ncdf_varnames=['sit','sisnthick'],
                cdr_file='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc',
                dist_thresh=10.,output_file=None,exclude_sources=EXCLUDE_SOURCES,max_days=15):
    # matches all CDR points to the closest ocean grid cell (within dist_thresh km)
    # and closest model time (within max_days) and returns/writes CDR + model values
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    t0=time.time()
    cdr_ds=xr.open_dataset(cdr_file).load() # ~150MB; lazy fancy indexing of the file is very slow
    if exclude_sources:
        cdr_ds=cdr_ds.isel(cdr_index=np.where(~cdr_ds.SOURCE.isin(exclude_sources).values)[0])
    # matching uses FDATE (days since 1900-01-01), like read_thickCDR
    obs_time=np.datetime64('1900-01-01','ns')+pd.to_timedelta(cdr_ds.FDATE.values,unit='D').values
    plat=cdr_ds.LAT.values.astype('float64')
    plon=axutil.lonTo180(cdr_ds.LON.values).astype('float64')
    logging.info("CDR: %d points",plat.size)

    files={v:sorted(glob.glob(model_path+v+'.H*.nc')) for v in file_varnames}
    for v in file_varnames:
        if not files[v]:
            raise FileNotFoundError(f"no {v}.H*.nc files in {model_path}")
    with xr.open_dataset(files[file_varnames[0]][0]) as ds:
        if "Latitude" in ds.coords: ds=ds.rename({"Longitude":"lon","Latitude":"lat"})
        mlat=ds.lat.values; mlon=ds.lon.values
        has_data=np.isfinite(ds[ncdf_varnames[0]].values).any(axis=0).flatten()
    ny,nx=mlat.shape
    sc,_,_=im.getAncilData(gridName=gridName)
    omask=sc.OceanMask.values.reshape(ny,nx).flatten()==1
    # only match to cells that have model data: model output converted with an older
    # ancil mask can be NaN in cells the current OceanMask calls ocean (912X672: 9093 coastal cells)
    if (omask & ~has_data).sum()>0:
        logging.warning("%d OceanMask cells have no model data and are not used for matching",(omask & ~has_data).sum())
    omask=omask & has_data
    oindex=np.arange(nx*ny)[omask]
    latlon=np.deg2rad(np.column_stack([mlat.flatten()[omask],axutil.lonTo180(mlon.flatten()[omask])]))
    tree=BallTree(latlon,metric='haversine')
    dist,idx=tree.query(np.deg2rad(np.column_stack([plat,plon])),k=1)
    dist=dist[:,0]*6371.;idx=oindex[idx[:,0]]
    near=dist<=dist_thresh
    logging.info("%d points within %.1f km of an ocean grid cell",near.sum(),dist_thresh)
    iy,ix=np.unravel_index(idx[near],(ny,nx))
    t_near=obs_time[near]

    out={};valid_all=None;model_time=None
    for v,ncvar in zip(file_varnames,ncdf_varnames):
        logging.info("Extracting %s (%s) from %d files",ncvar,v,len(files[v]))
        vals,mt,valid,attrs=extract_points(files[v],ncvar,t_near,iy,ix,monthly=v in ['heff','snow'],max_days=max_days)
        out[ncvar]=(vals,attrs)
        if model_time is None: # model_time and time window from the thickness variable
            model_time=mt;valid_all=valid
    keep=np.where(near)[0][valid_all]
    sub=np.where(valid_all)[0]
    res=cdr_ds.isel(cdr_index=keep).rename({'time':'obs_time'})
    for ncvar,(vals,attrs) in out.items():
        res[ncvar]=xr.DataArray(vals[sub],dims='cdr_index',attrs=attrs)
    res['model_time']=xr.DataArray(model_time[sub],dims='cdr_index')
    res['lon']=xr.DataArray(mlon[iy[sub],ix[sub]],dims='cdr_index')
    res['lat']=xr.DataArray(mlat[iy[sub],ix[sub]],dims='cdr_index')
    res['point_lon']=xr.DataArray(plon[near][sub].astype('float32'),dims='cdr_index',attrs={"long_name":"Point Longitude"})
    res['point_lat']=xr.DataArray(plat[near][sub].astype('float32'),dims='cdr_index',attrs={"long_name":"Point Latitude"})
    res['dist']=xr.DataArray(dist[near][sub],dims='cdr_index',attrs={"long_name":"Distance from closest grid point [km]"})
    res=res.sortby('cdr_index')
    res.attrs.update({'id':model_id,'model_path':model_path,'gridName':gridName,
                      'file_varnames':' '.join(file_varnames),'dist_thresh_km':dist_thresh,
                      'max_days':max_days,'cdr_file':cdr_file,
                      'excluded_sources':' '.join(exclude_sources or [])})
    logging.info("%d points matched in space and time, %.1f s",res.sizes['cdr_index'],time.time()-t0)
    if output_file:
        res.to_netcdf(output_file)
        logging.info("Wrote %s",output_file)
    cdr_ds.close()
    return res

# model runs that have been compared to the CDR. model_path defaults to
# /data/axel0/model_results/<model_group>/<model_id>/nc/
# file_varnames: ['heff','snow'] = monthly, ['hiday','snowday'] = daily
RUNS={
    '1c':   dict(model='CAPMAS',gridName='912X672',model_group='912X672',dist_thresh=10.,nt=44,file_varnames=['heff','snow']),
    '2c':   dict(model='CAPMAS',gridName='912X672',model_group='912X672',dist_thresh=10.,nt=21,file_varnames=['heff','snow']),
    '2g':   dict(model='BIOMAS',gridName='752X240',model_group='752X240',dist_thresh=10.,nt=76,file_varnames=['hiday','snowday']),
    '1e':   dict(model='BIOMAS',gridName='752X240',model_group='752X240',dist_thresh=25.,nt=44,file_varnames=['heff','snow']), # no daily snow
    '2cfs1':dict(model='PIOMAS',gridName='360X120',model_group='360X120',dist_thresh=25.,nt=46,file_varnames=['hiday','snowday']),
    '2sstC':dict(model='PIOMAS',gridName='360X120',model_group='360X120',dist_thresh=25.,nt=47,file_varnames=['hiday','snowday']), # public PIOMAS
    '3sst': dict(model='PIOMAS',gridName='360X120',model_group='360X120',dist_thresh=25.,nt=46,file_varnames=['hiday','snowday']), # no snowday nc as of 2026-11-02
    # GIOMAS: Arctic cells are ~27-33 km (median), up to ~45 km, so a larger threshold than 360X120
    '2cfs': dict(model='GIOMAS',gridName='360X276',model_group='360X276',dist_thresh=35.,nt=47,file_varnames=['hiday','snowday']),
    }

def driver(model_id='1c',output_root="/home/axel/ax_python_lib/thickCDR/extractDir/",
           cdr_file='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'): 
    # main driver program. Add new model runs to RUNS above.
    # older CDR version: '/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20190920_V3.0.nc'
    # with output_root=".../extractDir_2019v/"
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    run=RUNS[model_id]
    gridName=run['gridName']
    model_group=run['model_group']
    dist_thresh=run['dist_thresh']
    nt=run['nt']
    file_varnames=run['file_varnames']
    g=im.gridInfo(gridName) 
    nx=g.nx; ny=g.ny
    model_path=run.get('model_path',"/data/axel0/model_results/"+model_group+"/"+model_id+"/nc/")
#   this is the directory where the output is located 
    output_path=output_root+ "/"+gridName+"/"+model_id+"/"
    if not (os.path.isdir(output_path)): 
        logging.info("Extraction Directory: %s doesn't exists will be made",output_path)
        os.makedirs(output_path,exist_ok=True)

#   Below is constant for thickness extractions 
    ncdf_varnames=['sit','sisnthick']
#   open the CDR file and get all the sources 
    cdr=xr.open_dataset(cdr_file)
    sources=np.unique(cdr.SOURCE.values).tolist()
    cdr.close()
#    sources=['BGEP','CryoSat-AWI','IOS-CHK','IOS-EBS','IceBridge-QL','IceBridge-V2','NPEO']
#    sources=['CryoSat-AWI']
    sources.remove('AWI-WS')
    sources.remove('USSUB-HI')
    print(sources)
    match=True
    merge=True
#   sources=sources[0:2]
#    sources=['CryoSat-AWI']
    for source in sources: 
        print(source)
        id= model_id +"_"  + source # name used for identifcation inside of file 
        cdr_data,cdr_ds=read_thickCDR(source=source,cdrfile=cdr_file) 
        logging.info("Done reading cdr_data for %s: ",source)
#        cdr_ds=cdr.where(cdr.SOURCE==source,drop=True)
                    
        match_pointDB_file=output_path + model_id+'.matchDB.'+source+"." +file_varnames[0]+"."+file_varnames[1]+'.nc'
        merged_cdr_model_file=output_path + model_id+'.merged.'+source+'.'+file_varnames[0]+"."+file_varnames[1]+'.nc'
        model_ds=load_model_data(model_path,file_varnames,nx,ny,nt)
        if match:
            pmds=make_match_points_db(cdr_data=cdr_data,cdr_file=cdr_file,
                                      model_path=model_path,
                                      gridName=gridName,file_varnames=file_varnames,
                                      ncdf_varnames=ncdf_varnames,
                                      match_pointDB_file=match_pointDB_file,
                                      id=id,source=source,
                                      dist_thresh=dist_thresh,ds=model_ds)    
            
            if not isinstance(pmds,xr.Dataset):
                logging.info("skipping: %s ",source)
                continue
            pmds.close()
        if merge:
            obsAndModel=merge_pointDB_withCDR(cdr_ds=cdr_ds,cdr_file=cdr_file,match_pointDB_file=match_pointDB_file,
                                             merged_cdr_model_file=merged_cdr_model_file,
                                             ncdf_varnames=ncdf_varnames,source=source)
            cdr_ds.close()
    return 
    
def driver_onepass(model_id='1c',output_root="/home/axel/ax_python_lib/thickCDR/extractDir/",
                   cdr_file='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'):
    # one-pass alternative to driver(): writes <output_root>/<grid>/<model_id>/<model_id>.extract.<var0>.<var1>.nc
    run=RUNS[model_id]
    model_path=run.get('model_path',"/data/axel0/model_results/"+run['model_group']+"/"+model_id+"/nc/")
    output_path=output_root+run['gridName']+"/"+model_id+"/"
    os.makedirs(output_path,exist_ok=True)
    output_file=output_path+model_id+'.extract.'+'.'.join(run['file_varnames'])+'.nc'
    return extract_all(model_id,model_path,run['gridName'],run['file_varnames'],
                       cdr_file=cdr_file,dist_thresh=run['dist_thresh'],output_file=output_file)

if __name__ == "__main__":
    # python match_thickCDR_model.py <model_id> [--twostep]
    ap=argparse.ArgumentParser()
    ap.add_argument('model_id',choices=list(RUNS.keys()))
    ap.add_argument('--twostep',action='store_true',help='old per-source matchDB + merged files')
    args=ap.parse_args()
    if args.twostep:
        driver(args.model_id)
    else:
        driver_onepass(args.model_id)
#    subset=readMerged(source='BGEP')
#    test=toThick(subset)
    
    
    