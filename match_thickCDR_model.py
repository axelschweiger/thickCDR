#!/usr/bin/env pangeo312_mamba
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
 
    files=select_source_files(model_id,gridName,sources=sources,
                              input_root= "/home/axel/ax_python_lib/thickCDR/extractDir/")
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

def match_points(cdr_data,ds,varnames,ancil_data,dist_thresh=10.):
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
   if swe: 
      thick=rho_water/rho_ice * draft - snow  
   else:
      thick=rho_water/rho_ice * draft - rho_snow/rho_ice * snow ; 
   return thick
def ax_thick2draft(thick,snow,month=None,swe=False): 

#   if keyword set swe then convert swater equivalent to snow depth
#   input netcdf files have converted SWE to snow depth by multiplying 
#   by a scaler of 3. (ax_im_varinfo.py)
    rho_ice = 928  
    rho_water=1027.
    rho_fw = 1000. 
    rho_snow = rho_ice/3.0
    rho_snow_t=np.array([308,318,327,328,329,348,383,219,246,271,291,301]) 
    if isinstance(month,np.ndarray): 
        rho_snow=np.float32(rho_snow_t[month-1])
    # if swe is set the use as snow water equivalent
    if swe: 
       draft= (thick + snow) * rho_ice/rho_water
    else:
       stemp= snow  * rho_fw/rho_snow  # convert to snow depth
       draft= (thick + stemp)* rho_ice/rho_water
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
    snow=obsAndModel.sisnthick[draft_idx].data
    draft=ax_thick2draft(thick,snow,month=month)
    obsAndModel.DRAFT[idx,0]=draft
    return obsAndModel

def toThick(obsAndModel):
    # compute thickness, draft and THCKE_ICE_SNOW for sources that don't have them
    month=obsAndModel.obs_time.dt.month.values
    snow=obsAndModel.sisnthick.data
    
    thick_idx=np.where(~np.isnan(obsAndModel.THICK[:,0]))[0]# get indices for sources with thickness
    draft_idx=np.where(~np.isnan(obsAndModel.DRAFT[:,0]))[0] # get indices for sources with draft
    iAs_idx=np.where(~np.isnan(obsAndModel.THICK_ICE_SNOW[:,0]))[0] # get indices for sources with draft
    
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
def scatter_plot(ds,model_name,title=None):
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

    obs_thick=ds.THICK[:,0].data
#   obs_thick=ds.DRAFT[:,0].data
    mod_thick=ds.sit.data
    good_idx=np.where(np.isfinite(obs_thick))[0]
    mod_thick=mod_thick[good_idx]
    obs_thick=obs_thick[good_idx]
    mhas_nan = np.any(np.isnan(mod_thick))
    ohas_nan = np.any(np.isnan(obs_thick))
    print(mhas_nan)
    print(ohas_nan)
    
    model = sm.OLS(mod_thick, sm.add_constant(obs_thick),missing='drop')
    results = model.fit()
    print(results.params)
    print(results.summary())
    fig,ax=plt.subplots()
    ax.scatter(obs_thick,mod_thick)
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
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
    annotation_text = (f'Errors (mean/rms): {meane:.2f}/{rmse:.2f}\n'
                       f'R² = {results.rsquared:.2f}\n'
                       f'y = {m:.2f}x + {b:.2f}\n')
    
    ax.annotate(annotation_text, xy=(0.05, 0.80), xycoords='axes fraction',
             fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="None", lw=1))
    return fig,ax,results
def plot_histograms(ds,model_name, title=None): 
    import numpy as np
    import matplotlib.pyplot as plt
    if not isinstance(ds,xr.Dataset): # if ds is not a dataset open it as a file
        ds=xr.open_dataset(ds)
    ds.load()
    obs_thick=ds.THICK_ICE_SNOW[:,0] 
    obs_thick=ds.THICK[:,0]
    mod_thick=ds.sit.data
    fig,(ax0,ax1)=plt.subplots(nrows=2,ncols=1)
    ax0.hist([obs_thick,mod_thick], bins= 50, range=[0,8], alpha=0.5, label=['Obs', 'Mod'])
    ax0.legend(loc='upper right')
    ax0.set_title('Thickness Distributions\n'+title)
    diff=obs_thick-mod_thick
    ax1.hist(diff, bins= 50, range=[-3,3], alpha=0.5, label=['Obs-Mod'])
    ax1.legend(loc='upper right')
    ax1.set_tile=('Difference (Obs-Mod) ' + title)
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
    
    input_files=make_input_file_list(model_path,file_varnames)
    xchunk=nx
    ychunk=ny
    tchunk=12*nt 
    if not isinstance(cdr_data,dict):  
        logging.info("reading CDR data")
        cdr_data,_=read_thickCDR(source) # may want to do some filtering here. 
    if not isinstance(ds,xr.Dataset):   # load the model_data set if this isn't done 
        logging.info("Model Data set will be loaded  %s", source )
        with ProgressBar(): 
            ds=xr.open_mfdataset(input_files,chunks=({"x":xchunk,"y":ychunk,"time":tchunk}))
            ds=ds.chunk({'time':tchunk}) # no idea if that helps, would seem like we are doign this already above
            if "Latitude" in list(ds.coords.keys()): # some of the heff files have Latitude/Longitude as coordinates                                                 # rename
                ds=ds.rename({"Longitude":"lon","Latitude":"lat"})   
    #       match the data in points with the data from the closest grid point         
    with ProgressBar(): 
        logging.info("Starting Matching for :  %s",source)
        pmds = match_points(cdr_data,ds,ncdf_varnames,sc,dist_thresh=dist_thresh)
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
            cdr_ds=cdr_cds.where(cdr_ds.SOURCE==source,drop=True)
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

    good_idx=np.where(np.abs(diff_time) < 15)[0] # tuple with one array per dimension hence select [0]
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

def driver(): 
    # main driver program. Edit as needed. 
    logging.basicConfig(stream=sys.stdout,level=logging.INFO)
    cdr_file='/home/axel/projects/sea_ice_cdr/idl/axel_cdr/ncdf_dir/uThickCDR_summaries.20251010_V1.0.nc'
    """
    gridName='912X672'
    model_id='1c'
    model_group='capmas'
    dist_thresh=10.
    nt=44
    file_varnames=['heff','snow']
    model_path="/data/axel0/model_results/capmas/1c/nc/"
    """

    """
    gridName='912X672'
    model_id='2c'
    model_group='capmas'
    dist_thresh=10.
    nt=21
    file_varnames=['heff','snow']
    model_path="/data/axel0/model_results/capmas/2c/nc/"
    """
    """
    #   model_path="/data/axel0/model_results/2e/nc/"
    #   gridName='752X240
    """
    """
    gridName='360X120'   
    model_group=''
    model_id='2cfs1'
    dist_thresh=25. # maximum between grid cell center and observation
    file_varnames=['hiday','snowday'] 
    nt=46
    """
    """
    gridName='360X120'   
    model_group=''
    model_id='2sstC'
    dist_thresh=25. # maximum between grid cell center and observation
    nt=46
    file_varnames=['hiday','snowday']    
    model_path="/data/axel0/model_results/2sst/nc/"
    """
    gridName='360X120'   
    model_group=''
    model_id='3sst'
    dist_thresh=25. # maximum between grid cell center and observation
    nt=46
    file_varnames=['hiday','snowday']    
    model_path="/data/axel0/model_results/"+model_id+"/nc/"
    
    g=im.gridInfo(gridName) 
    nx=g.nx; ny=g.ny
    model_path="/data/axel0/model_results/"+model_group+"/"+model_id+"/nc/"
    output_root="/home/axel/ax_python_lib/thickCDR/extractDir/"
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
#    print(sources)
    match=True
    merge=True
#   sources=sources[0:2]
    for source in sources: 
        print(source)
        id= model_id +"_"  + source # name used for identifcation inside of file 
        cdr_data,cdr_ds=read_thickCDR(source=source) 
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
    
if __name__ == "__main__":
    #driver()
    subset=readMerged(source='BGEP')
    test=toThick(subset)
    
    
    