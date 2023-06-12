import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import regionmask
import pandas as pd
import geopandas
from scipy import stats
import pymannkendall as mk
import xesmf as xe
import statsmodels.api as sm
import pymannkendall as mk
from scipy.linalg import toeplitz

####################################################################################################################

### Adjust the lightness of the color, in order to make them less shiny ###

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])

####################################################################################################################

### Load netcdf and merge them, function that needs to be rewritten ###

# def load_netcdf(path, yr_s, yr_e,*var):
#     data = []
#     if len(var) == 0:
#         for i in range(yr_s,yr_e):
#             os.chdir(path)
#             data.append(xr.open_dataset(path+str(glob('*{}080106.nc'.format(str(i)))[0])))

#     else:
#         ls_var = var[0]
#         for i in range(yr_s,yr_e):
#             os.chdir(path)
#             data.append(xr.open_dataset(path+str(glob('*{}080106.nc'.format(str(i)))[0]))[ls_var])
#     return data

####################################################################################################################

### Add a new dimension to a dataset/datarray - middle_slices_ZS, creating a mask to slice dataset by elevation band ###
# middle_slices_ZS specify the center of the elevation band
# Needs an orography as variable/coordinates (u can specify name with arg elevation_band)
# ls_alt is the list of the boundary of each elevation slice

def per_alt(data,ls_alt, elevation_label = 'ZS' ): # ls_alt = np.arange(0,4200,300) (for example)
    
    data_per_alt = xr.concat([data.where((data[elevation_label] >= ls_alt[i])*(data[elevation_label] < ls_alt[i+1])) for i in range(0,len(ls_alt)-1)], dim = 'middle_slices_ZS')
    data_per_alt['middle_slices_ZS'] = ls_alt[1:] - (ls_alt[1] - ls_alt[0])/2
    
    return data_per_alt

####################################################################################################################

### Add a new dimension to a dataset/dataarray - region, creating a mask based on a shapefile to slice dataset by spatially defined region ###
# region names can be defined by passing the arg name
# Works for dataset that have lat/lon as coordinates

def applied_mask_from_shp(data, shp_path, data_lat_label = "lat", data_lon_label = "lon", name = [], wrap_lon = True):
    shp = geopandas.read_file(shp_path)
    if len(name) == 0:
        mask_shp = regionmask.from_geopandas(shp)
        specific_mask = mask_shp.mask_3D(data[data_lon_label],data[data_lat_label], drop= False, wrap_lon = wrap_lon)
    else:
        mask_shp = regionmask.from_geopandas(shp, names = name)
        specific_mask = mask_shp.mask_3D(data[data_lon_label],data[data_lat_label], drop= False, wrap_lon = wrap_lon)
        specific_mask['region'] = specific_mask['names']
    data_mask = data.where(specific_mask)
    return data_mask

####################################################################################################################

### Core function to calculate trends over large datasets to wrap with apply_u_func ###

def core_theilslopes(y, time,method, seasonal, period):
    
    dt = 365.25*10 # Factor of the trends to transform it in /dec [conversion from day to decade]
    
    if (np.count_nonzero(~np.isnan(y))/len(y))*100 < 80: # Skip if there are more than 20% of nan in the array
        
        return np.array([np.nan,np.nan,np.nan,np.nan])
    
    else:
        
        if method == 'OLS':
            
            ols_model = sm.OLS(y,sm.add_constant(pd.to_numeric(time.astype('datetime64[D]')))).fit()
            slope, lowslope, upslope = ols_model.params[1], ols_model.conf_int(0.05)[1][0],ols_model.conf_int(0.05)[1][1]
        
        elif method == 'GLS':
            
            if np.count_nonzero(~np.isnan(y)) >= len(y): # Skip if there are any nan, because GLS will fail
                ols_resid = sm.OLS(y,sm.add_constant(pd.to_numeric(time.astype('datetime64[D]')))).fit().resid # Calculate residual of the OLS
                resid_fit = sm.OLS(np.asarray(ols_resid)[1:], sm.add_constant(np.asarray(ols_resid)[:-1])).fit() # Fit the residual with an OLS
                rho = resid_fit.params[1] # Use it to determine the order of AR(1) = coefficient
                sigma = rho ** toeplitz(range(len(ols_resid)))
                gls_model = sm.GLS(y,sm.add_constant(pd.to_numeric(time.astype('datetime64[D]'))),sigma=sigma).fit() # Fit the GLS
                slope, lowslope, upslope = gls_model.params[1], gls_model.conf_int(0.05)[1][0],gls_model.conf_int(0.05)[1][1]
 
            else:
                slope, lowslope, upslope = np.nan,np.nan,np.nan
            
        elif method == 'TS':
            
            my = np.ma.masked_array(y, mask=np.isnan(y))
            slope, medintercept, lowslope, upslope = stats.mstats.theilslopes(my,time.astype('datetime64[D]'))
            slope, medintercept, lowslope, upslope = slope, medintercept, lowslope, upslope
            
        elif method == 'GLSAR':
            
            if np.count_nonzero(~np.isnan(y)) < len(y): # Skip if there are any nan
                slope, lowslope, upslope = np.nan,np.nan,np.nan
                
            else:
                glsar_model = sm.GLSAR(y,sm.add_constant(pd.to_numeric(time.astype('datetime64[D]'))),rho = 1).fit()
                slope, lowslope, upslope = glsar_model.params[1], glsar_model.conf_int(0.05)[1][0],glsar_model.conf_int(0.05)[1][1]

        if seasonal == True:
            mktest = mk.seasonal_test(y,period = period, alpha =0.05).p
        
        else:
            mktest = mk.original_test(y, alpha =0.05).p
        
        return np.array([slope*dt, lowslope*dt, upslope*dt, mktest])
    
####################################################################################################################

### Wrapper of the trends calculation with apply_u_func ###

def theilslopes(data, method = 'TS',seasonal = False, period = None):
    
    # Applied the apply_ufunc along all the axis of the dataarray
    ds_param = xr.apply_ufunc(core_theilslopes,data,data.time.values,method,seasonal,period, input_core_dims=[['time'],['time'],[],[],[]],output_core_dims=[["parameter"]],
                           vectorize=True,
                           dask="parallelized",
                           output_sizes={"parameter": 4})
    ls_parameter = ['slope','loslope','hislope','mktest']
    ds_param['parameter'] = ls_parameter
    ds_slope = [ds_param.sel(parameter=ls_parameter[i]) for i in range(0,len(ls_parameter))]
    del ds_param
    
    ds_slope = {'slope':ds_slope[0],'loslope':ds_slope[1],'hislope':ds_slope[2],"mktest":ds_slope[3]}
    data = xr.merge([data,ds_slope],compat='override')

    return data

import pymannkendall as mk


####################################################################################################################

### Select point that are inside a box that u need to specify boundaries in lat lon ###

def sellonlatbox(ds,spatial_dim,lat_inf=43.5,lat_sup=49.5,lon_inf=4,lon_sup=18, avoid_reindex = True):
    
    lonlatbox = ds[['lat','lon']]
    
    if avoid_reindex == True: # Specific case where where reindex the lat/lon
        for i in range(0,len(spatial_dim)):
            lonlatbox[spatial_dim[i]] = lonlatbox[spatial_dim[i]]+1
        lonlatbox = lonlatbox.where((lonlatbox['lat'] > lat_inf)*(lonlatbox['lat'] < lat_sup)*(lonlatbox['lon'] > lon_inf)*(lonlatbox['lon'] < lon_sup), drop = True)
        
        for i in range(0,len(spatial_dim)):
            lonlatbox[spatial_dim[i]] = lonlatbox[spatial_dim[i]]-1
    
    else:
        lonlatbox = lonlatbox.where((lonlatbox['lat'] > lat_inf)*(lonlatbox['lat'] < lat_sup)*(lonlatbox['lon'] > lon_inf)*(lonlatbox['lon'] < lon_sup), drop = True)
        
    return ds.sel({spatial_dim[i] : lonlatbox[spatial_dim[i]] for i in range(len(spatial_dim))})

####################################################################################################################

### Compute the consecutive snow cover duration (CSCD), SOD (snow onset date) and SMOD (snow melt out date) snow duration (SD) based on a given threshold 
# Outputs are given as integer for SCD, and datetime for the SOD and SMOD

def lcscd(data, threshold = 15):
    data = xr.where(data > threshold, True, False)
    cumulative = data.cumsum(dim='time')-data.cumsum(dim='time').where(data.values == 0).ffill(dim='time').fillna(0)
    scd = (cumulative.max(dim = 'time')).rename('scd')
    mod = (cumulative.idxmax(dim = 'time') + np.timedelta64(1,'D')).rename('mod')
    sod = (mod - scd.astype('timedelta64[D]')).rename('sod')
    sd = data.where(data == True, np.nan).count(dim = 'time').rename('sd')
    return xr.merge([scd,mod,sod,sd])

####################################################################################################################

### Wrapper for resampling 2D coordinates with specific weight function, using pyresample libraries ### Prefer xesmf

def init_resampling(data_src,data_trgt, wf = lambda r: 1/r**2, neighbours = 25):
    
    if len(np.shape(data_src['lon'])) == 1:
        lon2d_s, lat2d_s = np.meshgrid(data_src['lon'], data_src['lat'])
    else:
        lon2d_s, lat2d_s = data_src['lon'], data_src['lat']
    if len(np.shape(data_trgt['lon'])) == 1:
        lon2d_t, lat2d_t = np.meshgrid(data_trgt['lon'], data_trgt['lat'])
    else:
        lon2d_t, lat2d_t = data_trgt['lon'], data_trgt['lat']
        
    source_def = pyresample.geometry.SwathDefinition(lons=lon2d_s, lats=lat2d_s)
    target_def = pyresample.geometry.SwathDefinition(lons=lon2d_t, lats=lat2d_t)
    
    ds = xr.apply_ufunc(resampling_core,data_src,source_def,target_def,wf,neighbours,
                          input_core_dims=[['lat','lon'],[],[],[],[]],output_core_dims=[['lat','lon']],
                          exclude_dims=set(('lat','lon',)),
                          vectorize = True,dask="parallelized")
    ds = ds.rename({'lon':'rlon','lat':'rlat'})
    ds['lon'], ds['lat'] = data_trgt['lon'], data_trgt['lat']
    return ds

####################################################################################################################

### Wrapper for resampling 1D coordinates with specific weight function, using pyresample libraries ### Prefer xesmf

def init_resampling_1Dcoords(data_src,data_trgt, wf = lambda r: 1/r**2, neighbours = 25):
    
    if len(np.shape(data_src['lon'])) == 1:
        lon2d_s, lat2d_s = np.meshgrid(data_src['lon'], data_src['lat'])
    else:
        lon2d_s, lat2d_s = data_src['lon'], data_src['lat']
    if len(np.shape(data_trgt['rlon'])) == 1:
        lon2d_t, lat2d_t = np.meshgrid(data_trgt['rlon'], data_trgt['rlat'])
    else:
        lon2d_t, lat2d_t = data_trgt['rlon'], data_trgt['rlat']
        
    source_def = pyresample.geometry.SwathDefinition(lons=lon2d_s, lats=lat2d_s)
    target_def = pyresample.geometry.SwathDefinition(lons=lon2d_t, lats=lat2d_t)
    
    ds = xr.apply_ufunc(resampling_core,data_src,source_def,target_def,wf,neighbours,
                          input_core_dims=[['lat','lon'],[],[],[],[]],output_core_dims=[['lat','lon']],
                          exclude_dims=set(('lat','lon',)),
                          vectorize = True,dask="parallelized")
    ds = ds.rename({'lon':'rlon','lat':'rlat'})
    ds['lon'], ds['lat'] = data_trgt['rlon'], data_trgt['rlat']
    return ds

####################################################################################################################

### Core function for the resample ###

def resampling_core(data_src,source_def,target_def, wf, neighbours):
    ds = pyresample.kd_tree.resample_custom(source_def,np.array(data_src), \
        target_def,radius_of_influence=8000, neighbours=int(neighbours),\
                        fill_value=None, weight_funcs=wf)
    return ds

####################################################################################################################

### Function to resampled data monthly and seasonnaly ### Prefer not to use it know

def calc_mean(data,mean_type):

    ds_mean = []
    
    if mean_type == 'month':
        
        num_index = [9,10,11,12,1,2,3,4,5,6,7,8]
        ls_name = ['Sep', 'Oct','Nov', 'Dec','Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug']
        ds_res = data.drop([ var for var in data.variables if not 'time' in data[var].dims ]).resample(time='1MS').mean()
        ds_timeless = data.drop([ var for var in data.variables if     'time' in data[var].dims ])
        
        for i in range(0,len(num_index)):
            ds_month = ds_res.sel(time=ds_res['time.month'] == num_index[i])
            ds_month = ds_month.groupby('time.year').mean('time')
            ds_mean.append(ds_month)
            
        ds_mean = xr.concat(ds_mean,dim='month')
        ds_mean['month'] = ls_name
        ds_mean = xr.merge([ds_mean, ds_timeless])

    if mean_type == 'season':
        
        num_index = [11,2,5,8]
        ls_name = ['SON','DJF','MAM','JJA']
        ds_res = data.drop([ var for var in data.variables if not 'time' in data[var].dims ]).resample(time='Q-FEB').mean()
        ds_timeless = data.drop([ var for var in data.variables if     'time' in data[var].dims ])
        for i in range(0,len(num_index)):
            ds_season = ds_res.sel(time=ds_res['time.month'] == num_index[i])
            ds_season = ds_season.groupby('time.year').mean('time')
            ds_mean.append(ds_season)
       
        ds_mean = xr.concat(ds_mean,dim='season')
        ds_mean['season'] = ls_name
        ds_mean = xr.merge([ds_mean, ds_timeless])
        
    if mean_type == 'winter':
        
        num_index = [11,2,5,8]
        ls_name = ['NDJFMA']
        
        a = data.drop([ var for var in data.variables if not 'time' in data[var].dims ]).sel(time=slice(str(pd.DatetimeIndex([data.time[0].values]).year.values[0])+'-11',str(pd.DatetimeIndex([data.time[-1].values]).year.values[0])+'-04')).resample(time = '6MS').mean()
        ds_timeless = data.drop([ var for var in data.variables if     'time' in data[var].dims ])
        
        a = a.sel(time = a.time.dt.month.isin(11)).resample(time = '1Y').mean()
        a = a.groupby('time.year').mean('time')
        ds_mean = a.expand_dims('winter')
        ds_mean['winter'] = ['NDJFMA']
        ds_mean = xr.merge([ds_mean, ds_timeless])
    
#     ds_mean['year'] = pd.to_datetime(ds_mean['year'], format = '%Y')
    return ds_mean

####################################################################################################################

### Function to resample a datarray/dataset using the xesmf package ###
# For conservative regridding, xesmf needs the boundaries of each lat/lon points, so we recalculate them 

def resampled_xesmf(ds_in,ds_out,in_x = 'x',in_y = 'y',out_x = 'x', out_y = 'y',method='conservative', ignore_degenerate = True):

    if method == 'conservative':
        
        if len(np.shape(ds_in['lat'])) > 1: # Defined lat/lon bounds for 2D curvilinear grid
           
            diff_lat = ds_in.lat.differentiate(coord = in_y) # Compute the gradient along y
            lat_b = ds_in.lat - diff_lat/2 # Get the inferior boundaries
            lat_b = np.vstack([lat_b, lat_b[-1,:] + diff_lat[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
            lat_b = np.c_[lat_b, lat_b[:,-1] + np.append(diff_lat[:,-1], diff_lat[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

            lat_b = xr.DataArray(data = lat_b,dims = ('yb','xb'), coords = ( np.append(ds_in[in_y] - (ds_in[in_y][-1]-ds_in[in_y][-2])/2,ds_in[in_y][-1]+(ds_in[in_y][-1]-ds_in[in_y][-2])/2), np.append(ds_in[in_x] - (ds_in[in_x][-1]-ds_in[in_x][-2])/2 ,ds_in[in_x][-1]+(ds_in[in_x][-1]-ds_in[in_x][-2])/2) ) )

            diff_lon = ds_in.lon.differentiate(coord = in_x) # Compute the gradient along x
            lon_b = ds_in.lon - diff_lon/2 # Get the inferior boundaries
            lon_b = np.vstack([lon_b, lon_b[-1,:] + diff_lon[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
            lon_b = np.c_[lon_b, lon_b[:,-1] + np.append(diff_lon[:,-1], diff_lon[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

            lon_b = xr.DataArray(data = lon_b,dims = ('yb','xb'), coords = ( np.append(ds_in[in_y] - (ds_in[in_y][-1]-ds_in[in_y][-2])/2,ds_in[in_y][-1]+(ds_in[in_y][-1]-ds_in[in_y][-2])/2), np.append(ds_in[in_x] - (ds_in[in_x][-1]-ds_in[in_x][-2])/2 ,ds_in[in_x][-1]+(ds_in[in_x][-1]-ds_in[in_x][-2])/2) ) )
            
        else: # Defined lat/lon bounds for 1D rectilinear grid
            lat_b = np.linspace(ds_in.lat[0]-(ds_in.lat[1]-ds_in.lat[0])/2,ds_in.lat[-1]+(ds_in.lat[1]-ds_in.lat[0])/2,len(ds_in.lat)+1)
            lon_b = np.linspace(ds_in.lon[0]-(ds_in.lon[1]-ds_in.lon[0])/2,ds_in.lon[-1]+(ds_in.lon[1]-ds_in.lon[0])/2,len(ds_in.lon)+1)

            lon_b = xr.DataArray(data = lon_b,dims = ('xb'))
            lat_b = xr.DataArray(data = lat_b,dims = ('yb'))

        ds_in['lon_b'] = lon_b
        ds_in['lat_b'] = lat_b
        ds_in = ds_in.set_coords(['lat_b','lon_b'])
        
        if len(np.shape(ds_out['lat'])) > 1: # Defined lat/lon bounds for 2D curvilinear grid
            diff_lat = ds_out.lat.differentiate(coord = out_y) # Compute the gradient along y
            lat_b = ds_out.lat - diff_lat/2 # Get the inferior boundaries
            lat_b = np.vstack([lat_b, lat_b[-1,:] + diff_lat[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
            lat_b = np.c_[lat_b, lat_b[:,-1] + np.append(diff_lat[:,-1], diff_lat[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

            lat_b = xr.DataArray(data = lat_b,dims = ('yb','xb'), coords = ( np.append(ds_out[out_y] - (ds_out[out_y][-1]-ds_out[out_y][-2])/2,ds_out[out_y][-1]+(ds_out[out_y][-1]-ds_out[out_y][-2])/2), np.append(ds_out[out_x] - (ds_out[out_x][-1]-ds_out[out_x][-2])/2 ,ds_out[out_x][-1]+(ds_out[out_x][-1]-ds_out[out_x][-2])/2) ) )

            diff_lon = ds_out.lon.differentiate(coord = out_x) # Compute the gradient along x
            lon_b = ds_out.lon - diff_lon/2 # Get the inferior boundaries
            lon_b = np.vstack([lon_b, lon_b[-1,:] + diff_lon[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
            lon_b = np.c_[lon_b, lon_b[:,-1] + np.append(diff_lon[:,-1], diff_lon[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

            lon_b = xr.DataArray(data = lon_b,dims = ('yb','xb'), coords = ( np.append(ds_out[out_y] - (ds_out[out_y][-1]-ds_out[out_y][-2])/2,ds_out[out_y][-1]+(ds_out[out_y][-1]-ds_out[out_y][-2])/2), np.append(ds_out[out_x] - (ds_out[out_x][-1]-ds_out[out_x][-2])/2 ,ds_out[out_x][-1]+(ds_out[out_x][-1]-ds_out[out_x][-2])/2) ) )
            
        else: # Defined lat/lon bounds for 1D rectilinear grid
            lat_b = np.linspace(ds_out.lat[0]-(ds_out.lat[1]-ds_out.lat[0])/2,ds_out.lat[-1]+(ds_out.lat[1]-ds_out.lat[0])/2,len(ds_out.lat)+1)
            lon_b = np.linspace(ds_out.lon[0]-(ds_out.lon[1]-ds_out.lon[0])/2,ds_out.lon[-1]+(ds_out.lon[1]-ds_out.lon[0])/2,len(ds_out.lon)+1)

            lon_b = xr.DataArray(data = lon_b,dims = ('xb'))
            lat_b = xr.DataArray(data = lat_b,dims = ('yb'))

        ds_out['lon_b'] = lon_b
        ds_out['lat_b'] = lat_b
        ds_out = ds_out.set_coords(['lat_b','lon_b'])

    regrid = xe.Regridder(ds_in,ds_out, method = method, ignore_degenerate=ignore_degenerate)
    return regrid(ds_in,na_thres=0.5)



# def resampled_xesmf(ds_in,ds_out,in_x = 'x',in_y = 'y',out_x = 'x', out_y = 'y',method='conservative', ignore_degenerate = True):

#     if method == 'conservative':
        
#         if len(np.shape(ds_in['lat'])) > 1: # Defined lat/lon bounds for 2D curvilinear grid
           
#             diff_lat = ds_in.lat.differentiate(coord = in_y) # Compute the gradient along y
#             lat_b = ds_in.lat - diff_lat/2 # Get the inferior boundaries
#             lat_b = np.vstack([lat_b, lat_b[-1,:] + diff_lat[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
#             lat_b = np.c_[lat_b, lat_b[:,-1] + np.append(diff_lat[:,-1], diff_lat[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

#             lat_b = xr.DataArray(data = lat_b,dims = ('yb','xb'), coords = ( np.append(ds_in[in_y] - (ds_in[in_y][-1]-ds_in[in_y][-2])/2,ds_in[in_y][-1]+(ds_in[in_y][-1]-ds_in[in_y][-2])/2), np.append(ds_in[in_x] - (ds_in[in_x][-1]-ds_in[in_x][-2])/2 ,ds_in[in_x][-1]+(ds_in[in_x][-1]-ds_in[in_x][-2])/2) ) )

#             diff_lon = ds_in.lon.differentiate(coord = in_x) # Compute the gradient along x
#             lon_b = ds_in.lon - diff_lon/2 # Get the inferior boundaries
#             lon_b = np.vstack([lon_b, lon_b[-1,:] + diff_lon[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
#             lon_b = np.c_[lon_b, lon_b[:,-1] + np.append(diff_lon[:,-1], diff_lon[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

#             lon_b = xr.DataArray(data = lon_b,dims = ('yb','xb'), coords = ( np.append(ds_in[in_y] - (ds_in[in_y][-1]-ds_in[in_y][-2])/2,ds_in[in_y][-1]+(ds_in[in_y][-1]-ds_in[in_y][-2])/2), np.append(ds_in[in_x] - (ds_in[in_x][-1]-ds_in[in_x][-2])/2 ,ds_in[in_x][-1]+(ds_in[in_x][-1]-ds_in[in_x][-2])/2) ) )
            
#         else: # Defined lat/lon bounds for 1D rectilinear grid
#             lat_b = np.linspace(ds_in.lat[0]-(ds_in.lat[1]-ds_in.lat[0])/2,ds_in.lat[-1]+(ds_in.lat[1]-ds_in.lat[0])/2,len(ds_in.lat)+1)
#             lon_b = np.linspace(ds_in.lon[0]-(ds_in.lon[1]-ds_in.lon[0])/2,ds_in.lon[-1]+(ds_in.lon[1]-ds_in.lon[0])/2,len(ds_in.lon)+1)

#             lon_b = xr.DataArray(data = lon_b,dims = ('xb'))
#             lat_b = xr.DataArray(data = lat_b,dims = ('yb'))

#         ds_in['lon_b'] = lon_b
#         ds_in['lat_b'] = lat_b
#         ds_in = ds_in.set_coords(['lat_b','lon_b'])
        
#         if len(np.shape(ds_out['lat'])) > 1: # Defined lat/lon bounds for 2D curvilinear grid
#             diff_lat = ds_out.lat.differentiate(coord = in_y) # Compute the gradient along y
#             lat_b = ds_out.lat - diff_lat/2 # Get the inferior boundaries
#             lat_b = np.vstack([lat_b, lat_b[-1,:] + diff_lat[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
#             lat_b = np.c_[lat_b, lat_b[:,-1] + np.append(diff_lat[:,-1], diff_lat[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

#             lat_b = xr.DataArray(data = lat_b,dims = ('yb','xb'), coords = ( np.append(ds_out[in_y] - (ds_out[in_y][-1]-ds_out[in_y][-2])/2,ds_out[in_y][-1]+(ds_out[in_y][-1]-ds_out[in_y][-2])/2), np.append(ds_out[in_x] - (ds_out[in_x][-1]-ds_out[in_x][-2])/2 ,ds_out[in_x][-1]+(ds_out[in_x][-1]-ds_out[in_x][-2])/2) ) )

#             diff_lon = ds_out.lon.differentiate(coord = in_x) # Compute the gradient along x
#             lon_b = ds_out.lon - diff_lon/2 # Get the inferior boundaries
#             lon_b = np.vstack([lon_b, lon_b[-1,:] + diff_lon[-1,:]]) # Append the upper boundary (append a row at the top) taking the row-1 gradient values to append
#             lon_b = np.c_[lon_b, lon_b[:,-1] + np.append(diff_lon[:,-1], diff_lon[-1,-1])] # Same for the column, but you also have to append the gradient of 1 row at the top, just as you did the last line

#             lon_b = xr.DataArray(data = lon_b,dims = ('yb','xb'), coords = ( np.append(ds_out[in_y] - (ds_out[in_y][-1]-ds_out[in_y][-2])/2,ds_out[in_y][-1]+(ds_out[in_y][-1]-ds_out[in_y][-2])/2), np.append(ds_out[in_x] - (ds_out[in_x][-1]-ds_out[in_x][-2])/2 ,ds_out[in_x][-1]+(ds_out[in_x][-1]-ds_out[in_x][-2])/2) ) )
            
#         else: # Defined lat/lon bounds for 1D rectilinear grid
#             lat_b = np.linspace(ds_out.lat[0]-(ds_out.lat[1]-ds_out.lat[0])/2,ds_out.lat[-1]+(ds_out.lat[1]-ds_out.lat[0])/2,len(ds_out.lat)+1)
#             lon_b = np.linspace(ds_out.lon[0]-(ds_out.lon[1]-ds_out.lon[0])/2,ds_out.lon[-1]+(ds_out.lon[1]-ds_out.lon[0])/2,len(ds_out.lon)+1)

#             lon_b = xr.DataArray(data = lon_b,dims = ('xb'))
#             lat_b = xr.DataArray(data = lat_b,dims = ('yb'))

#         ds_out['lon_b'] = lon_b
#         ds_out['lat_b'] = lat_b
#         ds_out = ds_out.set_coords(['lat_b','lon_b'])

#     regrid = xe.Regridder(ds_in,ds_out, method = method, ignore_degenerate=ignore_degenerate)
#     return regrid(ds_in,na_thres=0.5)
####################################################################################################################

### TaylorDiagram, function belonging to Y.Copin ###
# Copin, Yannick. (2012). Taylor diagram for python/matplotlib (2018-12-06). Zenodo. https://doi.org/10.5281/zenodo.5548061

class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = np.pi
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = np.pi/2
        tlocs = np.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = plt.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=10, label=label)
        t = np.linspace(0, self.tmax)
        r = np.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(np.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax),
                             np.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


## For S2M

def per_massif(data):
    
    ds_massif = []
    massif = geopandas.read_file('/cnrm/cen/users/NO_SAVE/monteirod/Shapefile/massifs_safran_shp/massifs_alpes_4326.shp')
    massif_name=np.array(massif.nom)
    massif_nums2m = np.array(massif.massif_num) 

    for i in range(0,23):
        ds_massif.append(data.where(data.massif_number==massif_nums2m[i]))

    ds_massif = xr.concat(ds_massif, dim = 'massif')
    ds_massif['massif'] = massif_name

    return ds_massif
