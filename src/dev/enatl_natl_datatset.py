import xarray as xr 
import numpy as np


data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
            "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}
data = ["enatl","natl"]


max_depth = 2000
data_arrays = []

for d in data:
    da = xr.open_dataarray(data_path[d])

    # Drop all lat coordinates presenting a nan for depths (z) inferior to 2000
    # Select only data for depths < 2000
    #sub_da = da.sel(z=da.z.where(da.z < max_depth, drop=True))
    # For each lat, check if there is any nan across time, z, and lon
    #lat_nan = sub_da.isnull().any(dim=["time", "z", "lon"])
    # Get valid latitudes (i.e. where there is no nan)
    #valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
    # Select only the valid latitudes and drop all z coordinates superior to 2000.
    da = da.sel(z=da.z.where(da.z < max_depth, drop=True))

    #assert da.isnull().sum().item() == 0 


    data_arrays.append(da)

da = xr.concat(data_arrays, dim="time")
lat_nan = da.isnull().any(dim=["time", "z", "lon"])
valid_lats = lat_nan.where(lat_nan == False, drop=True).coords["lat"].values
da = da.sel(lat=valid_lats)
assert da.isnull().sum().item() == 0
del data_arrays


lat_size = len(da.lat)
lon_size = len(da.lon)
n_lat = int(np.ceil(lat_size / 64))
closest_lat_size = n_lat * 64
n_lon = int(np.ceil(lon_size / 64))
closest_lon_size = n_lon * 64
new_lat = np.linspace(da.lat.min().item(), da.lat.max().item(), closest_lat_size)
new_lon = np.linspace(da.lon.min().item(), da.lon.max().item(), closest_lon_size)

da = da.interp(lat=new_lat, lon=new_lon, method="cubic")

da.to_netcdf("/Odyssey/private/o23gauvr/input/enatl_natl_2000_192_256.nc")