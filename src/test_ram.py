import xarray as xr
import numpy as np

path = "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc"
print(path)
da = xr.open_dataarray(path)
print("da: ",da)
print("da shape: ",da.shape)

print("da data shape: ", da.data.shape)
print("da data head: ", da.data[0,:3,:3,:3])


print("da data mean: ",np.nanmean(da.data))