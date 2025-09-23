import sys
import os

running_path = "/Odyssey/private/o23gauvr/code/"
sys.path.insert(0, running_path)
os.chdir(running_path)

from FASCINATION.src.autoencoder_datamodule import AutoEncoderDatamodule_3D
import xarray as xr
import pickle

dm_path = "/Odyssey/private/o23gauvr/code/FASCINATION/pickle/dm_enatl_mean_std_along_depth_4_157_240_240.pkl"


data_path ={"enatl": "/Odyssey/public/enatl60/celerity/eNATL60_BLB002_sound_speed_regrid_0_botm.nc",
            "natl": "/Odyssey/public/natl60/celerity/NATL60GULF-CJM165_sound_speed_regrid_0_botm.nc"}


datamodule = AutoEncoderDatamodule_3D(
    input_da=xr.open_dataarray(data_path["enatl"]),         # your xarray DataArray
    dl_kw={"batch_size": 4, "num_workers": 2},
    norm_stats={"method": "mean_std_along_depth","norm_location": "datamodule"},
    manage_nan="supress_with_max_depth",
    n_profiles=None,
    reshape=None,#["factor_64"], #"RGB"
    dtype_str="float32"
)


print("Setting up datamodule for 'fit' stage...")
datamodule.setup(stage="fit")
print("Creating train dataloader...")
train_dataloader = datamodule.train_dataloader()

print("Setting up datamodule for 'test' stage...")
datamodule.setup(stage="test")
print("Creating test dataloader...")
test_dataloader = datamodule.test_dataloader()

#dm = {"train": train_dataloader, "test": test_dataloader}

print(f"Saving datamodule to {dm_path} ...")
with open(dm_path, 'wb') as f:
    pickle.dump(datamodule, f)
print("Done.")
