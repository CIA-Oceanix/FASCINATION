import os
import json
import hashlib
import pickle

class Datamodule_3D_pickle():
    def __init__(self,
                 dm_config,
                 dm):
        
        
        self.dm_config = dm_config
        self.depth_pre_treatment = dm.depth_pre_treatment
        self.norm_stats=dm.norm_stats
        self.depth_array = dm.depth_array
        self.coords = dm.coords
        self.train_ds = dm.train_ds
        self.test_ds = dm.test_ds

        json_string = json.dumps(dm_config)

        hash_object = hashlib.md5(json_string.encode()) 
        self.file_name = hash_object.hexdigest()

    
    def pickle_save(self,
                    save_dir):
        
        file_path = os.path.normpath(f"{save_dir}/{self.file_name}.pkl")
        
        with open(file_path, "wb") as f:
            pickle.dump(self, f)



    def pickle_load(self,
                    save_dir,
                    dm_config):
        
        json_string = json.dumps(dm_config)

        hash_object = hashlib.md5(json_string.encode()) 
        file_name = hash_object.hexdigest()

        file_path = os.path.normpath(f"{save_dir}/{file_name}.pkl")
        
        with open(file_path, "rb") as f:
            self = pickle.load(self, f)

        