import pytorch_lightning as pl
import numpy as np
import xarray as xr
import functools as ft
import itertools
from collections import namedtuple
import torch 

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

##*input : celerity data
##* target: accoustic data

# class BaseDataModule(pl.LightningDataModule):
#     def __init__(self, input_da, time_domain, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, **kwargs):
#         super().__init__()
#         self.input_da = input_da
#         self.time_domain = time_domain
#         self.xrds_kw = xrds_kw
#         self.dl_kw = dl_kw
#         self.aug_kw = aug_kw if aug_kw is not None else {}
#         self._norm_stats = norm_stats

#         self.train_ds = None
#         self.val_ds = None
#         self.test_ds = None
#         self._post_fn = None
    
#     def get_norm_stats(self):
#         if self._norm_stats is None:
#             self._norm_stats = self.train_mean_std()
#             print("Norm stats", self._norm_stats)
#         else:
#             self.norm_stats = tuple(self._norm_stats.values())
            
#         return self._norm_stats
    

#     def train_mean_std(self):
#         pass

#     def post_fn(self):
#         m, s = self.get_norm_stats()
#         normalize = lambda item: (item - m) / s
#         return ft.partial(ft.reduce,lambda i, f: f(i), [
#             TrainingItem._make,
#             lambda item: item._replace(input=normalize(item.celerity)),
#             lambda item: item._replace(tgt=normalize(item.accoustic)),
#         ])
        
#         ##*input : celerity data
#         ##* target: accoustic data
        
#     def setup(self):
#         pass


#     def train_dataloader(self):
#         return  torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)

#     def val_dataloader(self):
#         return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

#     def test_dataloader(self):
#         return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)


class Enatl_Natl_DataModule(pl.LightningDataModule):
    def __init__(self, loaded_da, time_domain, xrds_kw, dl_kw, aug_kw=None, norm_stats=None, **kwargs):
        super().__init__()
        self.input_da = loaded_da.input
        self.tgt_da = loaded_da.tgt
        self.time_domain = time_domain
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        #self.aug_kw = aug_kw if aug_kw is not None else {}
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        #self._post_fn = None
        
        self.mean_std_domain = kwargs.get('mean_std_domain', 'train')
        ##* Sets attribute mean_std_domain to the value passed in the keyword arguments (kwargs). 
        ##* If the 'mean_std_domain' keyword argument is not provided, it defaults to 'train'.

    def get_norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        else:
            self._norm_stats = tuple(self._norm_stats.values())
            
        return self._norm_stats
    
        
    def post_fn(self):
        ##* item is of type dataarray.data.astype(np.float32)
        m, s = self.get_norm_stats()
        print('m',m,'s',s)
        normalize = lambda item: (item - m) / s
        # return ft.partial(ft.reduce,lambda i, f: f(i), [
        #     TrainingItem._make,
        #     lambda item: item._replace(tgt=normalize(item.tgt)),
        #     lambda item: item._replace(input=normalize(item.input)),
        # ])
        
        return ft.partial(ft.reduce, lambda i, f: f(i), [normalize,])
        
        ##*input : celerity data
        ##* target: accoustic data

    def train_mean_std(self, variable='celerity'):
        train_data = self.input_da.sel(self.time_domain[self.mean_std_domain])
        ##* the selection over the spatial domain is already done on load_input()
        
        
        return (
            train_data
            [variable]
            .pipe(lambda da: (da.mean().item(), da.std().item()))
        )
            

    def setup(self, stage='test'):
        post_fn = self.post_fn()
        if stage == 'fit':
            train_data = self.input_da.sel(self.time_domain['train'])
            self.train_ds = XrDataset(
                train_data, **self.xrds_kw, postpro_fn=post_fn,
            )
            # if self.aug_kw:
            #     self.train_ds = AugmentedDataset(self.train_ds, **self.aug_kw)

            self.val_ds = XrDataset(
                self.input_da.sel(self.time_domain['val']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )
        else:
            self.test_ds = XrDataset(
                self.input_da.sel(self.time_domain['test']),
                **self.xrds_kw,
                postpro_fn=post_fn,
            )


    def train_dataloader(self):
        return  torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)


def matching_coords_test(cel_da,acc_da):

    dims = np.array(['time','lat','lon'])
    unmatched_dim = np.array([not np.array_equal(cel_da[dim].values, acc_da[dim].values) for dim in dims])
    if any(unmatched_dim):
        raise ValueError(f"Celerity and accoustic dataarrays don't have matching coordinates on {*dims[unmatched_dim],}")   ##* Unpacking with trailing comma. ##*https://stackoverflow.com/questions/42756537/f-string-syntax-for-unpacking-a-list-with-brace-suppression
  
  
def load_data(celerity_path,accoustic_path,acc_var,spatial_domain):

    cel_da = xr.open_dataarray(celerity_path).sel(spatial_domain).transpose('z','lat','lon','time').fillna(0)
    ###* fillna is necessary, otherwise nan are propagated through the convolutions
    ###* 0 makes physical sense, sound does not propagated at 0m.s plus 0 are ignored in convolution
    acc_da = xr.open_dataset(accoustic_path).sel(spatial_domain)[acc_var].transpose('lat','lon','time')
    ###! for now we restrict acc_var to the ecs only
    if "cutoff_freq" in acc_var:
        acc_da = acc_da["cutoff_freq"].where(acc_da["cutoff_freq"] > 10000, 10000)
    #matching_coords_test(cel_da,acc_da)
    ###TODO: see if we keep matching coords test

    # input_da = xr.Dataset({"input":cel_da, "tgt":acc_da}).to_array().transpose('variable','z','lat','lon','time')
    # ##* order in transpose should match xrds_kw.patch_dims & xrds_kw.strides  ##* to_array() est long et pourrait poser probleme car toutes les variables ne dépendent pas de smemes coordonnées 
    # ###* to_array() necessaire pour le XrDataset
    # ##*input : celerity data
    # ##* target: accoustic data

    return TrainingItem(cel_da,acc_da)


# def run(trainer, train_dm, test_dm, lit_mod, ckpt=None):
#     """
#     Fit and test on two distinct domains.
#     """
#     if trainer.logger is not None:
#         print()
#         print('Logdir:', trainer.logger.log_dir)
#         print()

#     trainer.fit(lit_mod, datamodule=train_dm, ckpt_path=ckpt)
#     trainer.test(lit_mod, datamodule=test_dm, ckpt_path='best')
    
    
    
    

class IncompleteScanConfiguration(Exception):
    pass

class DangerousDimOrdering(Exception):
    pass


class XrDataset(torch.utils.data.Dataset):
    """
    torch Dataset based on an xarray.DataArray with on the fly slicing.

    ### Usage: #### 
    If you want to be able to reconstruct the input

    the input xr.DataArray should:
        - have coordinates
        - have the last dims correspond to the patch dims in same order
        - have for each dim of patch_dim (size(dim) - patch_dim(dim)) divisible by stride(dim)

    the batches passed to self.reconstruct should:
        - have the last dims correspond to the patch dims in same order
    """
    def __init__(
            self, da, patch_dims, domain_limits=None, strides=None,
            check_full_scan=False, check_dim_order=False,
            postpro_fn=None
            ):
        """
        da: xarray.DataArray with patch dims at the end in the dim orders
        patch_dims: dict of da dimension to size of a patch 
        domain_limits: dict of da dimension to slices of domain to select for patch extractions
        strides: dict of dims to stride size (default to one)
        check_full_scan: Boolean: if True raise an error if the whole domain is not scanned by the patch size stride combination
        """
        super().__init__()
        self.return_coords = False
        self.postpro_fn = postpro_fn
        self.da = da.sel(**(domain_limits or {}))
        self.patch_dims = patch_dims
        self.strides = strides or {}
        da_dims = dict(zip(self.da.dims, self.da.shape))
        self.ds_size = {
            dim: max((da_dims[dim] - patch_dims[dim]) // self.strides.get(dim, 1) + 1, 0)
            for dim in patch_dims
        }


        if check_full_scan:
            for dim in patch_dims:
                if (da_dims[dim] - self.patch_dims[dim]) % self.strides.get(dim, 1) != 0:
                    raise IncompleteScanConfiguration(
                        f"""
                        Incomplete scan in dimension dim {dim}:
                        dataarray shape on this dim {da_dims[dim]}
                        patch_size along this dim {self.patch_dims[dim]}
                        stride along this dim {self.strides.get(dim, 1)}
                        [shape - patch_size] should be divisible by stride
                        """
                    )

        if check_dim_order:
            for dim in patch_dims:
                if not '#'.join(da.dims).endswith('#'.join(list(patch_dims))): 
                    raise DangerousDimOrdering(
                        f"""
                        input dataarray's dims should end with patch_dims 
                        dataarray's dim {da.dims}:
                        patch_dims {list(patch_dims)}
                        """
                )
                    
    def __len__(self):
        size = 1
        for v in self.ds_size.values():
            size *= v
        return size

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def get_coords(self):
        self.return_coords = True
        coords = []
        try:
            for i in range(len(self)):
                coords.append(self[i])
        finally:
            self.return_coords = False
            return coords

    def __getitem__(self, item):
        sl = {
                dim: slice(self.strides.get(dim, 1) * idx,
                           self.strides.get(dim, 1) * idx + self.patch_dims[dim])
                for dim, idx in zip(self.ds_size.keys(),
                                    np.unravel_index(item, tuple(self.ds_size.values())))
                }
        item =  self.da.isel(**sl)

        if self.return_coords:
            return item.coords.to_dataset()[list(self.patch_dims)]

        item = item.data.astype(np.float32)
        if self.postpro_fn is not None:
            return self.postpro_fn(item)
        return item

    def reconstruct(self, batches, weight=None):
        """
        takes as input a list of np.ndarray of dimensions (b, *, *patch_dims)
        return a stitched xarray.DataArray with the coords of patch_dims

    batches: list of torch tensor correspondin to batches without shuffle
        weight: tensor of size patch_dims corresponding to the weight of a prediction depending on the position on the patch (default to ones everywhere)
        overlapping patches will be averaged with weighting 
        """

        items = list(itertools.chain(*batches))
        return self.reconstruct_from_items(items, weight)

    def reconstruct_from_items(self, items, weight=None):
        if weight is None:
            weight = np.ones(list(self.patch_dims.values()))
        w = xr.DataArray(weight, dims=list(self.patch_dims.keys()))

        coords = self.get_coords()

        new_dims = [f'v{i}' for i in range(len(items[0].shape) - len(coords[0].dims))]
        dims = new_dims + list(coords[0].dims)

        das = [xr.DataArray(it.numpy(), dims=dims, coords=co.coords)
               for  it, co in zip(items, coords)]

        da_shape = dict(zip(coords[0].dims, self.da.shape[-len(coords[0].dims):]))
        new_shape = dict(zip(new_dims, items[0].shape[:len(new_dims)]))

        rec_da = xr.DataArray(
                np.zeros([*new_shape.values(), *da_shape.values()]),
                dims=dims,
                coords={d: self.da[d] for d in self.patch_dims} 
        )
        count_da = xr.zeros_like(rec_da)

        for da in das:
            rec_da.loc[da.coords] = rec_da.sel(da.coords) + da * w
            count_da.loc[da.coords] = count_da.sel(da.coords) + w

        return rec_da / count_da

class XrConcatDataset(torch.utils.data.ConcatDataset):
    """
    Concatenation of XrDatasets
    """
    def reconstruct(self, batches, weight=None):
        """
        Returns list of xarray object, reconstructed from batches
        """
        items_iter = itertools.chain(*batches)
        rec_das = []
        for ds in self.datasets:
            ds_items = list(itertools.islice(items_iter, len(ds)))
            rec_das.append(ds.reconstruct_from_items(ds_items, weight))
    
        return rec_das