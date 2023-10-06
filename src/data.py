import pytorch_lightning as pl
import numpy as np
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
import torch.utils.data
import xarray as xr
import itertools
import functools as ft
import tqdm
from collections import namedtuple

TrainingItem = namedtuple('TrainingItem', ['input', 'tgt'])

class UNetDataModule(pl.LightningDataModule):
    def __init__(self, input_da, domains, dl_kw, io_time_steps):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.dl_kw = dl_kw
        self.io_time_steps = io_time_steps

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.test_time = None
        self.test_var = None
        self.test_lat = None
        self.test_lon = None

        self.data_norm = False
    
    def setup(self, stage):
        if not self.data_norm:
            train_data = self.input_da.sel(self.domains['train'])
            for var in self.input_da.data_vars:
                mean, std = self.norm_stats(train_data[var])
                for i in range(len(self.input_da[var])):
                    self.input_da[var][i] = (self.input_da[var][i] - mean[i])/std[i]
                self.input_da[var] = self.input_da[var].transpose('time', 'var', 'lat', 'lon')
            self.data_norm = True

        if stage == 'fit':
            self.train_ds = UNetDataset(
                self.input_da.sel(self.domains['train']), self.io_time_steps
            )
            self.val_ds = UNetDataset(
                self.input_da.sel(self.domains['val']), self.io_time_steps
            )
        if stage == 'test':
            self.val_ds = UNetDataset(
                self.input_da.sel(self.domains['val']), self.io_time_steps
            )
            self.test_ds = UNetDataset(
                self.input_da.sel(self.domains['test']), self.io_time_steps
            )
            self.test_time = self.test_ds.da["time"]
            self.test_var = self.test_ds.da["var"]
            self.test_lat = self.test_ds.da["lat"]
            self.test_lon = self.test_ds.da["lon"]

    def norm_stats(self, dataset):
        mean = []
        std = []
        for i in range(len(dataset)):
            mean.append(dataset[i].mean().values.item())
            std.append(dataset[i].std().values.item())
        return mean, std
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=False, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class UNetDataset(torch.utils.data.Dataset):
    def __init__(self, da, io_time_steps=2):
        super().__init__()
        self.da = da
        self.io_time_steps = io_time_steps

    def __len__(self):
        return (len(self.da.time) - self.io_time_steps*2)//2
    
    def __getitem__(self, index):
        index *= 2
        item = (self.da.input[index:index+2].data.astype(np.float32), self.da.tgt[index+2:index+6].data.astype(np.float32))
        reshaped_item = [item[0].reshape(-1, *item[0].shape[2:]),
                     item[1]]
        return TrainingItem._make(reshaped_item)
    
class AutoEncoderDatamodule(pl.LightningDataModule):
    def __init__(self, input_da, domains, dl_kw):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.dl_kw = dl_kw

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False

    def setup(self, stage):
        if not self.is_data_normed:
            train_data = self.input_da.isel(self.domains['train'])
            for var in self.input_da.data_vars:
                mean, std = self.norm_stats(train_data[var])
                self.input_da = (self.input_da - mean)/std
            self.is_data_normed = True

        if stage == "fit":
            self.train_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['train'])
            )
            self.val_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['val'])
            )
        if stage == "test":
            self.val_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['val'])
            )
            self.test_ds = AutoEncoderDataset(
                self.input_da.isel(self.domains['test'])
            )


    def norm_stats(self, da):
        mean = da.mean()
        std = da.std()
        return mean, std
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)
    
class AutoEncoderDataset(torch.utils.data.Dataset):
    def __init__(self, da):
        super().__init__()
        self.da = da

    def __len__(self):
        return len(self.da.time)
    
    def __getitem__(self, index):
        return TrainingItem._make((np.nan_to_num(self.da.celerity[index].data.astype(np.float32)), np.nan_to_num(self.da.celerity[index].data.astype(np.float32))))
    
class AcousticPredictorDatamodule(pl.LightningDataModule):
    def __init__(self, input, target, domains, dl_kw):
        self.input = input
        self.target = target
        self.domains = domains
        self.dl_kw = dl_kw

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

        self.is_data_normed = False
    
    def setup(self, stage):
        if not self.is_data_normed:
            input_train, target_train = self.input.isel(self.domains['train']), self.target.isel(self.domains['train'])
            mean, std = self.norm_stats(input_train, target_train)
            self.input = (self.input - mean["input"])/std["input"]
            for j in self.target.data_vars:
                msk = self.target[j] != 0
                self.target[j][msk] = (self.target[j][msk] - mean[j])/std[j]
            self.is_data_normed = True
        
        if stage == 'fit':
            self.train_ds = AcousticPredictorDataset(
                self.input.isel(self.domains['train']), self.target.isel(self.domains['train'])
                )
            self.val_ds = AcousticPredictorDataset(
                self.input.isel(self.domains['val']), self.target.isel(self.domains['val'])
            )
        if stage == 'test':
            self.val_ds = AcousticPredictorDataset(
                self.input.isel(self.domains['val']), self.target.isel(self.domains['val'])
            )
            self.test_ds = AcousticPredictorDataset(
                self.input.isel(self.domains['test']), self.target.isel(self.domains['test'])
            )

    def train_dataloader(self):
        return torch.utils.data.Dataloader(self.train_ds, shuffle=True, **self.dl_kw)
    
    def val_dataloader(self):
        return torch.utils.data.Dataloader(self.val_ds, shuffle=False, **self.dl_kw)
    
    def test_dataloader(self):
        return torch.utils.data.Dataloader(self.test_ds, shuffle=False, **self.dl_kw)
    
    def norm_stats(self, input, target):
        mean, std = {}, {}
        mean["input"] = input.mean()
        std["input"] = input.std()
        for j in target.data_vars:
            msk = target[j] != 0
            mean[j] = target[j][msk].mean()
            std[j] = target[j][msk].std()

        return mean, std

class AcousticPredictorDataset(torch.utils.data.Dataset):
    def __init__(self, volume, variables):
        super().__init__()
        self.volume, self.variables = volume, variables

    def __len__(self):
        return min(len(self.volume.time), len(self.variables.time))
    
    def __getitem__(self, index):
        return TrainingItem._make((self.volume.celerity[index].data.astype(np.float32), self.variables[index].data.astype(np.float32)))
    
class AlternateDataset(torch.utils.data.IterableDataset):
    def __init__(self, da, io_time_steps=2):
        super().__init__()
        self.da = da
        self.io_time_steps = io_time_steps

    def __iter__(self):
        index = 0
        if index < len(self.da.time)-1:
            item = (self.da.input[index:index+2].data.astype(np.float32), self.da.tgt[index+2:index+6].data.astype(np.float32))
            reshaped_item = [item[0].reshape(-1, *item[0].shape[2:]),
                     item[1]]
            index += 2
            yield TrainingItem._make(reshaped_item)

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

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, inp_ds, aug_factor, aug_only=False, item_cls=TrainingItem):
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self.inp_ds = inp_ds
        self.perm = np.random.permutation(len(self.inp_ds))
        self.item_cls = item_cls

    def __len__(self):
        return len(self.inp_ds) * (1 + self.aug_factor - int(self.aug_only))

    def __getitem__(self, idx):
        if self.aug_only:
            idx = idx + len(self.inp_ds)

        if idx < len(self.inp_ds):
            return self.inp_ds[idx]

        tgt_idx = idx % len(self.inp_ds)
        perm_idx = tgt_idx
        for _ in range(idx // len(self.inp_ds)):
            perm_idx = self.perm[perm_idx]
        
        item = self.inp_ds[tgt_idx]
        perm_item = self.inp_ds[perm_idx]

        return self.item_cls(
            **{
                **item._asdict(),
                **{'input': np.where(np.isfinite(perm_item.input),
                             item.tgt, np.full_like(item.tgt,np.nan))
                 }
            }
        )


class BaseDataModule(pl.LightningDataModule):
    def __init__(self, input_da, domains, xrds_kw, dl_kw, aug_only=False, aug_factor=0, norm_stats=None):
        super().__init__()
        self.input_da = input_da
        self.domains = domains
        self.xrds_kw = xrds_kw
        self.dl_kw = dl_kw
        self.aug_factor = aug_factor
        self.aug_only = aug_only
        self._norm_stats = norm_stats

        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def norm_stats(self):
        if self._norm_stats is None:
            self._norm_stats = self.train_mean_std()
            print("Norm stats", self._norm_stats)
        return self._norm_stats

    def train_mean_std(self):
        train_data = self.input_da.sel(self.domains['train'])
        return train_data.sel(variable='tgt').pipe(lambda da: (da.mean().values.item(), da.std().values.item()))

    def setup(self, stage='test'):
        train_data = self.input_da.sel(self.domains['train'])
        post_fn = ft.partial(ft.reduce,lambda i, f: f(i), [
            lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1],
            TrainingItem._make,
        ])
        self.train_ds = XrDataset(
            train_data, **self.xrds_kw, postpro_fn=post_fn,
        )
        if self.aug_factor > 0:
            self.train_ds = AugmentedDataset(self.train_ds, aug_factor=self.aug_factor, aug_only=self.aug_only)

        self.val_ds = XrDataset(
            self.input_da.sel(self.domains['val']), **self.xrds_kw, postpro_fn=post_fn,
        )
        self.test_ds = XrDataset(
            self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,
        )


    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_ds, shuffle=True, **self.dl_kw)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_ds, shuffle=False, **self.dl_kw)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_ds, shuffle=False, **self.dl_kw)

class ConcatDataModule(BaseDataModule):
    def train_mean_std(self):
        sum, count = 0, 0
        train_data = self.input_da.sel(self.xrds_kw.get('domain_limits', {}))
        for domain in self.domains['train']:
            _sum, _count = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: (da.sum(), da.pipe(np.isfinite).sum()))
            sum += _sum
            count += _count

        mean = sum / count
        sum = 0
        for domain in self.domains['train']:
            _sum = train_data.sel(domain).sel(variable='tgt').pipe(lambda da: da - mean).pipe(np.square).sum()
            sum += _sum
        std = (sum / count)**0.5
        return mean.values.item(), std.values.item()

    def setup(self, stage='test'):
        post_fn = ft.partial(ft.reduce,lambda i, f: f(i), [
            lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1],
            TrainingItem._make,
        ])
        self.train_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['train']
        ])
        if self.aug_factor >= 1:
            self.train_ds = AugmentedDataset(self.train_ds, self.aug_factor)

        self.val_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['val']
        ])
        self.test_ds = XrConcatDataset([
            XrDataset(self.input_da.sel(domain), **self.xrds_kw, postpro_fn=post_fn,)
            for domain in self.domains['test']
        ])


class RandValDataModule(BaseDataModule):
    def __init__(self, val_prop, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.val_prop = val_prop

    def setup(self, stage='test'):
        post_fn = ft.partial(ft.reduce,lambda i, f: f(i), [
            lambda item: (item - self.norm_stats()[0]) / self.norm_stats()[1],
            TrainingItem._make,
        ])
        train_ds = XrDataset(self.input_da.sel(self.domains['train']), **self.xrds_kw, postpro_fn=post_fn,)
        n_val = int(self.val_prop * len(train_ds))
        n_train = len(train_ds) - n_val
        self.train_ds, self.val_ds = torch.utils.data.random_split(train_ds, [n_train, n_val])

        if self.aug_factor > 1:
            self.train_ds = AugmentedDataset(self.train_ds, self.aug_factor)

        self.test_ds = XrDataset(self.input_da.sel(self.domains['test']), **self.xrds_kw, postpro_fn=post_fn,)

