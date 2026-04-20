# Normalization After Split Implementation

## Summary of Changes

Both datamodule files have been updated to support **normalization after split** with the new `normalize_per_split` parameter.

### Key Changes

#### 1. New Parameter: `normalize_per_split`
- **Type**: `bool`
- **Default**: `False` (maintains backward compatibility)
- **Description**: Controls whether normalization parameters are computed separately for each split

#### 2. Normalization Workflow

**When `normalize_per_split=False` (default - backward compatible):**
- Normalization stats are computed from the **train split**
- Stats are applied to all splits (train, val, test)
- Each split's `norm_stats` attribute includes: `{"method": "...", "params": {...}, "norm_from": "train"}`

**When `normalize_per_split=True` (new):**
- Normalization stats are computed **separately for each split**
- Each split uses its own normalization parameters
- Each split's `norm_stats` attribute includes: `{"method": "...", "params": {...}, "norm_from": "<split_name>"}`
  - `norm_from` can be: `"train"`, `"val"`, or `"test"`

#### 3. Execution Order
The normalization now happens **AFTER the temporal split** instead of before:
1. RGB depth layers processing
2. Temporal split (creating train, val, test)
3. **Normalization computation and application**
4. Attribute assignment

### File-by-File Changes

#### `/Odyssey/private/o23gauvr/code/FASCINATION/src/autoencoder_datamodule_good_split.py`
- Single data source (enatl)
- Train, val, test all split from same data
- Moves normalization to after split
- Supports both `normalize_per_split` modes

**Example usage:**
```python
dm = AEDatamodule(
    data_name="enatl",
    dl_kw={"batch_size": 4, "num_workers": 8},
    norm_stats={"method": "min_max"},
    normalize_per_split=True,  # Compute sep stats for each split
    ...
)
dm.setup()

# Access normalization info:
train_norm = dm.train_ds.input.attrs['norm_stats']
print(train_norm['norm_from'])  # "train"
print(val_norm['norm_from'])    # "val"
```

#### `/Odyssey/private/o23gauvr/code/FASCINATION/src/autoencoder_datamodule_natl_enatl.py`
- Dual data source (enatl for train, natl for val+test)
- Maintains backward compatibility with existing `test_norm` parameter
- When `normalize_per_split=False`, uses `test_norm` logic:
  - `"on_train"`: All splits use train normalization
  - `"on_test"`: Train uses its own, val/test use natl dataset stats
- When `normalize_per_split=True`: Ignores `test_norm` and computes separate stats for each split

**Example usage:**
```python
# Old behavior (backward compatible)
dm = AEDatamodule(
    dl_kw={"batch_size": 4, "num_workers": 8},
    norm_stats={"method": "min_max"},
    test_norm="on_test",  # Still works as before
    normalize_per_split=False,  # Default
    ...
)

# New behavior
dm = AEDatamodule(
    dl_kw={"batch_size": 4, "num_workers": 8},
    norm_stats={"method": "min_max"},
    normalize_per_split=True,  # Use per-split normalization
    ...
)
#  test_norm is ignored when normalize_per_split=True
```

### Key Features

✅ **Normalization occurs after split** - All splits are normalized with their respective statistics

✅ **`norm_from` tracking** - Each split's norm_stats has a `"norm_from"` key indicating which split was used to compute parameters

✅ **Backward compatible** - Default behavior (`normalize_per_split=False`) maintains original functionality

✅ **Dual source support** - Works with both single and dual data source datamodules

✅ **Flexible normalization methods** - Supports min_max, mean_std, and mean_std_along_depth methods on any split

### Usage Example

```python
# Track which split's statistics were used
def get_norm_info(datamodule):
    train_norm = datamodule.train_ds.input.attrs['norm_stats']
    val_norm = datamodule.val_ds.input.attrs['norm_stats']
    test_norm = datamodule.test_ds.input.attrs['norm_stats']
    
    print(f"Train normalized from: {train_norm.get('norm_from', 'N/A')}")
    print(f"Val normalized from: {val_norm.get('norm_from', 'N/A')}")
    print(f"Test normalized from: {test_norm.get('norm_from', 'N/A')}")
    
    if train_norm.get('norm_from') == 'train':
        print("Using train statistics for normalization")
    else:
        print("Using per-split normalization statistics")
```

### Testing

To verify the implementation:
```bash
cd /Odyssey/private/o23gauvr/code/FASCINATION
python test_normalization_per_split.py  # Test with datamodule_good_split.py
```
