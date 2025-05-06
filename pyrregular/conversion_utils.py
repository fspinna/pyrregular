from copy import deepcopy

import awkward as ak
import numba as nb
import numpy as np
import sparse


@nb.njit
def dense_rank(a):
    unique_values = np.unique(a)
    ranks = np.searchsorted(unique_values, a)
    return ranks


@nb.njit
def find_breakpoints(a):
    return np.hstack(
        (np.array([0]), np.argwhere(np.diff(a)).ravel() + 1, np.array([len(a)]))
    )


@nb.njit
def find_2d_breakpoints(a, b):
    return np.hstack(
        (
            np.array([0]),
            np.argwhere(np.diff(a) | np.diff(b)).ravel() + 1,
            np.array([len(a)]),
        )
    )


@nb.njit(parallel=True)
def remove_fill_values_from_time_idx(
    coords, ts_level=True, ts_idx=0, signal_idx=1, time_idx=-1
):
    if ts_level:
        breakpoints = find_breakpoints(coords[ts_idx, :])
    else:
        breakpoints = find_2d_breakpoints(coords[ts_idx, :], coords[signal_idx, :])
    out = coords.copy()
    for i in nb.prange(len(breakpoints) - 1):
        start = breakpoints[i]
        end = breakpoints[i + 1]
        dense_time = dense_rank(coords[time_idx, start:end])
        out[time_idx, start:end] = dense_time
    return out


def reset_time_index(
    arr: sparse.COO,
    time_id: np.ndarray,
    ts_level=True,
    ts_idx=0,
    signal_idx=1,
    time_idx=-1,
    index_scale=1e-9,
    absolute_time=True,
    concatenate_time=False,
    normalize_time=False,
):
    new_coords = remove_fill_values_from_time_idx(
        arr.coords,
        ts_level=ts_level,
        ts_idx=ts_idx,
        signal_idx=signal_idx,
        time_idx=time_idx,
    )
    new_time_idx = sparse.COO(
        coords=new_coords,
        data=time_id.astype(np.float64)[arr.coords[time_idx]] * index_scale,
        fill_value=arr.fill_value,
    )
    if ts_level:
        new_time_idx = sparse.nanmax(new_time_idx, axis=signal_idx, keepdims=True)
    if not absolute_time:
        new_time_idx = new_time_idx - new_time_idx[:, :, 0:1]
    if normalize_time:
        abs_mean = sparse.nanmean(new_time_idx)
        new_time_idx = new_time_idx - sparse.nanmin(new_time_idx, axis=2, keepdims=True)
        new_time_idx = new_time_idx / (
            sparse.nanmax(new_time_idx, axis=2, keepdims=True) + (abs_mean * 1e-8)
        )  # avoids divisions by 0 when there is only 1 timestamp
    if concatenate_time:
        return (
            sparse.concatenate(
                [
                    sparse.COO(
                        coords=new_coords, data=arr.data, fill_value=arr.fill_value
                    ),
                    new_time_idx,
                ],
                axis=1,
            ),
            new_time_idx,
        )
    else:
        return (
            sparse.COO(coords=new_coords, data=arr.data, fill_value=arr.fill_value),
            new_time_idx,
        )


def ak_dropnan(arr, axis=None):
    return ak.drop_none(ak.nan_to_none(arr), axis=axis)


def to_pypots(X, y=None):
    if y is None:
        return dict(
            X=np.swapaxes(X, 1, 2),
        )
    else:
        return dict(
            X=np.swapaxes(X, 1, 2),
            y=y,
        )


def to_tslearn(X):
    return X.swapaxes(1, 2)


def fill_time_index(arr):
    T = deepcopy(arr)

    # time delta
    a_diff = T[:, :, 1:] - T[:, :, :-1]

    # mean time delta
    delta_mean = np.nanmean(a_diff, axis=2, keepdims=True)

    # last timestep
    last_valid_t = np.nanmax(T, axis=2, keepdims=True)

    # find where the nans are
    nan_mask = np.isnan(T)

    # where nans are there is an increasing value from 1 to the last nan
    nan_indices = np.cumsum(nan_mask, axis=2)

    replacement = (last_valid_t + delta_mean * nan_indices)[nan_mask]
    T[nan_mask] = replacement
    return T
