import numpy as np
import pandas as pd
import xarray as xr

from pyrregular.conversion_utils import _ak_dropnan, _fill_time_index, _reset_time_index
from pyrregular.io_utils import save_to_file


@xr.register_dataarray_accessor("irr")
class IrregularAccessor:
    def __init__(self, da):
        self._da = da
        self.dims = {dim: i for i, dim in enumerate(da.dims)}

    def __getitem__(self, key):
        out = self._da.__getitem__(key)
        if out["time_id"].size == 1:
            return out
        # keep only the timestamps where the selection has at least one value
        time_idx = out.dims.index("time_id")
        return out.isel(time_id=np.unique(out.data.coords[time_idx]))

    def get_task(self, task="default"):
        return self._da.attrs["configs"][task]

    def get_task_target_and_split(self, task="default"):
        task = self.get_task(task)
        return self._da[task["target"]].data, self._da[task["split"]].data

    def reset_time_index(
        self,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
        normalize_time=False,
    ):
        return _reset_time_index(
            arr=self._da.data,
            time_id=self._da["time_id"].data,
            ts_level=ts_level,
            ts_idx=self.dims["ts_id"],
            signal_idx=self.dims["signal_id"],
            time_idx=self.dims["time_id"],
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
            normalize_time=normalize_time,
        )

    def to_dense(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
        normalize_time=False,
    ):
        if reset_time_index:
            X, T = self.reset_time_index(
                ts_level=ts_level,
                index_scale=index_scale,
                absolute_time=absolute_time,
                concatenate_time=concatenate_time,
                normalize_time=normalize_time,
            )
        else:
            # the global time axis is shared by all series: T is returned as is
            if concatenate_time:
                raise ValueError("concatenate_time=True requires reset_time_index=True")
            return self._da.data.todense(), self._da["time_id"].data.reshape(1, 1, -1)
        return X.todense(), T.todense()

    def to_tslearn(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
    ):
        X, T = self.to_dense(
            reset_time_index=reset_time_index,
            ts_level=ts_level,
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
        )
        return np.swapaxes(X, 1, 2), np.swapaxes(T, 1, 2)

    def to_aeon(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
    ):
        X, T = self.to_dense(
            reset_time_index=reset_time_index,
            ts_level=ts_level,
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
        )
        return X, T

    def to_sktime(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
    ):
        X, T = self.to_dense(
            reset_time_index=reset_time_index,
            ts_level=ts_level,
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
        )
        return X, T

    def to_awkward(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
        dropna=True,
    ):
        import awkward as ak

        X, T = self.to_dense(
            reset_time_index=reset_time_index,
            ts_level=ts_level,
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
        )
        X = ak.Array(X)
        T = ak.Array(T)
        if dropna:
            X = _ak_dropnan(X)
            T = _ak_dropnan(T)
        return X, T

    def to_list(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=1e-9,
        absolute_time=True,
        concatenate_time=False,
        dropna=True,
    ):
        X, T = self.to_awkward(
            reset_time_index=reset_time_index,
            ts_level=ts_level,
            index_scale=index_scale,
            absolute_time=absolute_time,
            concatenate_time=concatenate_time,
            dropna=dropna,
        )
        return X.to_list(), T.to_list()

    def to_long(
        self,
        reset_time_index=True,
        ts_level=True,
        index_scale=None,
        absolute_time=True,
        static=False,
    ):
        """Return the data in long format: one row per observation.

        Columns are ts_id, signal_id, time_id and value_id, as read by read_csv.
        Ids are categoricals with the original labels. Times are the original
        values unless index_scale is given (then float, scaled like to_dense).
        absolute_time=False makes times relative to the first one of each series
        (or of each series/signal pair if ts_level=False). static=True adds every
        other coordinate as a column, repeated on each row.
        reset_time_index has no effect and is kept for backward compatibility.
        """
        arr = self._da.data
        idx = {dim: arr.coords[i] for dim, i in self.dims.items()}

        time = self._da["time_id"].values
        if index_scale is not None:
            time = time.astype(np.float64) * index_scale
        time = time[idx["time_id"]]
        if not absolute_time:
            group = [idx["ts_id"]] if ts_level else [idx["ts_id"], idx["signal_id"]]
            time = time - pd.Series(time).groupby(group).transform("min").to_numpy()

        columns = {
            "ts_id": _repeat(self._da["ts_id"].values, idx["ts_id"]),
            "signal_id": _repeat(self._da["signal_id"].values, idx["signal_id"]),
            "time_id": time,
            "value_id": arr.data,
        }
        if static:
            for name, coord in self._da.coords.items():
                if name in self.dims:
                    continue
                if coord.ndim == 0:
                    columns[name] = np.repeat(coord.values, arr.nnz)
                else:
                    columns[name] = _repeat(coord.values, idx[coord.dims[0]])
        return pd.DataFrame(columns)

    def to_hdf5(self, filename, compression="gzip", compression_opts=None):
        save_to_file(
            data_array=self._da,
            filename=filename,
            compression=compression,
            compression_opts=compression_opts,
        )


def _repeat(values, codes):
    """values[codes], as a categorical when values are strings or objects."""
    if values.dtype.kind in "OUS":
        cat = pd.Categorical(values)
        return pd.Categorical.from_codes(cat.codes[codes], cat.categories)
    return values[codes]
