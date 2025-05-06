import awkward as ak
import numpy as np
import sparse
import xarray as xr

from pyrregular.conversion_utils import ak_dropnan, fill_time_index, reset_time_index
from pyrregular.io_utils import save_to_file


@xr.register_dataarray_accessor("irr")
class IrregularAccessor:
    def __init__(self, da):
        self._da = da
        self.dims = {dim: i for i, dim in enumerate(da.dims)}

    def __getitem__(self, key):
        import numpy as np

        out = self._da.__getitem__(key)
        if out["time_id"].size == 1:
            return out
        dims = {dim: i for i, dim in enumerate(out.dims)}
        return out[..., np.sort(out.data.coords[dims["time_id"]])]
        # return out[..., np.sort(out.data.coords[-1])]

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
        return reset_time_index(
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
            X = self._da.data
            T = self._da["time_id"].data.reshape(1, 1, -1)
            if concatenate_time:
                X = sparse.concatenate([X, T], axis=1)
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
            X = ak_dropnan(X)
            T = ak_dropnan(T)
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
        self, reset_time_index=True, ts_level=True, index_scale=1e-9, absolute_time=True
    ):
        if reset_time_index:
            X, _ = self.reset_time_index(
                ts_level=ts_level,
                index_scale=index_scale,
                absolute_time=absolute_time,
                concatenate_time=False,
            )
            return np.concatenate([X.coords, X.data[np.newaxis, :]], axis=0).T
        else:
            T = (
                self._da["time_id"].data.astype(np.float64)[
                    self._da.data.coords[self.dims["time_id"]]
                ]
                * index_scale
            )
            if not absolute_time:
                T = T - T[0]
            return np.concatenate([self._da.data.coords, T[np.newaxis, :]], axis=0).T

    def to_hdf5(self, filename, compression="gzip", compression_opts=None):
        save_to_file(
            data_array=self._da,
            filename=filename,
            compression=compression,
            compression_opts=compression_opts,
        )

