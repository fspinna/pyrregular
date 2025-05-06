from datetime import datetime, timedelta, timezone
from os import PathLike
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import sparse
import xarray as xr
import yaml
from tqdm.auto import tqdm
from xarray import DataArray


def save_to_file(
    data_array: xr.DataArray, filename: str, compression="gzip", compression_opts=1
):
    with h5py.File(filename, "w") as f:
        # Save sparse data and coordinates
        f.create_dataset(
            "data",
            data=data_array.data.data,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.create_dataset(
            "coords",
            data=data_array.data.coords,
            compression=compression,
            compression_opts=compression_opts,
        )
        f.attrs["__shape"] = data_array.data.shape

        # Save xarray dimensions
        f.create_dataset("dims", data=np.array(data_array.dims, dtype="S"))

        # Save xarray coordinates
        for coord, values in data_array.coords.items():
            dims = data_array[coord].dims
            values = data_array[coord].values
            if values.dtype.kind == "M":  # Check for datetime dtype
                values = values.astype("int64")  # Convert to Unix timestamp
                f.attrs[f"__{coord}/type"] = "M"
            elif values.dtype.kind == "U":  # Check for Unicode dtype
                values = values.astype("S")  # Convert to byte string
                f.attrs[f"__{coord}/type"] = "U"
            else:
                f.attrs[f"__{coord}/type"] = values.dtype.kind
            f.create_dataset(f"coords_xarray/{coord}", data=values)
            f.create_dataset(
                f"coords_xarray_dims/{coord}", data=np.array(dims, dtype="S")
            )

        # Save metadata
        for key, value in data_array.attrs.items():
            f.attrs[key] = str(value)


def load_from_file(filename: str) -> xr.DataArray:
    with h5py.File(filename, "r") as f:
        # Load sparse data and coordinates
        data = f["data"][:]
        coords = f["coords"][:]
        shape = tuple(f.attrs["__shape"])
        s = sparse.COO(coords, data, shape=shape, fill_value=np.nan)
        dims = f["dims"][:].astype("U")

        # Load metadata (except for development metadata)
        # TODO: remove the eval function and use a safer way to deserialize the metadata
        attrs = {}
        for key in f.attrs.keys():
            if key[:2] == "__":
                continue  # Skip special attributes
            value = f.attrs[key]
            try:
                deserialized_value = eval(
                    value
                )  # Try to interpret the string as a Python literal
                attrs[key] = deserialized_value
            except (SyntaxError, NameError):
                attrs[key] = value  # If eval fails, use the original string

        # Load xarray dimension coordinates
        coords_xarray = {}
        for coord in f["coords_xarray"].keys():
            values = f[f"coords_xarray/{coord}"][:]
            kind = f.attrs[f"__{coord}/type"]
            values = values.astype(kind)
            if kind == "M":
                values = values.astype(
                    "datetime64[ns]"
                )  # TODO: we need to make sure this works also for other kinds of datetime
            elif kind == "O":
                try:
                    values = values.astype("U")
                except ValueError:
                    pass
            else:
                values = values.astype(kind)
            coords_xarray[coord] = (
                tuple(f[f"coords_xarray_dims/{coord}"][:].astype("U")),
                values,
            )
    da = xr.DataArray(data=s, dims=dims, coords=coords_xarray, attrs=attrs)
    return da


def load_yaml(filename: str):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def _read_csvs(filenames: list, chunksize=1000, reader_fun=None, **kwargs):
    for filename in filenames:
        for df in pd.read_csv(filename, chunksize=chunksize):
            if reader_fun is None:
                for row in df.to_dict(orient="records"):
                    yield row
            else:
                reader_fun(df, **kwargs)


def _parse_filenames(filenames):
    if isinstance(filenames, str):
        filenames = [filenames]
    elif isinstance(filenames, PathLike):
        filenames = [str(filenames)]
    elif isinstance(filenames, dict):
        pass
    return filenames


def _get_metadata(
    filenames: str | list | dict,
    ts_ids: list,
    time_ids: list,
    signal_ids: list,
    reader_fun=_read_csvs,
    verbose=False,
    **kwargs,
):
    dates = set()
    ids = set()
    signals = set()
    n_records = 0

    for row in tqdm(
        reader_fun(filenames=_parse_filenames(filenames), **kwargs),
        disable=not verbose,
        desc="Getting dataset metadata",
    ):
        dates.add(tuple([row[time_id] for time_id in time_ids]))
        ids.add(tuple([row[ts_id] for ts_id in ts_ids]))
        signals.add(tuple([row[signal_id] for signal_id in signal_ids]))
        n_records += 1

    return dates, ids, signals, n_records


def read_csv(
    filenames: str | list | dict,
    ts_id: str = "ts_id",
    signal_id="signal_id",
    time_id="time_id",
    value_id="value_id",
    dims=None,  # {dim1: [coords...], dim2: [coords...], ...}
    reader_fun=_read_csvs,
    time_index_as_datetime=True,
    verbose=False,
    attrs=None,
    **kwargs,
):
    if attrs is None:
        attrs = {}

    # get metadata
    dates, ids, signals, n_records = _get_metadata(
        filenames=_parse_filenames(filenames),
        ts_ids=[ts_id] + dims["ts_id"],
        signal_ids=[signal_id] + dims["signal_id"],
        time_ids=[time_id] + dims["time_id"],
        reader_fun=reader_fun,
        verbose=verbose,
        **kwargs,
    )

    # create mapping from idx to id for each dimension
    dims_mapping = dict(
        time_id=dict([(k, v) for v, k in enumerate(sorted(dates))]),
        ts_id=dict([(k, v) for v, k in enumerate(sorted(ids))]),
        signal_id=dict([(k, v) for v, k in enumerate(sorted(signals))]),
    )

    sparse_matrix = sparse.DOK(
        shape=(len(ids), len(signals), len(dates)), fill_value=np.nan
    )

    for row in tqdm(
        reader_fun(filenames=_parse_filenames(filenames), **kwargs),
        total=n_records,
        disable=not verbose,
        desc="Reading dataset",
    ):
        date = tuple([row[id] for id in [time_id] + dims["time_id"]])
        signal = tuple([row[id] for id in [signal_id] + dims["signal_id"]])
        id = tuple([row[id] for id in [ts_id] + dims["ts_id"]])
        value = row[value_id]

        # these should not happen because a nan value should be omitted in the long format but just in case
        if value == "":
            value = np.nan
        elif pd.isna(value):
            value = np.nan

        sparse_matrix[
            dims_mapping["ts_id"][id],
            dims_mapping["signal_id"][signal],
            dims_mapping["time_id"][date],
        ] = value

    if time_index_as_datetime:
        date = [
            datetime.fromtimestamp(float(d[0])) for d in dims_mapping["time_id"].keys()
        ]
    else:
        date = [d[0] for d in dims_mapping["time_id"].keys()]

    xarr = DataArray(
        sparse_matrix.to_coo(),
        dims=("ts_id", "signal_id", "time_id"),
        coords=dict(
            [
                ("ts_id", [id[0] for id in dims_mapping["ts_id"]]),
                ("signal_id", [id[0] for id in dims_mapping["signal_id"]]),
                ("time_id", date),
            ]
            + [
                (ts_some, ("ts_id", [k[i + 1] for k in dims_mapping["ts_id"]]))
                for i, ts_some in enumerate(dims["ts_id"])
            ]
            + [
                (
                    signal_some,
                    ("signal_id", [k[i + 1] for k in dims_mapping["signal_id"]]),
                )
                for i, signal_some in enumerate(dims["signal_id"])
            ]
            + [
                (time_some, ("time_id", [k[i + 1] for k in dims_mapping["time_id"]]))
                for i, time_some in enumerate(dims["time_id"])
            ]
        ),
        attrs=attrs,
    )
    xarr["ts_id"] = xarr["ts_id"].astype("str")
    xarr["signal_id"] = xarr["signal_id"].astype("str")

    # Replace None with np.nan
    for coord in xarr.coords:
        if coord in ["ts_id", "signal_id", "time_id"]:
            continue
        else:
            xarr[coord].values = np.array(
                [np.nan if v is None else v for v in xarr[coord].values]
            )

    return xarr


def get_current_aoe_time():
    aoe_tz = timezone(timedelta(hours=-12))
    current_time_utc = datetime.utcnow()
    current_time_aoe = current_time_utc.replace(tzinfo=timezone.utc).astimezone(aoe_tz)
    return current_time_aoe.isoformat()
