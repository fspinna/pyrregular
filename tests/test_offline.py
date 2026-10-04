"""Offline tests on a tiny hand-made dataset (no downloads).

The dataset has 2 series, 2 signals and 5 distinct timestamps:

    series  signal  time  value
    a       s1      1     1.0
    a       s1      2     2.0
    a       s1      3     3.0
    a       s2      2     20.0
    a       s2      3     30.0
    b       s1      10    100.0
    b       s2      10    1000.0
    b       s2      20    2000.0

Global time axis: [1, 2, 3, 10, 20], so the raw array has shape (2, 2, 5).
After resetting the time index, series a has 3 timestamps and series b has 2,
so the dense time axis has length 3.

Tests marked xfail(strict=True) document known bugs: they fail today, and
pytest will error as soon as a fix makes them pass, so the fix must remove
the marker.
"""

import warnings
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

import pyrregular.accessor  # noqa: F401 (registers the .irr accessor)
from pyrregular import list_datasets, list_paper_datasets, list_paper_models
from pyrregular.io_utils import load_from_file, read_csv, save_to_file

nan = np.nan

ROWS = [
    ("a", "s1", 1, 1.0),
    ("a", "s1", 2, 2.0),
    ("a", "s1", 3, 3.0),
    ("a", "s2", 2, 20.0),
    ("a", "s2", 3, 30.0),
    ("b", "s1", 10, 100.0),
    ("b", "s2", 10, 1000.0),
    ("b", "s2", 20, 2000.0),
]

ATTRS = {"configs": {"default": {"target": "y", "split": "s"}}, "name": "toy"}


def _write_csv(path, rows):
    pd.DataFrame(rows, columns=["ts_id", "signal_id", "time_id", "value_id"]).to_csv(
        path, index=False
    )
    return path


@pytest.fixture
def da_csv(tmp_path):
    """DataArray as returned by read_csv."""
    return read_csv(
        _write_csv(tmp_path / "toy.csv", ROWS),
        time_index_as_datetime=False,
        attrs=ATTRS,
    )


@pytest.fixture
def da(da_csv, tmp_path):
    """DataArray after an HDF5 round trip, i.e. what load_dataset returns."""
    save_to_file(da_csv, tmp_path / "toy.h5")
    return load_from_file(tmp_path / "toy.h5")


def assert_equal_nan(actual, expected):
    np.testing.assert_array_equal(np.asarray(actual), np.array(expected))


# --- read_csv ---------------------------------------------------------------


def test_read_csv_layout(da_csv):
    assert da_csv.dims == ("ts_id", "signal_id", "time_id")
    assert da_csv.shape == (2, 2, 5)
    assert list(da_csv.ts_id.values) == ["a", "b"]
    assert list(da_csv.signal_id.values) == ["s1", "s2"]
    assert list(da_csv.time_id.values) == [1, 2, 3, 10, 20]
    assert da_csv.data.nnz == len(ROWS)


def test_read_csv_parses_date_strings_in_time_order(tmp_path):
    # as text these sort as 01/05/2023, 06/15/2022, 12/31/2022
    rows = [
        ("a", "s", "12/31/2022", 1.0),
        ("a", "s", "01/05/2023", 2.0),
        ("a", "s", "06/15/2022", 3.0),
    ]
    da = read_csv(_write_csv(tmp_path / "dates.csv", rows))
    assert da.time_id.dtype.kind == "M"
    assert_equal_nan(
        da.time_id.values.astype("datetime64[D]"),
        np.array(["2022-06-15", "2022-12-31", "2023-01-05"], dtype="datetime64[D]"),
    )
    assert_equal_nan(da.data.todense()[0, 0], [3.0, 1.0, 2.0])


# --- to_dense -----------------------------------------------------------------


def test_to_dense_default(da):
    X, T = da.irr.to_dense(index_scale=1)
    assert_equal_nan(
        X,
        [
            [[1, 2, 3], [nan, 20, 30]],  # a: s2 has no value at t=1
            [[100, nan, nan], [1000, 2000, nan]],  # b: padded to length 3
        ],
    )
    assert_equal_nan(T, [[[1, 2, 3]], [[10, 20, nan]]])  # one time row per series


def test_to_dense_signal_level(da):
    # ts_level=False: each signal gets its own time index, so a/s2 starts at t=2
    X, T = da.irr.to_dense(index_scale=1, ts_level=False)
    assert_equal_nan(
        X,
        [
            [[1, 2, 3], [20, 30, nan]],
            [[100, nan, nan], [1000, 2000, nan]],
        ],
    )
    assert_equal_nan(
        T,
        [
            [[1, 2, 3], [2, 3, nan]],
            [[10, nan, nan], [10, 20, nan]],
        ],
    )


def test_to_dense_relative_time(da):
    _, T = da.irr.to_dense(index_scale=1, absolute_time=False)
    assert_equal_nan(T, [[[0, 1, 2]], [[0, 10, nan]]])


def test_to_dense_normalized_time(da):
    # each series is rescaled to [0, 1); the tiny epsilon avoids division by 0
    _, T = da.irr.to_dense(index_scale=1, normalize_time=True)
    np.testing.assert_allclose(T, [[[0, 0.5, 1]], [[0, 1, nan]]], rtol=1e-6)


def test_to_dense_concatenate_time(da):
    X, T = da.irr.to_dense(index_scale=1, concatenate_time=True)
    assert X.shape == (2, 3, 3)  # time appended as a third signal
    assert_equal_nan(X[:, 2:3, :], T)


def test_to_dense_without_reset(da):
    # T is the global time axis, shared by all series
    X, T = da.irr.to_dense(index_scale=1, reset_time_index=False)
    assert_equal_nan(
        X,
        [
            [[1, 2, 3, nan, nan], [nan, 20, 30, nan, nan]],
            [[nan, nan, nan, 100, nan], [nan, nan, nan, 1000, 2000]],
        ],
    )
    assert_equal_nan(T, [[[1, 2, 3, 10, 20]]])


def test_to_dense_without_reset_rejects_concatenate_time(da):
    with pytest.raises(ValueError, match="requires reset_time_index=True"):
        da.irr.to_dense(reset_time_index=False, concatenate_time=True)


def test_to_dense_directly_after_read_csv(da_csv):
    # read_csv yields Fortran-ordered coords; the numba kernel needs C order
    X, _ = da_csv.irr.to_dense(index_scale=1)
    assert X.shape == (2, 2, 3)


def test_to_dense_passes_shape_to_sparse(da):
    # sparse<0.16 warns when COO gets no shape; sparse>=0.16 raises ValueError
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="shape should be provided")
        da.irr.to_dense()


# --- to_long ------------------------------------------------------------------


def test_to_long_rows(da):
    # the original long format: the toy rows, with labels and original times
    L = da.irr.to_long()
    assert isinstance(L["ts_id"].dtype, pd.CategoricalDtype)
    assert isinstance(L["signal_id"].dtype, pd.CategoricalDtype)
    expected = pd.DataFrame(ROWS, columns=["ts_id", "signal_id", "time_id", "value_id"])
    pd.testing.assert_frame_equal(
        L.astype({"ts_id": str, "signal_id": str}), expected, check_dtype=False
    )


def test_to_long_relative_time(da):
    # per series: a starts at t=1, b at t=10
    L = da.irr.to_long(absolute_time=False)
    assert L["time_id"].tolist() == [0, 1, 2, 1, 2, 0, 0, 10]
    # per series/signal pair: a/s2 starts at t=2
    L = da.irr.to_long(absolute_time=False, ts_level=False)
    assert L["time_id"].tolist() == [0, 1, 2, 0, 1, 0, 0, 10]


# --- .irr[...] ----------------------------------------------------------------


def test_getitem_single_series_and_signal(da):
    out = da.irr[0, 0]
    assert_equal_nan(out.time_id, [1, 2, 3])
    assert_equal_nan(out.data.todense(), [1, 2, 3])


def test_getitem_drops_empty_timestamps(da):
    # s2 across both series: t=1 has no s2 value, so it is dropped
    out = da.irr[:, 1]
    assert_equal_nan(out.time_id, [2, 3, 10, 20])


def test_getitem_no_duplicate_timestamps(da):
    out = da.irr[0]  # series a: t=2 and t=3 exist in both signals
    assert_equal_nan(out.time_id, [1, 2, 3])


# --- HDF5 ---------------------------------------------------------------------


def test_hdf5_roundtrip(da_csv, da):
    assert da.dims == da_csv.dims
    assert_equal_nan(da.data.todense(), da_csv.data.todense())
    for coord in da_csv.coords:
        assert_equal_nan(da[coord].values, da_csv[coord].values)
    assert da.attrs == ATTRS


def test_hdf5_roundtrip_keeps_int64_coords(tmp_path):
    t0 = 1_600_000_000_000_000_000  # epoch nanoseconds stored as plain ints
    rows = [("a", "s", t0 + i, 1.0) for i in range(3)]
    da = read_csv(_write_csv(tmp_path / "big.csv", rows), time_index_as_datetime=False)
    save_to_file(da, tmp_path / "big.h5")
    loaded = load_from_file(tmp_path / "big.h5")
    assert loaded.time_id.dtype == np.int64
    assert_equal_nan(loaded.time_id.values, da.time_id.values)


def test_hdf5_roundtrip_keeps_datetime_unit(da_csv, tmp_path):
    # pandas >= 3 produces datetime64[us]; loading must not read it as ns
    times = np.array(
        [
            "2023-01-01T01:00",
            "2023-01-01T02:00",
            "2023-01-01T03:00",
            "2023-01-02T00:00",
            "2023-01-03T00:00",
        ],
        dtype="datetime64[us]",
    )
    da = da_csv.assign_coords(time_id=times)
    save_to_file(da, tmp_path / "us.h5")
    loaded = load_from_file(tmp_path / "us.h5")
    assert loaded.time_id.dtype == "datetime64[ns]"
    assert_equal_nan(loaded.time_id.values, times.astype("datetime64[ns]"))


def test_hdf5_attrs_are_not_executed(da_csv, tmp_path):
    marker = tmp_path / "executed"
    path = tmp_path / "evil.h5"
    save_to_file(da_csv, path)
    with h5py.File(path, "a") as f:
        f.attrs["evil"] = f"open({str(marker)!r}, 'w').close()"
    loaded = load_from_file(path)
    assert not marker.exists()
    assert isinstance(loaded.attrs["evil"], str)


# --- dataset from docs/notebooks/dataset_conversion.ipynb --------------------

DOCS = Path(__file__).parent.parent / "docs" / "notebooks"


def _read_docs_csv():
    # same call as in dataset_conversion.ipynb
    return read_csv(
        filenames=DOCS / "your_original_dataset.csv",
        ts_id="time_series_id",
        time_id="timestamp",
        signal_id="channel_id",
        value_id="value",
        dims={"ts_id": ["labels"], "signal_id": [], "time_id": []},
        time_index_as_datetime=True,
    )


def test_docs_csv_matches_committed_h5():
    da = _read_docs_csv()
    ref = load_from_file(DOCS / "your_dataset.h5")
    assert da.shape == ref.shape == (3, 3, 100)
    assert_equal_nan(da.data.todense(), ref.data.todense())
    for coord in ref.coords:
        assert_equal_nan(da[coord].values, ref[coord].values)


def test_docs_h5_to_dense():
    X, T = load_from_file(DOCS / "your_dataset.h5").irr.to_dense()
    assert X.shape[:2] == (3, 3)


def test_docs_to_long_static_roundtrip(tmp_path):
    # to_long(static=True) gives back a file that read_csv turns into the same data
    ref = load_from_file(DOCS / "your_dataset.h5")
    df = ref.irr.to_long(static=True)
    assert list(df.columns) == ["ts_id", "signal_id", "time_id", "value_id", "labels"]
    df.to_csv(tmp_path / "long.csv", index=False)
    back = read_csv(
        tmp_path / "long.csv",
        dims={"ts_id": ["labels"], "signal_id": [], "time_id": []},
        time_index_as_datetime=True,
    )
    assert_equal_nan(back.data.todense(), ref.data.todense())
    for coord in ref.coords:
        assert_equal_nan(back[coord].values, ref[coord].values)


# --- dataset lists ------------------------------------------------------------


def test_list_datasets_includes_first_registry_line():
    registry = DOCS.parent.parent / "pyrregular" / "registry.txt"
    names = [line.split()[0] for line in registry.read_text().splitlines() if line]
    assert list_datasets() == sorted(names)
    assert "InsectWingbeat.h5" in list_datasets()


def test_list_paper_datasets():
    paper = list_paper_datasets()
    assert len(paper) == 34
    assert set(paper) <= set(list_datasets())
    assert set(list_datasets()) - set(paper) == {
        "Ais.h5",
        "CombinedTrajectories.h5",
        "Geolife.h5",
        "TDrive.h5",
    }


def test_version():
    import pyrregular

    assert isinstance(pyrregular.__version__, str) and pyrregular.__version__


def test_read_csv_rejects_non_static_columns(tmp_path):
    # label changes inside series a: it is not static, so read_csv must not split a
    rows = [
        ("a", "s", 1, 1.0, "x"),
        ("a", "s", 2, 2.0, "x"),
        ("a", "s", 3, 3.0, "y"),
        ("b", "s", 1, 5.0, "z"),
    ]
    path = tmp_path / "static.csv"
    pd.DataFrame(
        rows, columns=["ts_id", "signal_id", "time_id", "value_id", "label"]
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="'label' is not static for ts_id 'a'"):
        read_csv(
            path,
            dims={"ts_id": ["label"], "signal_id": [], "time_id": []},
            time_index_as_datetime=False,
        )


def test_api_token_is_sent(monkeypatch):
    # capture the downloader instead of downloading
    from pyrregular import repository

    seen = {}

    def fake_fetch(name, downloader):
        seen["name"], seen["downloader"] = name, downloader
        return "fake/path.h5"

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(repository.REPOSITORY, "fetch", fake_fetch)
    assert (
        repository.load_dataset_from_file("Garment", api_token="hf_x") == "fake/path.h5"
    )
    assert seen["name"] == "Garment.h5"
    assert seen["downloader"].kwargs["headers"] == {"Authorization": "Bearer hf_x"}


def test_fill_time_index():
    from pyrregular.conversion_utils import _fill_time_index

    T = np.array(
        [
            [[0.0, 1.0, 3.0, nan, nan]],  # mean step 1.5
            [[5.0, nan, nan, nan, nan]],  # one timestamp: no step, use nan_delta
        ]
    )
    assert_equal_nan(_fill_time_index(T)[:, 0, :], [[0, 1, 3, 4.5, 6], [5, 6, 7, 8, 9]])
    assert_equal_nan(_fill_time_index(T, nan_delta=2)[1, 0], [5, 7, 9, 11, 13])


def test_list_paper_models():
    # same names as the columns of the paper's result tables
    header = (DOCS.parent.parent / "assets/results/accuracy_mean.csv").read_text()
    assert list_paper_models() == header.splitlines()[0].split(",")[2:]
    # every entry points to an existing pipeline, checked without importing it
    from pyrregular.models import PAPER_MODELS

    for path in PAPER_MODELS.values():
        module, pipeline = path.split(":")
        source = (DOCS.parent.parent / f"pyrregular/models/{module}.py").read_text()
        assert f"\n{pipeline} = " in source


def test_downloads_are_pinned_to_a_data_version():
    import pooch

    from pyrregular import repository

    # a fixed commit of the Hugging Face repo, never the moving main branch
    assert f"/resolve/{repository.DATA_REVISION}/" in repository.REPOSITORY.base_url
    assert "/resolve/main/" not in repository.REPOSITORY.base_url
    assert len(repository.DATA_REVISION) == 40
    # data-v1 keeps the original cache folder (no re-download for existing users)
    if repository.DATA_VERSION == "data-v1":
        assert str(repository.REPOSITORY.path) == str(pooch.os_cache("pyrregular"))
