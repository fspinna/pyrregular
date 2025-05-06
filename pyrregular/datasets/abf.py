from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew, skewnorm
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xarray import DataArray

from pyrregular.data_utils import data_final_folder, data_original_folder
from pyrregular.io_utils import load_yaml, read_csv
from pyrregular.reader_interface import ReaderInterface

LABEL_MAP = {0: "bowl", 1: "alembic", 2: "flask"}

RANDOM_STATE = 42


class Abf(ReaderInterface):
    fast_to_test = True

    @staticmethod
    def read_original_version(verbose=False):
        return read_abf(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        split_default = ["train" if i == 1 else "test" for i in data["split"]]
        data = data.assign_coords(split_default=("ts_id", split_default))
        # the following is necessary to make the data compatible with some model benchmarks
        data = data.assign_coords(
            time_id=("time_id", (data["time_id"].to_numpy() * 100_000_000).astype(int))
        )
        data = data.rename({"class_default": "class_labels"})
        data = data.rename({"y": "class_default"})
        return data


def _sample_skew_data(a, size):
    return skewnorm.rvs(a, size=size)


def _shape_function(n):
    x_values = np.linspace(0, 1, n)
    theta_values = x_values * np.pi
    y_values = -np.sin(theta_values)
    y_values = (y_values - y_values.mean()) / y_values.std()
    return y_values


def _generate_time_instances(instances_per_class, size, skewness):
    values = np.sort(
        [_sample_skew_data(skewness, size) for _ in range(instances_per_class)]
    )
    values = (values - values.min(axis=1, keepdims=True)) / (
        values.max(axis=1, keepdims=True) - values.min(axis=1, keepdims=True)
    )
    return values


def _generate_abf_data(
    instances_per_class=10,
    size=128,
    skewness=10,
    random_state=None,
    noise=True,
    noise_level=0.15,
    n_train_instances_per_class=10,
):
    np.random.seed(random_state)
    y = _shape_function(n=size)
    y = y.reshape(1, -1)
    Y = np.repeat(y, instances_per_class * 3, axis=0)
    if noise:
        Y += np.random.normal(0, noise_level, size=(instances_per_class * 3, size))
    cylinders_t = _generate_time_instances(instances_per_class, size, 0)
    bells_t = _generate_time_instances(instances_per_class, size, skewness)
    funnels_t = _generate_time_instances(instances_per_class, size, -skewness)
    split_vector = _get_train_test_split(
        instances_per_class * 3, n_train_instances_per_class
    )
    times = np.concatenate((cylinders_t, bells_t, funnels_t))
    skewness_real = skew(times, axis=1)
    return (
        times,
        np.concatenate(
            (
                np.zeros(instances_per_class),
                np.ones(instances_per_class),
                np.full(instances_per_class, 2),
            )
        ),
        Y,
        skewness_real,
        split_vector,
    )


def _generate_long_abf(
    instances_per_class=10,
    size=128,
    skewness=10,
    random_state=None,
    noise=True,
    noise_level=0.15,
):
    return _abf_to_long(
        *_generate_abf_data(
            instances_per_class, size, skewness, random_state, noise, noise_level
        )
    )


def _save_long_icbf(
    instances_per_class=10,
    size=128,
    skewness=10,
    random_state=RANDOM_STATE,
    noise=True,
    noise_level=0.25,
):
    df = _generate_long_abf(
        instances_per_class, size, skewness, random_state, noise, noise_level
    )
    df.to_csv(
        data_original_folder() / "alembics_bowls_flasks" / "abf_long.csv", index=False
    )
    return


def _abf_to_long(t, y, X, skewness, split_vector):
    df = pd.DataFrame()
    for i in tqdm(range(X.shape[0])):
        for j in range(X.shape[1]):
            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        {
                            "time_id": t[i, j],
                            "y": int(y[i]),
                            "value": X[i, j],
                            "class_default": LABEL_MAP[y[i]],
                            "skewness": skewness[i],
                            "ts_id": str(i),
                            "signal_id": str(0),
                            "split": split_vector[i],
                        },
                        index=[0],
                    ),
                ]
            )
    return df


def _get_train_test_split(n_instances, n_instances_train_per_class=10):
    starting_idxs = np.arange(0, n_instances, n_instances // 3)
    split_vector = np.zeros(n_instances, dtype=int)
    for idx in starting_idxs:
        split_vector[idx : idx + n_instances_train_per_class] = 1
    return split_vector


def read_abf(verbose=False):
    return read_csv(
        filenames=[data_original_folder() / "alembics_bowls_flasks/abf_long.csv"],
        ts_id="ts_id",
        time_id="time_id",
        signal_id="signal_id",
        value_id="value",
        dims={
            "ts_id": ["y", "class_default", "split", "skewness"],
            "signal_id": [],
            "time_id": [],
        },
        verbose=verbose,
        time_index_as_datetime=False,
    )


if __name__ == "__main__":
    # Abf.save_fixed()
    df = Abf.load_final_version()
