import numpy as np
import sparse
import xarray as xr
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from pyrregular.data_utils import data_final_folder, data_original_folder
from pyrregular.io_utils import get_current_aoe_time, save_to_file

univariate_variable_length = [
    "AllGestureWiimoteX",
    "AllGestureWiimoteY",
    "AllGestureWiimoteZ",
    "GestureMidAirD1",
    "GestureMidAirD2",
    "GestureMidAirD3",
    "GesturePebbleZ1",
    "GesturePebbleZ2",
    "PickupGestureWiimoteZ",
    "PLAID",
    "ShakeGestureWiimoteZ",
]

# 4 fixed length univariate time series classification problems with missing values"""
univariate_missing_values = [
    "DodgerLoopDay",
    "DodgerLoopGame",
    "DodgerLoopWeekend",
    "MelbournePedestrian",
]

# 7 variable length multivariate time series classification problems [4]"""
multivariate_unequal_length = [
    "AsphaltObstaclesCoordinates",
    "AsphaltPavementTypeCoordinates",
    "AsphaltRegularityCoordinates",
    "CharacterTrajectories",
    "InsectWingbeat",
    "JapaneseVowels",
    "SpokenArabicDigits",
]


UEA_UCR_DATASETS = (
    univariate_variable_length + univariate_missing_values + multivariate_unequal_length
)


def save_fixed_dataset(dataset_name):
    X_coo = sparse.load_npz(
        data_original_folder() / dataset_name / f"{dataset_name}.npz"
    )
    split = np.load(data_original_folder() / dataset_name / f"{dataset_name}_split.npy")
    target = np.load(
        data_original_folder() / dataset_name / f"{dataset_name}_target.npy"
    )
    le = LabelEncoder()
    target_num = le.fit_transform(target)
    metadata = dict(
        title=dataset_name,
        authors="TODO",
        license="TODO",
        source="https://timeseriesclassification.com/",
        configs=dict(
            default=dict(
                task="classification",
                split="split_default",
                target="class_default",
            )
        ),
        _is_fixed=True,
        _fixed_at=get_current_aoe_time(),
    )
    df = xr.DataArray(
        X_coo,
        dims=["ts_id", "signal_id", "time_id"],
        coords={
            "ts_id": np.arange(X_coo.shape[0]),
            "signal_id": np.arange(X_coo.shape[1]),
            "time_id": np.arange(X_coo.shape[2]),
        },
    )
    df = df.assign_coords(
        split_default=("ts_id", split),
        class_default=("ts_id", target_num),
        label=("ts_id", target),
    )
    df.attrs = metadata
    save_to_file(
        data_array=df,
        filename=data_final_folder() / (dataset_name + ".h5"),
    )


def save_fixed_datasets():
    for dataset_name in tqdm(UEA_UCR_DATASETS):
        save_fixed_dataset(dataset_name)


if __name__ == "__main__":
    save_fixed_datasets()
