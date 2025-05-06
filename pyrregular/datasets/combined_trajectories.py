from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class CombinedTrajectories(ReaderInterface):
    fast_to_test = False

    @staticmethod
    def read_original_version(verbose=False):
        return read_combined_trajectories(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        instances_per_class = (
            data["individual_id"]
            .to_dataframe()
            .iloc[:, :1]
            .groupby(by="individual_id", as_index=False)
            .size()
        )
        individual_to_consider = instances_per_class[
            instances_per_class["size"] > 3
        ].individual_id.tolist()
        data = data[data.individual_id.isin(individual_to_consider)]

        le = LabelEncoder()
        y = le.fit_transform(data["individual_id"].values)

        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"],
            y,
            test_size=0.3,
            stratify=y,
            random_state=42,
        )
        ts_id_train = set(ts_id_train.values)
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]

        data = data.assign_coords(
            split_default=("ts_id", train_or_test),
            class_default=("ts_id", y),
        )

        return data


def __convert_date(x):
    try:
        return datetime.strptime(" ".join(x), "%Y-%m-%d %H:%M:%S").timestamp()
    except:
        return np.nan


def _read_combined_trajectories(filenames: list):
    filename = filenames[0]

    dtypes = {
        "latitude": float,
        "longitude": float,
        "altitude": float,
        "date": str,
        "time": str,
        "individual_id": int,
        "trajectory_id": int,
    }
    # Read the CSV file into a Pandas DataFrame
    for df in pd.read_csv(filename, chunksize=10**3, dtype=dtypes):
        df["timestamp"] = df[["date", "time"]].apply(
            __convert_date,
            axis=1,
        )

        df.drop(columns=["date", "time"], inplace=True)
        df["absolute_trajectory_id"] = df[["individual_id", "trajectory_id"]].apply(
            lambda x: "_".join(x.astype(str)), axis=1
        )

        melted_df = df.melt(
            id_vars=[
                "absolute_trajectory_id",
                "trajectory_id",
                "individual_id",
                "timestamp",
            ],
            value_vars=["longitude", "latitude", "altitude"],
        ).dropna()

        for row in melted_df.to_dict(orient="records"):
            yield row


def read_combined_trajectories(verbose=False):
    return read_csv(
        filenames=data_original_folder()
        / "combined_trajectories/combined_trajectories.csv",
        ts_id="absolute_trajectory_id",
        time_id="timestamp",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["trajectory_id", "individual_id"],
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_read_combined_trajectories,
        verbose=verbose,
    )


if __name__ == "__main__":
    # CombinedTrajectories.save_unfixed(True)
    # CombinedTrajectories.save_fixed(True)
    #
    # # print(Taxi.read(False))
    # # print(CombinedTrajectories.load_unfixed())
    # df = CombinedTrajectories.load_unfixed()
    # print()

    df = CombinedTrajectories.load_final_version()
    print()
