import glob

import numpy as np
import pandas as pd
import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class Pamap2(ReaderInterface):

    @staticmethod
    def read_original_version(verbose) -> DataArray:
        return read_pamap(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        activity, activity_count = np.unique(data["activity_name"], return_counts=True)
        activity = activity[activity_count > 1]

        data = data.where(data.activity_name.isin(activity), drop=True)
        # as suggested by the authors, we discard all the 0s activities (transient activities)
        data = data.where(data.activity_name != "transient", drop=True)

        le = LabelEncoder()
        y = le.fit_transform(data["activity_id"].values)

        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"], y, test_size=0.3, stratify=y, random_state=42
        )
        ts_id_train = set(ts_id_train.values)
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]
        data = data.assign_coords(
            split_default=("ts_id", train_or_test), class_default=("ts_id", y)
        )

        return data


activity_names = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}


def get_column_names():
    column_names = ["timestamp_s", "activity_id", "heart_rate"]
    IMU = [
        "temperature",
        "acceleration_16g_0",
        "acceleration_16g_1",
        "acceleration_16g_2",
        "acceleration_6g_0",
        "acceleration_6g_1",
        "acceleration_6g_2",
        "gyroscope_0",
        "gyroscope_1",
        "gyroscope_2",
        "magnetometer_0",
        "magnetometer_1",
        "magnetometer_2",
        "orientation_0",
        "orientation_1",
        "orientation_2",
        "orientation_3",
    ]
    for imu_position in ["hand", "chest", "ankle"]:
        column_names += [f"{imu_position}_{signal_name}" for signal_name in IMU]

    return column_names


def _read_pamap(filenames, verbose=True):
    count = -1
    last_values = None
    for file in tqdm(
        filenames,
        disable=not verbose,
        desc="reading PAMAP2 files",
        position=0,
        leave=False,
    ):
        subject_id = file.split("/")[-2][0] + file[:-4].split("subject")[-1]

        df_subject = pd.read_table(
            file, sep="\s+", header=None, names=get_column_names()
        )
        df_subject["activity_name"] = df_subject.activity_id.astype(int).apply(
            lambda x: activity_names[x]
        )
        df_subject["subject_id"] = subject_id

        ts_ids = []
        for row in tqdm(
            df_subject[["activity_id", "subject_id"]].itertuples(index=False),
            disable=not verbose,
            total=len(df_subject),
            desc="partitioning the data",
            position=1,
            leave=False,
        ):
            if row != last_values:
                last_values = row
                count += 1
            ts_ids.append(count)
        df_subject["ts_id"] = ts_ids

        melted_df_subject = df_subject.melt(
            id_vars=[
                "ts_id",
                "timestamp_s",
                "activity_id",
                "activity_name",
                "subject_id",
            ],
            value_vars=get_column_names()[1:],
        )

        for row in melted_df_subject.to_dict(orient="records"):
            yield row


def read_pamap(verbose=True):
    return read_csv(
        filenames=glob.glob(str(data_original_folder()) + "/PAMAP2/*/*.dat"),
        ts_id="ts_id",
        time_id="timestamp_s",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["activity_id", "activity_name", "subject_id"],
            "signal_id": [],
            "time_id": ["timestamp_s"],
        },
        reader_fun=_read_pamap,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Pamap2.save_unfixed(verbose=True)

    # Pamap2.save_fixed(verbose=True)
    dataset = Pamap2.load_final_version()
    print()

    # print(PAMAP2.read(False))
    # print(PAMAP2.load_unfixed())
    # print(PAMAP2.load_fixed())
