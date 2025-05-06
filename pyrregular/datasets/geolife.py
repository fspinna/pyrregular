import os
from glob import glob

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class Geolife(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_geolife(verbose=verbose)


def read_geolife(verbose=False):
    return read_csv(
        filenames=data_original_folder() / "geolife/Data/*",
        ts_id="tid",
        time_id="time",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["user"],
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_read_geolife,
        verbose=verbose,
    )


mode_names = [
    "walk",
    "bike",
    "bus",
    "car",
    "subway",
    "train",
    "airplane",
    "boat",
    "run",
    "motorcycle",
    "taxi",
]


def _read_plt(plt_file):
    points = pd.read_csv(plt_file, skiprows=6, header=None, parse_dates=[[5, 6]])

    # for clarity rename columns
    points.rename(inplace=True, columns={"5_6": "time", 0: "lat", 1: "lon", 3: "alt"})

    # points.time = points.time.apply(lambda x: x.timestamp())
    points.alt = points.alt.apply(lambda x: np.nan if x == -777 else x)

    # remove unused columns
    points.drop(inplace=True, columns=[2, 4])

    points["tid"] = plt_file.split("/")[-1][:-4]

    return points


def _read_labels(labels_file):
    mode_ids = {s: i + 1 for i, s in enumerate(mode_names)}

    labels = pd.read_csv(
        labels_file,
        skiprows=1,
        header=None,
        parse_dates=[[0, 1], [2, 3]],
        delim_whitespace=True,
    )

    # for clarity rename columns
    labels.columns = ["start_time", "end_time", "label"]

    # replace 'label' column with integer encoding
    labels["label"] = [mode_ids[i] for i in labels["label"]]

    return labels


def apply_labels(points, labels):
    indices = labels["start_time"].searchsorted(points["time"], side="right") - 1
    no_label = (indices < 0) | (
        points["time"].values >= labels["end_time"].iloc[indices].values
    )
    points["label"] = labels["label"].iloc[indices].values
    points.loc[no_label, "label"] = np.NaN


def _read_user(user_folder):
    plt_files = glob(os.path.join(user_folder, "Trajectory", "*.plt"))
    df = pd.concat([_read_plt(f) for f in plt_files])

    labels_file = os.path.join(user_folder, "labels.txt")
    if os.path.exists(labels_file):
        labels = _read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df["label"] = np.NAN

    return df


def _read_all_users(folder):
    subfolders = glob(folder)
    dfs = []
    bar = tqdm(subfolders, leave=False, disable=True)
    for i, sf in enumerate(bar):
        bar.set_description(
            "[%d/%d] processing user %s" % (i + 1, len(subfolders), sf.split("/")[-1])
        )
        df = _read_user(sf)
        df["user"] = int(sf.split("/")[-1])
        dfs.append(df)
    return pd.concat(dfs)


def _read_geolife(filenames: list):
    filename = filenames[0]

    # Read the CSV file into a Pandas DataFrame
    df = _read_all_users(filename)

    df.time = df.time.apply(lambda x: x.timestamp())
    df.label = df.label.apply(
        lambda x: mode_names[int(x) - 1] if not np.isnan(x) else x
    )

    prec_row = [-1 for _ in df.columns]
    tid = -1

    tids = []

    for row in df.values:
        if (
            row[-1] != prec_row[-1]
            or row[-2] != prec_row[-2]
            or row[-3] != prec_row[-3]
        ):
            prec_row = row
            tid += 1

        tids.append(tid)

    df["tid"] = tids
    del tids

    melted_df = df.melt(
        id_vars=["tid", "user", "time"],
        value_vars=[
            "lat",
            "lon",
            "alt",
        ],
    )

    for row in melted_df.to_dict(orient="records"):
        yield row


if __name__ == "__main__":
    # Geolife.save_unfixed(True)
    # Geolife.save_fixed(True)

    # print(Geolife.read(False))
    # print(Geolife.load_unfixed())
    print(Geolife.load_final_version())
