import pandas as pd
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import (
    load_yaml,
    read_csv,
)
from pyrregular.reader_interface import ReaderInterface


class Garment(ReaderInterface):
    fast_to_test = True

    @staticmethod
    def read_original_version(verbose=False):
        return read_garment(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        productivity_numerical = data.loc[:, "actual_productivity"][:, -1].to_numpy()
        productivity_binary = (productivity_numerical > 0.75) * 1
        mapping = {0: "low", 1: "high"}
        productivity_class = [mapping[x] for x in productivity_binary]
        split = [
            (
                "test"
                if i
                in [
                    "finishing_3",
                    "finishing_7",
                    "finishing_11",
                    "sweing_1",
                    "sweing_6",
                    "sweing_12",
                ]
                else "train"
            )
            for i in data["ts_id"]
        ]
        data = data.drop_sel(dict(signal_id="actual_productivity"))
        data = data.assign_coords(
            productivity_numerical=("ts_id", productivity_numerical),
            productivity_class=("ts_id", productivity_class),
            productivity_binary=("ts_id", productivity_binary),
            split=("ts_id", split),
        )
        return data


def _dataset_garment(filenames: list):
    filenames = filenames[0]

    df = pd.read_csv(filenames)

    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y").apply(
        lambda x: x.timestamp()
    )
    df.department = df.department.str.replace(" ", "")
    df["ts_id"] = df.department + "_" + df.team.astype(str)

    df = df.melt(id_vars=["date", "ts_id"] + ["department", "team", "quarter", "day"])

    for row in df.to_dict(orient="records"):
        yield row


def read_garment(verbose=False):
    attrs = load_yaml(
        str(data_original_folder() / "garments_worker_productivity/attrs.yml")
    )
    return read_csv(
        filenames=data_original_folder()
        / "garments_worker_productivity/garments_worker_productivity.csv",
        ts_id="ts_id",
        time_id="date",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["department", "team"],
            "signal_id": [],
            "time_id": ["quarter", "day"],
        },
        reader_fun=_dataset_garment,
        attrs=attrs,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Garment.save_fixed()
    df = Garment.load_final_version()
