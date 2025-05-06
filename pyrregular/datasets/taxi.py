import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class Taxi(ReaderInterface):
    fast_to_test = False

    @staticmethod
    def read_original_version(verbose=False):
        return read_taxi(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"],
            data["class"],
            test_size=0.3,
            stratify=data["class"],
            random_state=42,
        )
        ts_id_train = set(ts_id_train.values)
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]
        data = data.assign_coords(split_default=("ts_id", train_or_test))

        data = data.assign_coords(
            class_default=("ts_id", LabelEncoder().fit_transform(data["class"])),
        )

        return data


def _read_taxi(filenames: list):
    filename = filenames[0]

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename).rename(columns={"c1": "longitude", "c2": "latitude"})

    melted_df = df.melt(
        id_vars=["tid", "class", "t"], value_vars=["longitude", "latitude"]
    )

    for row in melted_df.to_dict(orient="records"):
        yield row


def read_taxi(verbose=False):
    return read_csv(
        filenames=data_original_folder() / "taxi/taxi.csv",
        ts_id="tid",
        time_id="t",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["class"],
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_read_taxi,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Taxi.save_unfixed(False)
    # Taxi.save_fixed(True)

    # print(Taxi.read(False))
    # print(Taxi.load_unfixed())
    # print(Taxi.load_fixed())
    Taxi.save_final_version()
    df = Taxi.load_final_version()
