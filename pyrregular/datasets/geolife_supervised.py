import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class GeolifeSupervised(ReaderInterface):

    @staticmethod
    def read_original_version(verbose=False):
        return read_geolife_supervised(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"],
            data["label"],
            test_size=0.3,
            stratify=data["label"],
            random_state=42,
        )
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]
        data = data.assign_coords(split_default=("ts_id", train_or_test))

        # Preparing target class for 2/2 task
        data = data.assign_coords(
            class_default=("ts_id", LabelEncoder().fit_transform(data["label"]))
        )

        mapping = {  # 0 private transportation, 1 otherwise
            "subway": 1,
            "airplane": 1,
            "train": 1,
            "taxi": 1,
            "bus": 1,
            "motorcycle": 0,
            "run": 0,
            "walk": 0,
            "boat": 0,
            "car": 0,
            "bike": 0,
        }

        data = data.assign_coords(
            class_simplified=("ts_id", [mapping[x] for x in data["label"].values]),
        )

        return data


def read_geolife_supervised(verbose=False):
    return read_csv(
        filenames=data_original_folder() / "geolife_supervised/geolife_supervised.csv",
        ts_id="tid",
        time_id="time",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["user", "label"],
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_read_geolife_supervised,
        verbose=verbose,
    )


def _read_geolife_supervised(filenames: list):
    filename = filenames[0]

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filename)

    melted_df = df.melt(
        id_vars=["tid", "label", "user", "time"], value_vars=["lat", "lon"]
    )

    for row in melted_df.to_dict(orient="records"):
        yield row


if __name__ == "__main__":
    # GeolifeSupervised.save_unfixed(True)
    # GeolifeSupervised.save_fixed(True)
    #
    # print(GeolifeSupervised.read(False))
    # print(GeolifeSupervised.load_unfixed())
    # print(GeolifeSupervised.load_fixed())
    # GeolifeSupervised.save_fixed()
    df = GeolifeSupervised.load_final_version()
