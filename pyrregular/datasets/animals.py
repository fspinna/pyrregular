from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class Animals(ReaderInterface):
    fast_to_test = True

    @staticmethod
    def read_original_version(verbose=False):
        return read_animals(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"],
            data["class"],
            test_size=0.3,
            stratify=data["class"],
            random_state=42,
        )
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]
        data = data.assign_coords(
            split_default=("ts_id", train_or_test),
        )

        data = data.assign_coords(
            class_default=("ts_id", LabelEncoder().fit_transform(data["class"]))
        )

        return data


def read_animals(verbose=False):
    return read_csv(
        filenames=data_original_folder() / "animals/animals.csv",
        ts_id="tid",
        time_id="t",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["class"],
            "signal_id": [],
            "time_id": [],
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    # Animals.save_fixed()
    df = Animals.load_final_version()
