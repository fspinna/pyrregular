from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class Seabirds(ReaderInterface):
    fast_to_test = True

    @staticmethod
    def read_original_version(verbose=False):
        return read_seabirds(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:

        # FIXME: I dont have bird_id in my data_mid
        # # Bird id
        # ts_id_train, _, _, _ = train_test_split(
        #     data["ts_id"],
        #     data["bird"],
        #     test_size=0.3,
        #     stratify=data["bird"],
        #     random_state=42,
        # )
        # ts_id_train = set(ts_id_train.values)
        # train_or_test = [
        #     "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        # ]
        # data = data.assign_coords(split_bird=("ts_id", train_or_test))

        # FIXME: this raises an error
        # # Colony
        # ts_id_train, _, _, _ = train_test_split(
        #     data["ts_id"],
        #     data["colony2"],
        #     test_size=0.3,
        #     stratify=data["colony2"],
        #     random_state=42,
        # )
        # ts_id_train = set(ts_id_train.values)
        # train_or_test = [
        #     "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        # ]
        # data = data.assign_coords(split_colony=("ts_id", train_or_test))

        # Species
        ts_id_train, _, _, _ = train_test_split(
            data["ts_id"],
            data["species"],
            test_size=0.3,
            stratify=data["species"],
            random_state=42,
        )
        ts_id_train = set(ts_id_train.values)
        train_or_test = [
            "train" if i in ts_id_train else "test" for i in data["ts_id"].values
        ]
        data = data.assign_coords(split_default=("ts_id", train_or_test))

        data = data.assign_coords(
            species=("ts_id", LabelEncoder().fit_transform(data["species"])),
        )

        return data


def read_seabirds(verbose=False):  # TODO: check con francesco
    return read_csv(
        filenames=data_original_folder() / "seabirds/seabirds.csv",
        ts_id="tid",
        time_id="date_time",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["bird", "species", "year", "colony2"],
            "signal_id": [],
            "time_id": [],
        },
        verbose=verbose,
    )


if __name__ == "__main__":
    # Seabirds.save_unfixed(False)
    # Seabirds.save_fixed(True)
    #
    # print(Seabirds.read(False))
    # print(Seabirds.load_unfixed())
    # print(Seabirds.load_fixed())

    # df = Seabirds.load_unfixed()

    # Seabirds.save_fixed()
    df = Seabirds.load_final_version()
