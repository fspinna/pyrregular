from datetime import datetime
from glob import glob

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm.auto import tqdm
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class TDrive(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        attrs = super(TDrive, TDrive)._get_metadata()

        return read_csv(
            filenames=data_original_folder() / "t-drive/*.txt",
            ts_id="tid",
            time_id="timestamp",
            signal_id="variable",
            value_id="value",
            dims={
                "ts_id": [],
                "signal_id": [],
                "time_id": [],
            },
            reader_fun=_read_t_drive,
            attrs=attrs,
            verbose=verbose,
        )

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose) -> DataArray:
        return data


def _read_t_drive(filenames: list):
    filename = filenames[0]
    dfs = []

    for file in tqdm(glob(filename), desc="tid", leave=False):
        df = pd.read_csv(file, names=["tid", "timestamp", "longitude", "latitude"])
        if len(df) == 0:
            continue
        dfs.append(df)

    dfs = pd.concat(dfs, ignore_index=True)
    dfs["timestamp"] = dfs.timestamp.apply(
        lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timestamp()
    )

    melted_df = dfs.melt(
        id_vars=["tid", "timestamp"], value_vars=["longitude", "latitude"]
    )

    for row in melted_df.to_dict(orient="records"):
        yield row


if __name__ == "__main__":
    TDrive.save_intermediate_version(False)
    TDrive.save_final_version(True)

    print(TDrive.read_original_version(False))
    print(TDrive.load_intermediate_version())
    print(TDrive.load_final_version())
