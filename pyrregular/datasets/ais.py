from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xarray import DataArray

from pyrregular.data_utils import data_final_folder, data_original_folder
from pyrregular.io_utils import (
    load_from_file,
    load_yaml,
    read_csv,
    save_to_file,
)
from pyrregular.reader_interface import ReaderInterface


class Ais(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_ais(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        # we use the length of the vessel as the class label (3 sizes)
        labels = np.digitize(data["length"].to_numpy(), bins=[80, 100])
        train_indices, test_indices = train_test_split(
            np.arange(len(data["length"])),
            test_size=0.3,
            stratify=labels,
            random_state=42,
        )
        split = np.array(["train"] * len(data["length"]))
        split[test_indices] = "test"

        data = data.assign_coords(
            split_default=("ts_id", split), class_default=("ts_id", labels)
        )
        return data


def _dataset_ais(filenames: list):
    # Read the CSV file into a Pandas DataFrame
    for filename in filenames:
        df = pd.read_parquet(filename)
        df["date_time_utc"] = pd.to_datetime(
            df["date_time_utc"], format="%Y-%m-%d %H:%M:%S"
        ).apply(
            lambda x: int(x.timestamp())
        )  # FIXME
        for i in range(len(df)):
            row = (
                df.iloc[i : i + 1]
                .melt(id_vars=["mmsi", "date_time_utc", "imo_nr", "length"])
                .to_dict(orient="records")
            )
            for new_row in row:
                yield new_row


def read_ais(verbose=False):
    attrs = load_yaml(
        str(
            data_original_folder()
            / "terrestrial_vessel_automatic_identification_system/attrs.yml"
        )
    )
    return read_csv(
        filenames=[
            data_original_folder()
            / "terrestrial_vessel_automatic_identification_system"
            / f"2020-2_group_{i}"
            for i in range(40)
        ],
        ts_id="mmsi",
        time_id="date_time_utc",
        signal_id="variable",
        value_id="value",
        dims={"ts_id": ["imo_nr", "length"], "signal_id": [], "time_id": []},
        reader_fun=_dataset_ais,
        attrs=attrs,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Ais.save_fixed()
    df = Ais.load_final_version()
