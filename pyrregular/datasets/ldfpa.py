from pathlib import Path

import pandas as pd
from xarray import DataArray

from pyrregular.data_utils import data_final_folder, data_original_folder
from pyrregular.io_utils import (
    load_from_file,
    load_yaml,
    read_csv,
    save_to_file,
)
from pyrregular.reader_interface import ReaderInterface


class Ldfpa(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_ldfpa(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        # the test set is a different user
        split = [
            "test" if "E" in ts_id else "train" for ts_id in data["ts_id"].to_numpy()
        ]

        data = data.assign_coords(
            split_default=("ts_id", split),
        )
        return data


def _dataset_ldfpa(filenames: list):
    filenames = filenames[0]

    # Read the CSV file into a Pandas DataFrame
    df = pd.read_csv(filenames, header=None)

    # Rename the columns
    column_names = [
        "Sequence_Name",
        "Tag_ID",
        "Timestamp",
        "Date",
        "x_coordinate",
        "y_coordinate",
        "z_coordinate",
        "Activity",
    ]
    df.columns = column_names

    # Convert 'Date' to datetime format
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y %H:%M:%S:%f").apply(
        lambda x: x.timestamp()
    )

    # Convert 'Sequence_Name' to categorical data type
    df["Sequence_Name"] = pd.Categorical(
        df["Sequence_Name"],
        categories=[
            "A01",
            "A02",
            "A03",
            "A04",
            "A05",
            "B01",
            "B02",
            "B03",
            "B04",
            "B05",
            "C01",
            "C02",
            "C03",
            "C04",
            "C05",
            "D01",
            "D02",
            "D03",
            "D04",
            "D05",
            "E01",
            "E02",
            "E03",
            "E04",
            "E05",
        ],
    )

    # Map Tag_ID to human-readable label
    tag_id_map = {
        "010-000-024-033": "ANKLE_LEFT",
        "010-000-030-096": "ANKLE_RIGHT",
        "020-000-033-111": "CHEST",
        "020-000-032-221": "BELT",
    }
    df["Tag_ID"] = df["Tag_ID"].map(tag_id_map)

    # Convert 'Activity' to categorical data type
    activity_categories = [
        "walking",
        "falling",
        "lying down",
        "lying",
        "sitting down",
        "sitting",
        "standing up from lying",
        "on all fours",
        "sitting on the ground",
        "standing up from sitting",
        "standing up from sitting on the ground",
    ]
    df["Activity_Name"] = pd.Categorical(df["Activity"], categories=activity_categories)

    # keep also the activity as a number
    df["Activity"] = df["Activity_Name"].cat.codes

    # Pivot the DataFrame so that Tag_ID becomes a column
    # This will keep the 'Sequence_Name' and 'Activity' as part of the index
    pivoted_df_full_date = df.pivot(
        index=["Sequence_Name", "Activity", "Activity_Name", "Date"],
        columns="Tag_ID",
        values=["x_coordinate", "y_coordinate", "z_coordinate"],
    )

    # Flatten the multi-level column index
    pivoted_df_full_date.columns = [
        f"{coordinate}_{tag}" for coordinate, tag in pivoted_df_full_date.columns
    ]

    # Reset the DataFrame index so that 'Sequence_Name' and 'Activity' become regular columns
    pivoted_df_full_date.reset_index(inplace=True)

    pivoted_df_full_date["tid"] = (
        pivoted_df_full_date["Sequence_Name"].astype(str)
        + "_"
        + pivoted_df_full_date["Activity"].astype(str)
    )

    pivoted_df_full_date = pivoted_df_full_date.melt(
        id_vars=["tid", "Date", "Sequence_Name", "Activity", "Activity_Name"]
    ).dropna()

    for row in pivoted_df_full_date.to_dict(orient="records"):
        yield row


def read_ldfpa(verbose=False):
    attrs = load_yaml(
        str(data_original_folder() / "localization_data_for_person_activity/attrs.yml")
    )
    return read_csv(
        filenames=data_original_folder()
        / "localization_data_for_person_activity/ConfLongDemo_JSI.txt",
        ts_id="tid",
        time_id="Date",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": ["Sequence_Name", "Activity", "Activity_Name"],
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_dataset_ldfpa,
        attrs=attrs,
        verbose=verbose,
    )


if __name__ == "__main__":
    # Ldfpa.save_fixed()
    df = Ldfpa.load_final_version()
