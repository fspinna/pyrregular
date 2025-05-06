import numpy as np
import pandas as pd
from xarray import DataArray

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import (
    load_yaml,
    read_csv,
)
from pyrregular.reader_interface import ReaderInterface


class Physionet2012(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_physionet2012(verbose=verbose)

    @staticmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        mapping = {"a": "train", "b": "test"}
        split = data["set"].to_numpy()
        split = [mapping[split[i]] for i in range(len(split))]
        data = data.assign_coords(split_default=("ts_id", split))
        return data


def time_str_to_seconds(time_str):
    hours, minutes = map(int, time_str.split(":"))
    return hours * 3600 + minutes * 60


def process_files(file_list, set_name, possible_columns, general_descriptors):
    processed_dfs = pd.DataFrame()
    for file in file_list:
        df = pd.read_csv(file)
        # if df["Time"].isna().sum() > 0:
        #     print(file)
        df["Count"] = df.groupby(["Time", "Parameter"]).cumcount()
        pivot_no_agg = (
            df.pivot(index=["Time", "Count"], columns="Parameter", values="Value")
            .reset_index()
            .drop(columns="Count")
        )
        pivot_no_agg["Weight"] = pivot_no_agg["Weight"].replace(-1, np.nan)
        desired_columns = sorted(
            list(set(list(pivot_no_agg.columns) + possible_columns))
        )
        final_pivoted_df = pivot_no_agg.reindex(columns=desired_columns)
        final_pivoted_df[general_descriptors] = final_pivoted_df[
            general_descriptors
        ].iloc[0]
        final_pivoted_df["set"] = set_name
        processed_dfs = pd.concat([processed_dfs, final_pivoted_df])
    return processed_dfs


def _dataset_physionet2012(filenames: dict):
    possible_columns = [
        "RecordID",
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "Weight",
        "Albumin",
        "ALP",
        "ALT",
        "AST",
        "Bilirubin",
        "BUN",
        "Cholesterol",
        "Creatinine",
        "DiasABP",
        "FiO2",
        "GCS",
        "Glucose",
        "HCO3",
        "HCT",
        "HR",
        "K",
        "Lactate",
        "Mg",
        "MAP",
        "MechVent",
        "Na",
        "NIDiasABP",
        "NIMAP",
        "NISysABP",
        "PaCO2",
        "PaO2",
        "pH",
        "Platelets",
        "RespRate",
        "SaO2",
        "SysABP",
        "Temp",
        "TropI",
        "TropT",
        "Urine",
        "WBC",
    ]
    general_descriptors = ["RecordID", "Age", "Gender", "Height", "ICUType"]

    static_columns = [
        "Age",
        "Gender",
        "Height",
        "ICUType",
        "SAPS-I",
        "SOFA",
        "Length_of_stay",
        "Survival",
        "In-hospital_death",
    ]

    set_a_files = sorted([file for file in filenames["set-a"].glob("*.txt")])
    set_b_files = sorted(
        [file for file in filenames["set-b"].glob("*.txt") if file.name != "143656.txt"]
    )
    # "143656.txt" is an emtpy record, containing only patient information and no measurements

    outcomes = pd.concat(
        [pd.read_csv(filenames["Outcomes-a"]), pd.read_csv(filenames["Outcomes-b"])]
    )

    dfs_a = process_files(set_a_files, "a", possible_columns, general_descriptors)
    dfs_b = process_files(set_b_files, "b", possible_columns, general_descriptors)
    dfs = pd.concat([dfs_a, dfs_b])
    dfs = dfs.join(outcomes.set_index("RecordID"), on="RecordID")
    dfs["Time_ID"] = dfs["Time"].apply(time_str_to_seconds)
    dfs_melted = dfs.melt(
        id_vars=static_columns + ["Time_ID", "Time", "RecordID"] + ["set"]
    )
    dfs_melted = dfs_melted[dfs_melted["value"].notna()].reset_index(drop=True)
    for i in range(len(dfs_melted)):
        yield dfs_melted.iloc[i : i + 1].to_dict(orient="records")[0]


def read_physionet2012(verbose=False):
    attrs = load_yaml(str(data_original_folder() / "physionet2012/attrs.yml"))
    return read_csv(
        filenames={
            "set-a": data_original_folder() / "physionet2012/set-a",
            "set-b": data_original_folder() / "physionet2012/set-b",
            "Outcomes-a": data_original_folder() / "physionet2012/Outcomes-a.txt",
            "Outcomes-b": data_original_folder() / "physionet2012/Outcomes-b.txt",
        },
        ts_id="RecordID",
        time_id="Time_ID",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": [
                "Age",
                "Gender",
                "Height",
                "ICUType",
                "SAPS-I",
                "SOFA",
                "Length_of_stay",
                "Survival",
                "In-hospital_death",
                "set",
            ],
            "signal_id": [],
            "time_id": ["Time"],
        },
        reader_fun=_dataset_physionet2012,
        attrs=attrs,
        verbose=verbose,
        time_index_as_datetime=False,
    )


if __name__ == "__main__":
    # Physionet2012.save_fixed()
    df = Physionet2012.load_final_version()
