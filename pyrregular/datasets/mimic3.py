import pathlib

import pandas as pd

from pyrregular.data_utils import data_original_folder
from pyrregular.io_utils import (
    load_yaml,
    read_csv,
)
from pyrregular.reader_interface import ReaderInterface

FOLDER_NAME = "mimic-iii/in-hospital-mortality"
STATIC_COLUMNS = [
    "stay",
    "y_true",
    "SUBJECT_ID",
    "EPISODE",
    # "SUBJECT_ID_EPISODE",
    "split",
    "Icustay",
    "Ethnicity",
    "Gender",
    "Age",
    "Length of Stay",
    "Mortality",
    "Diagnosis 4019",
    "Diagnosis 4280",
    "Diagnosis 41401",
    "Diagnosis 42731",
    "Diagnosis 25000",
    "Diagnosis 5849",
    "Diagnosis 2724",
    "Diagnosis 51881",
    "Diagnosis 53081",
    "Diagnosis 5990",
    "Diagnosis 2720",
    "Diagnosis 2859",
    "Diagnosis 2449",
    "Diagnosis 486",
    "Diagnosis 2762",
    "Diagnosis 2851",
    "Diagnosis 496",
    "Diagnosis V5861",
    "Diagnosis 99592",
    "Diagnosis 311",
    "Diagnosis 0389",
    "Diagnosis 5859",
    "Diagnosis 5070",
    "Diagnosis 40390",
    "Diagnosis 3051",
    "Diagnosis 412",
    "Diagnosis V4581",
    "Diagnosis 2761",
    "Diagnosis 41071",
    "Diagnosis 2875",
    "Diagnosis 4240",
    "Diagnosis V1582",
    "Diagnosis V4582",
    "Diagnosis V5867",
    "Diagnosis 4241",
    "Diagnosis 40391",
    "Diagnosis 78552",
    "Diagnosis 5119",
    "Diagnosis 42789",
    "Diagnosis 32723",
    "Diagnosis 49390",
    "Diagnosis 9971",
    "Diagnosis 2767",
    "Diagnosis 2760",
    "Diagnosis 2749",
    "Diagnosis 4168",
    "Diagnosis 5180",
    "Diagnosis 45829",
    "Diagnosis 4589",
    "Diagnosis 73300",
    "Diagnosis 5845",
    "Diagnosis 78039",
    "Diagnosis 5856",
    "Diagnosis 4271",
    "Diagnosis 4254",
    "Diagnosis 4111",
    "Diagnosis V1251",
    "Diagnosis 30000",
    "Diagnosis 3572",
    "Diagnosis 60000",
    "Diagnosis 27800",
    "Diagnosis 41400",
    "Diagnosis 2768",
    "Diagnosis 4439",
    "Diagnosis 27651",
    "Diagnosis V4501",
    "Diagnosis 27652",
    "Diagnosis 99811",
    "Diagnosis 431",
    "Diagnosis 28521",
    "Diagnosis 2930",
    "Diagnosis 7907",
    "Diagnosis E8798",
    "Diagnosis 5789",
    "Diagnosis 79902",
    "Diagnosis V4986",
    "Diagnosis V103",
    "Diagnosis 42832",
    "Diagnosis E8788",
    "Diagnosis 00845",
    "Diagnosis 5715",
    "Diagnosis 99591",
    "Diagnosis 07054",
    "Diagnosis 42833",
    "Diagnosis 4275",
    "Diagnosis 49121",
    "Diagnosis V1046",
    "Diagnosis 2948",
    "Diagnosis 70703",
    "Diagnosis 2809",
    "Diagnosis 5712",
    "Diagnosis 27801",
    "Diagnosis 42732",
    "Diagnosis 99812",
    "Diagnosis 4139",
    "Diagnosis 3004",
    "Diagnosis 2639",
    "Diagnosis 42822",
    "Diagnosis 25060",
    "Diagnosis V1254",
    "Diagnosis 42823",
    "Diagnosis 28529",
    "Diagnosis E8782",
    "Diagnosis 30500",
    "Diagnosis 78791",
    "Diagnosis 78551",
    "Diagnosis E8889",
    "Diagnosis 78820",
    "Diagnosis 34590",
    "Diagnosis 2800",
    "Diagnosis 99859",
    "Diagnosis V667",
    "Diagnosis E8497",
    "Diagnosis 79092",
    "Diagnosis 5723",
    "Diagnosis 3485",
    "Diagnosis 5601",
    "Diagnosis 25040",
    "Diagnosis 570",
    "Diagnosis 71590",
    "Diagnosis 2869",
    "Diagnosis 2763",
    "Diagnosis 5770",
    "Diagnosis V5865",
    "Diagnosis 99662",
    "Diagnosis 28860",
    "Diagnosis 36201",
    "Diagnosis 56210",
    "Age_fix",
]
GLASCOW_DICT_MOTOR = {
    "Localizes Pain": 5,
    "Obeys Commands": 6,
    "Flex-withdraws": 4,
    "Abnormal Flexion": 3,
    "Abnormal extension": 2,
    "No response": 1,
}
GLASCOW_DICT_EYE = {
    "Spontaneously": 4,
    "To Speech": 3,
    "To Pain": 2,
}
GLASCOW_DICT_VERBAL = {
    "Oriented": 5,
    "Confused": 4,
    "Inappropriate Words": 3,
    "Incomprehensible sounds": 2,
    "No Response-ETT": 1,
    "1.0 ET/Trach": 1,
    "No Response": 1,
}


def _dataset_mimic3(filenames: dict):
    files = filenames["ts_files"]

    # Read labels
    labels_train = (
        pd.read_csv(filenames["labels_train"])
        .sort_values("stay")
        .reset_index(drop=True)
    )
    labels_train["SUBJECT_ID"] = labels_train["stay"].apply(
        lambda x: int(x.split("_")[0])
    )
    labels_train["EPISODE"] = labels_train["stay"].apply(lambda x: x.split("_")[1])
    labels_train["SUBJECT_ID_EPISODE"] = (
        labels_train["SUBJECT_ID"].astype(str) + "_" + labels_train["EPISODE"]
    )
    labels_train["split"] = "train"

    labels_test = (
        pd.read_csv(filenames["labels_test"]).sort_values("stay").reset_index(drop=True)
    )
    labels_test["SUBJECT_ID"] = labels_test["stay"].apply(
        lambda x: int(x.split("_")[0])
    )
    labels_test["EPISODE"] = labels_test["stay"].apply(lambda x: x.split("_")[1])
    labels_test["SUBJECT_ID_EPISODE"] = (
        labels_test["SUBJECT_ID"].astype(str) + "_" + labels_test["EPISODE"]
    )
    labels_test["split"] = "test"

    labels = pd.concat([labels_train, labels_test], ignore_index=True)

    # Read other metadata files
    file_path_other_train = filenames["other_train"]
    df_episodes_train = pd.DataFrame()
    for folder in file_path_other_train.iterdir():
        if folder.is_dir():  # Check if it's a directory
            # Find matching files
            matching_files = [
                file
                for file in folder.iterdir()
                if file.is_file()
                and "episode" in file.name.lower()
                and "timeseries" not in file.name.lower()
                and file.suffix == ".csv"
            ]
            for file in matching_files:
                df = pd.read_csv(file)
                df["SUBJECT_ID"] = file.parent.stem
                df["EPISODE"] = file.stem
                df["SUBJECT_ID_EPISODE"] = (
                    df["SUBJECT_ID"].astype(str) + "_" + df["EPISODE"].astype(str)
                )
                df_episodes_train = pd.concat(
                    [df_episodes_train, df], ignore_index=True
                )
    df_episodes_train.drop(columns=["SUBJECT_ID", "EPISODE"], inplace=True)

    file_path_other_test = filenames["other_test"]
    df_episodes_test = pd.DataFrame()
    for folder in file_path_other_test.iterdir():
        if folder.is_dir():  # Check if it's a directory
            # Find matching files
            matching_files = [
                file
                for file in folder.iterdir()
                if file.is_file()
                and "episode" in file.name.lower()
                and "timeseries" not in file.name.lower()
                and file.suffix == ".csv"
            ]
            for file in matching_files:
                df = pd.read_csv(file)
                df["SUBJECT_ID"] = file.parent.stem
                df["EPISODE"] = file.stem
                df["SUBJECT_ID_EPISODE"] = (
                    df["SUBJECT_ID"].astype(str) + "_" + df["EPISODE"].astype(str)
                )
                df_episodes_test = pd.concat([df_episodes_test, df], ignore_index=True)
    df_episodes_test.drop(columns=["SUBJECT_ID", "EPISODE"], inplace=True)

    df_episodes = pd.concat([df_episodes_train, df_episodes_test], ignore_index=True)

    # Merge labels and metadata
    all_labels = labels.merge(df_episodes, on="SUBJECT_ID_EPISODE", how="left")
    all_labels.drop(["Weight", "Height"], axis=1, inplace=True)
    all_labels["Age_fix"] = all_labels["Age"].apply(
        lambda x: x if x < 300 else x - 211
    )  # Fix age

    # Read time series files
    for i in range(len(files)):
        ts = pd.read_csv(files[i])
        ts["SUBJECT_ID_EPISODE"] = files[i].stem.replace("_timeseries", "")
        ts = ts.merge(all_labels, on="SUBJECT_ID_EPISODE", how="left")

        # Process Glascow coma scale columns
        ts["Glascow coma scale motor response"] = ts[
            "Glascow coma scale motor response"
        ].apply(
            lambda x: (
                float(x.split(" ")[0])
                if isinstance(x, str) and x.split(" ")[0].isdigit()
                else GLASCOW_DICT_MOTOR.get(x, x)
            )
        )
        ts["Glascow coma scale eye opening"] = ts[
            "Glascow coma scale eye opening"
        ].apply(
            lambda x: (
                float(x.split(" ")[0])
                if isinstance(x, str) and x.split(" ")[0].isdigit()
                else GLASCOW_DICT_EYE.get(x, x)
            )
        )
        ts["Glascow coma scale verbal response"] = ts[
            "Glascow coma scale verbal response"
        ].apply(
            lambda x: (
                float(x.split(" ")[0])
                if isinstance(x, str) and x.split(" ")[0].isdigit()
                else GLASCOW_DICT_VERBAL.get(x, x)
            )
        )

        ts_melted = ts.melt(
            id_vars=list(STATIC_COLUMNS) + ["Hours", "SUBJECT_ID_EPISODE"]
        )
        ts_melted = ts_melted[ts_melted["value"].notna()].reset_index(drop=True)

        for j in range(len(ts_melted)):
            yield ts_melted.iloc[j : j + 1].to_dict(orient="records")[0]


def read_mimic3(verbose=False):
    return read_csv(
        filenames={
            "ts_files": (
                sorted(
                    [
                        file
                        for file in (
                            data_original_folder() / FOLDER_NAME / "train"
                        ).glob("*.csv")
                    ]
                )[:-1]
                + sorted(
                    [
                        file
                        for file in (
                            data_original_folder() / FOLDER_NAME / "test"
                        ).glob("*.csv")
                    ]
                )[:-1]
            ),
            "labels_train": data_original_folder()
            / FOLDER_NAME
            / "train"
            / "listfile.csv",
            "labels_test": data_original_folder()
            / FOLDER_NAME
            / "test"
            / "listfile.csv",
            "other_train": data_original_folder() / "mimic-iii/root" / "train",
            "other_test": data_original_folder() / "mimic-iii/root" / "test",
        },
        ts_id="SUBJECT_ID_EPISODE",
        time_id="Hours",
        signal_id="variable",
        value_id="value",
        dims={
            "ts_id": STATIC_COLUMNS,
            "signal_id": [],
            "time_id": [],
        },
        reader_fun=_dataset_mimic3,
        verbose=verbose,
        attrs=dict(),
        time_index_as_datetime=False,
    )


class Mimic3(ReaderInterface):
    @staticmethod
    def read_original_version(verbose=False):
        return read_mimic3(verbose=verbose)


if __name__ == "__main__":
    # Mimic3.save_unfixed(True)
    # Mimic3.save_fixed(True)
    df = Mimic3.load_final_version()
