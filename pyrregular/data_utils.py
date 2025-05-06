import pathlib

import numpy as np
import pandas as pd

HUGGINGFACE_REPO_NAME = "pyrregular"
DATA_ORIGINAL_FOLDER = "data" + "/" + HUGGINGFACE_REPO_NAME + "/" + "data_original"
DATA_INTERMEDIATE_FOLDER = (
    "data" + "/" + HUGGINGFACE_REPO_NAME + "/" + "data_intermediate"
)
DATA_FINAL_FOLDER = "data" + "/" + HUGGINGFACE_REPO_NAME + "/" + "data_final"
METADATA_FOLDER = "metadata"


def get_project_root() -> pathlib.Path:
    return pathlib.Path(__file__).parent


def data_original_folder():
    return get_project_root() / ".." / DATA_ORIGINAL_FOLDER


def data_intermediate_folder():
    return get_project_root() / ".." / DATA_INTERMEDIATE_FOLDER


def data_final_folder():
    return get_project_root() / ".." / DATA_FINAL_FOLDER


def metadata_folder():
    return get_project_root() / ".." / METADATA_FOLDER


def list_final_datasets():
    return sorted(list(data_final_folder().glob("*.h5")))


def list_metadata_files():
    return sorted(list(metadata_folder().glob("*.yml")))


def list_registry_datasets():
    return sorted(list(pd.read_csv(get_project_root() / "registry.txt", sep=" ", index_col=0).index))


def infer_static_columns(df, id_column, dropna=False):
    df_grouped = df.groupby(id_column).nunique(dropna=dropna)
    return [
        c
        for c in df_grouped.columns
        if np.array_equal(df_grouped[c].to_numpy(), np.ones(len(df_grouped[c])))
        and c != id_column
    ]


if __name__ == "__main__":
    out = list_registry_datasets()
