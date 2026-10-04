import os
from typing import Optional

import pooch
import xarray as xr

from pyrregular.data_utils import get_project_root
from pyrregular.io_utils import load_from_file

# Each data version is a fixed commit of the Hugging Face repo, tagged data-vN,
# so that a pyrregular release always downloads the files it was made for.
DATA_VERSION = "data-v1"
DATA_REVISION = "f416b1b432f248a73323222f2378d0e8433ca6d2"  # commit of data-v1


def _cache_path():
    # data-v1 keeps the original cache folder; later versions get their own
    # subfolder, so that different versions never overwrite each other
    path = pooch.os_cache("pyrregular")
    return path if DATA_VERSION == "data-v1" else path / DATA_VERSION


REPOSITORY = pooch.create(
    path=_cache_path(),
    base_url=(
        "https://huggingface.co/datasets/splandi/pyrregular/resolve/"
        f"{DATA_REVISION}/data_final/"
    ),
    registry=None,
)

REPOSITORY.load_registry(get_project_root() / "registry.txt")


def download_dataset_from_huggingface(
    name, use_api_token=False, api_token=None, progressbar=True
):
    if use_api_token:
        if api_token is None:
            api_token = os.getenv("HF_TOKEN")
        if api_token is None:
            raise ValueError("You need to provide an API token to download the dataset")
        downloader = pooch.HTTPDownloader(
            progressbar=progressbar,
            headers={"Authorization": f"Bearer {api_token}"},
        )
    else:
        downloader = pooch.HTTPDownloader(progressbar=progressbar)
    return REPOSITORY.fetch(name, downloader=downloader)


def load_dataset_from_file(name, api_token=None):
    if ".h5" not in name:
        name += ".h5"
    file = download_dataset_from_huggingface(
        name, use_api_token=api_token is not None, api_token=api_token
    )
    return file


def load_dataset_from_huggingface(name, api_token=None):
    return load_from_file(load_dataset_from_file(name, api_token))


def load_dataset_from_huggingface_via_xarray(
    name: str, api_token: Optional[str] = None
) -> xr.Dataset:
    """Load a dataset from the online repository.

    Args:
        name (str): The name of the dataset to load.
        api_token (Optional[str]): The API token to use for authentication.

    Returns:
        xr.Dataset: A dataset loaded from Hugging Face.
    """
    return xr.load_dataset(load_dataset_from_file(name, api_token), engine="pyrregular")
