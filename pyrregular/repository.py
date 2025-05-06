import os

import pooch
import xarray as xr

from pyrregular.data_utils import get_project_root
from pyrregular.io_utils import load_from_file

REPOSITORY = pooch.create(
    path=pooch.os_cache("pyrregular"),
    base_url="https://huggingface.co/datasets/splandi/pyrregular/resolve/main/data_final/",
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
            **dict(headers={"Authorization": f"Bearer {api_token}"})
        )
    else:
        downloader = pooch.HTTPDownloader(progressbar=progressbar)
    return REPOSITORY.fetch(name, downloader=downloader)


def load_dataset_from_file(name, api_token=None):
    if ".h5" not in name:
        name += ".h5"
    file = download_dataset_from_huggingface(name, api_token)
    return file


def load_dataset_from_huggingface(name, api_token=None):
    return load_from_file(load_dataset_from_file(name, api_token))


def load_dataset_from_huggingface_via_xarray(name, api_token=None):
    return xr.load_dataset(load_dataset_from_file(name, api_token), engine="pyrregular")


if __name__ == "__main__":
    pass
    # df = load_dataset_from_huggingface("Garment.h5")
