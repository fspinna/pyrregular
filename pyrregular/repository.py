import pooch
import os
from pyrregular.io_utils import load_from_file
from pyrregular.data_utils import get_project_root

REPOSITORY = pooch.create(
    path=pooch.os_cache("pyrregular"),
    base_url="https://huggingface.co/datasets/splandi/pyrregular/resolve/main/data_final/",
    registry=None,
)

REPOSITORY.load_registry(get_project_root() / "registry.txt")


def download_dataset_from_huggingface(name, api_token=None):
    if api_token is None:
        api_token = os.getenv("HF_TOKEN")
    if api_token is None:
        raise ValueError("You need to provide an API token to download the dataset")
    downloader = pooch.HTTPDownloader(
        **dict(headers={"Authorization": f"Bearer {api_token}"})
    )
    return REPOSITORY.fetch(name, downloader=downloader)


def load_dataset_from_huggingface(name, api_token=None):
    if ".h5" not in name:
        name += ".h5"
    file = download_dataset_from_huggingface(name, api_token)
    return load_from_file(file)


if __name__ == "__main__":
    df = load_dataset_from_huggingface("Garment.h5")
