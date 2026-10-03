from importlib.metadata import PackageNotFoundError, version

from pyrregular.data_utils import list_paper_datasets
from pyrregular.data_utils import list_registry_datasets as list_datasets
from pyrregular.repository import (
    load_dataset_from_huggingface_via_xarray as load_dataset,
)

try:
    __version__ = version("pyrregular")  # set by setuptools-scm at build/install
except PackageNotFoundError:  # source tree that was never installed
    __version__ = "unknown"
