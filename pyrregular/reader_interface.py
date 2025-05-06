import os.path
from abc import ABC, abstractmethod
from pathlib import Path

from xarray import DataArray

from pyrregular.data_utils import (
    data_final_folder,
    data_intermediate_folder,
    metadata_folder,
)
from pyrregular.io_utils import (
    get_current_aoe_time,
    load_from_file,
    load_yaml,
    save_to_file,
)


class ReaderInterface(ABC):
    """
    Abstract base class defining the interface for reading, fixing, and
    saving datasets as xarray DataArray objects in a consistent pipeline.

    Subclasses must implement:
      - read_original_version(verbose: bool) -> DataArray
      - _fix_intermediate_version(data: DataArray, verbose: bool) -> DataArray

    Public methods handle:
      - writing the “intermediate” and “final” HDF5 files
      - enriching metadata (timestamps, YAML metadata)
      - loading back intermediate and final versions
    """

    fast_to_test = False

    @staticmethod
    @abstractmethod
    def read_original_version(verbose) -> DataArray:
        pass

    @classmethod
    def save_final_version(cls, verbose=True):
        data = cls.load_intermediate_version()
        data = cls.fix_intermediate_version(data, verbose)
        save_to_file(
            data_array=data,
            filename=data_final_folder()
            / (str(cls.__get_name_from_classname()) + ".h5"),
        )

    @classmethod
    def save_intermediate_version(cls, verbose=True):
        save_to_file(
            data_array=cls.read_original_version(verbose=verbose),
            filename=data_intermediate_folder()
            / (str(cls.__get_name_from_classname()) + ".h5"),
        )

    @staticmethod
    @abstractmethod
    def _fix_intermediate_version(data: DataArray, verbose=True) -> DataArray:
        return data

    @classmethod
    def fix_intermediate_version(cls, data: DataArray, verbose=True) -> DataArray:
        data = cls._fix_intermediate_version(data, verbose)
        metadata = cls._get_metadata()
        data.attrs = dict()  # shouldn't be necessary, but just to be sure
        data.attrs["_is_fixed"] = True
        data.attrs["_fixed_at"] = get_current_aoe_time()
        data.attrs.update(metadata)
        return data

    @classmethod
    def load_intermediate_version(cls) -> DataArray:
        data = load_from_file(
            f"{data_intermediate_folder() / cls.__get_name_from_classname()}.h5"
        )
        return data

    @classmethod
    def load_final_version(cls) -> DataArray:
        data = load_from_file(
            f"{data_final_folder() / cls.__get_name_from_classname()}.h5"
        )
        return data

    @classmethod
    def _get_metadata(cls):
        metadata_file = f"{metadata_folder() / cls.__get_name_from_classname()}.yml"

        return load_yaml(metadata_file) if os.path.exists(metadata_file) else None

    @classmethod
    def __get_name_from_classname(cls):
        return Path(cls.__name__)
