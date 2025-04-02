from xarray.backends import BackendEntrypoint
from pyrregular.io_utils import load_from_file
from xarray.core.indexing import NdArrayLikeIndexingAdapter
from pyrregular.accessor import IrregularAccessor  # this is needed to register the accessor


class IrregularEntrypoint(BackendEntrypoint):
    def open_dataset(
        self,
        filename_or_obj,
        *,
        drop_variables=None,
    ):
        da = load_from_file(filename_or_obj)
        da.data = NdArrayLikeIndexingAdapter(da.data)
        da.name = "data"
        return da.to_dataset()

