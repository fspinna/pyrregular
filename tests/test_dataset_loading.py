import pytest

from pyrregular import load_dataset
from tests.constants import TEST_CASES_FAST as TEST_CASES


@pytest.mark.parametrize("dataset", TEST_CASES)
def test_are_datasets_loading(dataset):
    try:
        df = load_dataset(dataset)
    except Exception as e:
        pytest.fail(f"Failed to load dataset {dataset}: {e}")
