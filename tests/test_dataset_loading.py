import pytest

from pyrregular import load_dataset
from tests.constants import TEST_CASES_FAST as TEST_CASES


@pytest.fixture(params=TEST_CASES)
def loaded_dataset(request):
    dataset = request.param
    try:
        df = load_dataset(dataset)["data"]
    except Exception as e:
        pytest.fail(f"Failed to load dataset {dataset}: {e}")
    return dataset, df


def test_dataset_dense_conversion(loaded_dataset):
    dataset, df = loaded_dataset
    try:
        X, _ = df.irr.to_dense()
    except Exception as e:
        pytest.fail(f"Failed to convert dataset {dataset} to dense: {e}")
