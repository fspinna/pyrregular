import pytest

from pyrregular import load_dataset
from tests.constants import TEST_CASES_FAST as TEST_CASES

pytestmark = pytest.mark.network


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


def test_registry_matches_files_at_data_revision():
    # registry.txt and DATA_REVISION must change together: the hashes in the
    # registry must be those of the files at the pinned Hugging Face commit
    import requests

    from pyrregular import repository

    url = (
        "https://huggingface.co/api/datasets/splandi/pyrregular/tree/"
        f"{repository.DATA_REVISION}/data_final"
    )
    tree = requests.get(url, timeout=60).json()
    remote = {
        entry["path"].split("/")[-1]: entry["lfs"]["oid"]
        for entry in tree
        if entry["type"] == "file" and "lfs" in entry
    }
    assert remote == dict(repository.REPOSITORY.registry)
