import numpy as np
import pytest
import sparse

from tests.utils import TEST_CASES


@pytest.mark.parametrize(
    "dataset",
    TEST_CASES,
)
def test_does_not_have_nan_instances(dataset):
    nan_idxs = np.where(
        sparse.all(
            sparse.isnan(dataset.load_final_version().data), axis=(1, 2)
        ).todense()
    )[0]
    assert (
        len(nan_idxs) == 0
    ), f"Dataset {dataset.__name__} has {len(nan_idxs)} nan instances: {nan_idxs}"


@pytest.mark.parametrize(
    "dataset",
    TEST_CASES,
)
def test_metadata_has_configs(dataset):
    df = dataset.load_final_version()
    assert "configs" in df.attrs, f"Dataset {dataset.__name__} does not have configs"


@pytest.mark.parametrize(
    "dataset",
    TEST_CASES,
)
def test_metadata_task_columns_exist_in_xarray(dataset):
    df = dataset.load_final_version()
    missing_cols = list()
    for task_name in list(df.attrs["configs"].keys()):
        target = df.attrs["configs"][task_name]["target"]
        split = df.attrs["configs"][task_name]["split"]
        if target not in df.coords:
            missing_cols.append((task_name, target))
        if split not in df.coords:
            missing_cols.append((task_name, split))
    assert (
        len(missing_cols) == 0
    ), f"Dataset {dataset.__name__} is missing the following columns: {missing_cols}"


@pytest.mark.parametrize(
    "dataset",
    TEST_CASES,
)
def test_classes_are_label_encoded(dataset):
    df = dataset.load_final_version()
    incorrect_cols = list()
    for task_name in list(df.attrs["configs"].keys()):
        target = df.attrs["configs"][task_name]["target"]
        task = df.attrs["configs"][task_name]["task"]
        if task == "classification":
            y = df[target].data
            if not np.all(np.unique(y) == np.arange(len(np.unique(y)))):
                incorrect_cols.append((task_name, target))

    assert (
        len(incorrect_cols) == 0
    ), f"Dataset {dataset.__name__} has the following columns that are not label encoded: {incorrect_cols}"


@pytest.mark.parametrize(
    "dataset",
    TEST_CASES,
)
def test_are_ts_id_unique(dataset):
    df = dataset.load_final_version()
    assert np.unique(df["ts_id"].data, return_counts=True)[1].max() == 1


if __name__ == "__main__":
    pytest.main()
