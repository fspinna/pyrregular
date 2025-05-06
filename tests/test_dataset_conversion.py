import pytest

from tests.utils import IDS_FAST, TEST_CASES_FAST

# @pytest.mark.parametrize(
#     "read_function",
#     TEST_CASES,
#     ids=IDS,
# )
# def test_read_csv_for_exceptions(read_function):
#     try:
#         read_function.read()
#     except Exception as e:
#         pytest.fail(
#             f"{read_function.__name__} raised an exception unexpectedly with input '{e}"
#         )


@pytest.mark.parametrize(
    "read_function",
    TEST_CASES_FAST,
    ids=IDS_FAST,
)
def test_quick_read_csv_for_exceptions(read_function):
    try:
        read_function.read()
    except Exception as e:
        pytest.fail(
            f"{read_function.__name__} raised an exception unexpectedly with input '{e}"
        )


if __name__ == "__main__":
    pytest.main()
