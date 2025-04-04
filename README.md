# pyrregular

# Installation

You can install via pip with:

```bash
pip install pyrregular
```

# Quick Guide
## Load a dataset

```python
from pyrregular import load_dataset

df = load_dataset("Garment.h5")
```

The dataset is saved in the default os cache directory, which can be found with:

```python
import pooch

print(pooch.os_cache("pyrregular"))
```


## Contributing (work in progress)
### The "Long Format"
The basic format to convert any dataset to our representation is the long format.
The long format is simply a tuple:

```(time_series_id, channel_id, timestamp, value, static_var_1, static_var_2, ...)```.

If your dataset contains rows that are in this format, you are almost good to go. Else, there will be a little bit of preprocessing to do.

#### Case 1. (easy) Your dataset is already in the long format

Let's assume for now your dataset is already in this form, and you do not have static variables. Here is a minimal working example.

```python
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class YourDataset(ReaderInterface):
    @staticmethod
    def read(verbose=False):
        return read_your_dataset(verbose=verbose)

def read_your_dataset(verbose=False):
    return read_csv(
        filenames="your_original_dataset.csv",
        ts_id="name_of_your_time_series_id_column",
        time_id="name_of_your_timestamp_column",
        signal_id="name_of_your_channel_id_column",
        value_id="name_of_your_value_column",
        verbose=verbose,
    )
```

This gets a little bit more complicated if you also have static variables. First, you need to identify the dimension static variables depend on. 
The most common case for a static variable is to depend on time `ts_id` (e.g., a patient's age).



```python
from pyrregular.io_utils import read_csv
from pyrregular.reader_interface import ReaderInterface


class YourDataset(ReaderInterface):
    @staticmethod
    def read(verbose=False):
        return read_your_dataset(verbose=verbose)

def read_your_dataset(verbose=False):
    return read_csv(
        filenames="your_original_dataset.csv",
        ts_id="name_of_your_time_series_id_column",
        time_id="name_of_your_timestamp_column",
        signal_id="name_of_your_channel_id_column",
        value_id="name_of_your_value_column",
        dims={
            "ts_id": ["name_of_your_static_variable_1_column", "name_of_your_static_variable_2_column"],
            "signal_id": [],
            "time_id": [],
        },
        verbose=verbose,
    )
```
