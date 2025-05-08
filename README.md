# pyrregular


![Logo](https://github.com/fspinna/pyrregular/blob/main/assets/images/logo_01.png?raw=true)


|               | **[📖 Documentation](https://fspinna.github.io/pyrregular/)** · **[⚙️ Tutorials](https://github.com/fspinna/pyrregular/blob/main/docs/notebooks)**                                                                                                                                                                                               |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**     | [![build](https://github.com/fspinna/pyrregular/actions/workflows/build.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/build.yml) [![docs](https://github.com/fspinna/pyrregular/actions/workflows/sphinx.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/sphinx.yml) [![pypi publish](https://github.com/fspinna/pyrregular/actions/workflows/python-publish.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/python-publish.yml) 
| **Code**      | [![PyPI version](https://img.shields.io/pypi/v/pyrregular.svg)](https://pypi.org/project/pyrregular/) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrregular) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)                                                   |
| **Community** | [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/fspinna/pyrregular/issues)                                                                                                                                                                                   |
| **Citation**  | coming soon...                                                                                                                                                                                                                                                                                                                            |


# Installation

You can install via pip with:

```bash
pip install pyrregular
```

For third party models use:

```bash
pip install pyrregular[models]
```


# Quick Guide
## List datasets
If you want to see all the datasets available, you can use the `list_datasets` function:

```python
from pyrregular import list_datasets

df = list_datasets()
```


## Load a dataset
To load a dataset, you can use the `load_dataset` function. For example, to load the "Garment" dataset, you can do:

```python
from pyrregular import load_dataset

df = load_dataset("Garment.h5")
```

The dataset is saved in the default os cache directory, which can be found with:

```python
import pooch

print(pooch.os_cache("pyrregular"))
```

The repository is hosted at: https://huggingface.co/datasets/splandi/pyrregular/

## Downstream tasks
### Classification
To use the dataset for classification, you can just "densify" it:

```python
from pyrregular import load_dataset

df = load_dataset("Garment.h5")
X, _ = df.irr.to_dense()
y, split = df.irr.get_task_target_and_split()

X_train, X_test = X[split != "test"], X[split == "test"]
y_train, y_test = y[split != "test"], y[split == "test"]

# We have ready-to-go models from various libraries:
from pyrregular.models.rocket import rocket_pipeline

model = rocket_pipeline
model.fit(X_train, y_train)
model.score(X_test, y_test)
```


