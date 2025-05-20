![Logo](https://github.com/fspinna/pyrregular/blob/main/assets/images/logo_01.png?raw=true)


|               | **[📖 Documentation](https://fspinna.github.io/pyrregular/)** · **[⚙️ Tutorials](https://github.com/fspinna/pyrregular/blob/main/docs/notebooks)**                                                                                                                                                                                               |
|---------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **CI/CD**     | [![build](https://github.com/fspinna/pyrregular/actions/workflows/build.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/build.yml) [![docs](https://github.com/fspinna/pyrregular/actions/workflows/sphinx.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/sphinx.yml) [![pypi publish](https://github.com/fspinna/pyrregular/actions/workflows/python-publish.yml/badge.svg)](https://github.com/fspinna/pyrregular/actions/workflows/python-publish.yml) 
| **Code**      | [![PyPI version](https://img.shields.io/pypi/v/pyrregular.svg)](https://pypi.org/project/pyrregular/) ![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyrregular) [![!black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)                                                   |
| **Community** | [![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/fspinna/pyrregular/issues)                                                                                                                                                                                   |
| **Paper**     | [![arXiv](https://img.shields.io/badge/arXiv-2505.06047-b31b1b.svg)](https://arxiv.org/pdf/2505.06047)                                                                                                                                                                                                                                                                                                                            |



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

# Available Datasets

| 📈 Dataset                                   | 📖 Source                                                             |
|----------------------------------------------|---------------------------------------------------------------------|
| Alembics Bowls Flasks                        | Spinnato & Landi, 2025                                              |
| AllGestureWiimoteX                           | Guna et al., 2014                                                   |
| AllGestureWiimoteY                           | Guna et al., 2014                                                   |
| AllGestureWiimoteZ                           | Guna et al., 2014                                                   |
| Animals                                      | Ferrero et al., 2018                                                |
| AsphaltObstaclesCoordinates                  | Souza, 2018                                                         |
| AsphaltPavementTypeCoordinates               | Souza, 2018                                                         |
| AsphaltRegularityCoordinates                 | Souza, 2018                                                         |
| CharacterTrajectories                        | Williams et al., 2006                                               |
| DodgerLoopDay                                | Ihler et al., 2006                                                  |
| DodgerLoopGame                               | Ihler et al., 2006                                                  |
| DodgerLoopWeekend                            | Ihler et al., 2006                                                  |
| Geolife                                      | Zheng et al., 2009; Zheng et al., 2008; Zheng et al., 2010          |
| GestureMidAirD1                              | Caputo et al., 2018                                                 |
| GestureMidAirD2                              | Caputo et al., 2018                                                 |
| GestureMidAirD3                              | Caputo et al., 2018                                                 |
| GesturePebbleZ1                              | Mezari & Maglogiannis, 2018                                         |
| GesturePebbleZ2                              | Mezari & Maglogiannis, 2018                                         |
| GPS Data of Seabirds                         | Browning et al., 2018                                               |
| InsectWingbeat                               | Chen et al., 2014                                                   |
| JapaneseVowels                               | Kudo et al., 1999                                                   |
| Localization Data for Person Activity        | Vidulin et al., 2010                                                |
| MelbournePedestrian                          | City of Melbourne, 2019                                             |
| MIMIC-III Clinical Database (Demo)           | Johnson et al., 2016; Johnson et al., 2019; Goldberger et al., 2000 |
| PAMAP2 Physical Activity Monitoring          | Reiss & Stricker, 2012                                              |
| PhysioNet 2012                               | Silva et al., 2012                                                  |
| PhysioNet 2019                               | Reyna et al., 2020                                                  |
| PickupGestureWiimoteZ                        | Guna et al., 2014                                                   |
| PLAID                                        | Gao et al., 2014                                                    |
| Productivity Prediction of Garment Employees | Imran et al., 2021                                                  |
| ShakeGestureWiimoteZ                         | Guna et al., 2014                                                   |
| SpokenArabicDigits                           | Hammami & Bedda, 2010                                               |
| Taxi                                         | Moreira-Matias et al., 2013                                         |
| Vehicles                                     | Chorochronos Archive, 2019                                          |





# Citation
If you use this package in your research, please cite the following paper:

```bibtex
@misc{spinnato2025pyrregular,
      title={PYRREGULAR: A Unified Framework for Irregular Time Series, with Classification Benchmarks}, 
      author={Francesco Spinnato and Cristiano Landi},
      year={2025},
      eprint={2505.06047},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2505.06047}, 
}
```

