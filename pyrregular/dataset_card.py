"""Generate the Hugging Face dataset card (README.md) from ``metadata/*.yml``.

Usage, from the repository root:

    python -m pyrregular.dataset_card > card.md
"""

from pyrregular.data_utils import list_metadata_files
from pyrregular.io_utils import load_yaml

# metadata license (SPDX id) -> Hugging Face license id, for the card header
HF_LICENSES = {
    "CC-BY-4.0": "cc-by-4.0",
    "CC-BY-NC-SA-4.0": "cc-by-nc-sa-4.0",
    "CC0-1.0": "cc0-1.0",
    "ODC-By-1.0": "odc-by",
    "ODbL-1.0": "odbl",
    "unknown": "unknown",
    "other": "other",
}
CARD_FIELDS = ("license", "license_source", "source", "citation")
REPO = "https://github.com/fspinna/pyrregular"

BODY = """
![Pyrregular: irregular time series datasets and benchmarks](https://github.com/fspinna/pyrregular/blob/main/assets/images/logo_01.png?raw=true)

# Pyrregular: Irregular Time Series Datasets and Benchmarks

**Published at ICLR 2026** · {n_datasets} datasets, including the benchmark of
34 datasets and 12 classifiers ·
[GitHub]({repo}) · [Documentation](https://fspinna.github.io/pyrregular/) ·
[PyPI](https://pypi.org/project/pyrregular/) ·
[Paper](https://openreview.net/forum?id=qetBM8nLkf)

Naturally irregular time series (uneven sampling, missing observations, signals
recorded at different times, variable-length sequences) in one standardized
format, for classification and regression. These are the datasets of the
[pyrregular]({repo}) Python library, including the 34 of the ICLR 2026
benchmark. The files are HDF5, read by the library (the dataset viewer is off):

```bash
pip install pyrregular
```

```python
from pyrregular import load_dataset
df = load_dataset("Garment.h5")
```

The files are versioned with git tags (`data-v1`, ...); each pyrregular release
downloads one fixed version.

## Licenses

Each dataset keeps the license of its original source; we redistribute it in a
converted format, with attribution. `unknown` means the source states no license
(most of these come from the UEA & UCR time series archive). If you hold the rights
to a dataset and want it removed or its attribution changed, please open an issue
at {repo}/issues.

| Dataset | License | Source | Citation |
|---|---|---|---|
{table}

## Citation

Please cite the original source of each dataset (Citation column) and pyrregular:

```bibtex
@inproceedings{{
    spinnato2026pyrregular,
    title={{{{PYRREGULAR}}: A Unified Framework for Irregular Time Series, with Classification Benchmarks}},
    author={{Francesco Spinnato and Cristiano Landi}},
    booktitle={{The Fourteenth International Conference on Learning Representations}},
    year={{2026}},
    url={{https://openreview.net/forum?id=qetBM8nLkf}}
}}
```
"""


def _link(text, url):
    return text if url in ("NA", "tbd") else f"[{text}]({url})"


def dataset_card():
    """Return the dataset card as a string; fail on incomplete metadata."""
    rows, licenses = [], set()
    for file in list_metadata_files():
        meta = load_yaml(file)
        missing = [field for field in CARD_FIELDS if field not in meta]
        if missing:
            raise ValueError(f"{file.name}: missing {missing}")
        if meta["license"] not in HF_LICENSES:
            raise ValueError(f"{file.name}: unmapped license {meta['license']!r}")
        licenses.add(HF_LICENSES[meta["license"]])
        rows.append(
            f"| {file.stem} | {_link(meta['license'], meta['license_source'])} "
            f"| {_link('source', meta['source'])} | {_link('cite', meta['citation'])} |"
        )
    header = ["---", "pretty_name: pyrregular", "viewer: false", "license:"]
    header += [f"- {license}" for license in sorted(licenses)]
    if "other" in licenses:
        header.append("license_name: dataset-specific")
    header.append("---")
    body = BODY.format(repo=REPO, n_datasets=len(rows), table="\n".join(rows))
    return "\n".join(header) + "\n" + body


if __name__ == "__main__":
    print(dataset_card())
