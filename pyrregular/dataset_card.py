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
# pyrregular datasets

Irregular time series datasets converted to the format of the
[pyrregular]({repo}) library ([paper](https://openreview.net/forum?id=qetBM8nLkf)).

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
    header = ["---", "pretty_name: pyrregular", "license:"]
    header += [f"- {license}" for license in sorted(licenses)]
    if "other" in licenses:
        header.append("license_name: dataset-specific, see the Licenses table")
    header.append("---")
    return "\n".join(header) + "\n" + BODY.format(repo=REPO, table="\n".join(rows))


if __name__ == "__main__":
    print(dataset_card())
