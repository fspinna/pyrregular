import pathlib

import pooch

from pyrregular.data_utils import data_final_folder, get_project_root


def new_registry_from_data_folder(folder=data_final_folder()):
    folder = pathlib.Path(folder)
    new_registry = {}
    for file in list(pathlib.Path(folder).glob("*.h5")):
        new_registry[file.name] = pooch.file_hash(str(folder / file.name))
    return new_registry


def save_registry(folder=data_final_folder()):
    with open(get_project_root() / "registry.txt", "w") as f:
        for name, hash in new_registry_from_data_folder(folder=folder).items():
            f.write(f"{name} {hash}\n")


if __name__ == "__main__":
    save_registry()
