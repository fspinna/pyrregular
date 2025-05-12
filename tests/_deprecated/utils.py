import importlib
import importlib.util
from pathlib import Path

from pyrregular.data_utils import get_project_root


def filename_to_classname(filename):
    return "".join(word.capitalize() for word in filename.stem.split("_"))


def load_classes(directory_path):
    directory = Path(directory_path)

    classes = []
    class_names = []

    for file_path in directory.glob("*.py"):
        if file_path.name == "__init__.py":
            continue  # Skip __init__.py file
        # Convert filename to class name
        class_name = filename_to_classname(file_path)
        class_names.append(class_name)

        # Load the module
        spec = importlib.util.spec_from_file_location(class_name, file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Get the class from the module
        cls = getattr(module, class_name)
        classes.append(cls)

    return classes, class_names


TEST_CASES, IDS = load_classes(get_project_root() / "datasets")
IDS_FAST = [i for i, f in zip(IDS, TEST_CASES) if f.fast_to_test]
TEST_CASES_FAST = [f for i, f in zip(IDS, TEST_CASES) if f.fast_to_test]
