from pathlib import Path

import tomli

import pamiq_torch_modules


def test_version():
    project_path = Path(__file__).parent.parent / "pyproject.toml"
    with open(project_path, "rb") as f:
        pyproject = tomli.load(f)

    assert pyproject["project"]["version"] == pamiq_torch_modules.__version__
