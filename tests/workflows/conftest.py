"""Tests consume the same public synthetic fixture builder as downstream owners."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import pytest

spec = spec_from_file_location("ks_shared_fixture", Path(__file__).parent / "fixtures/v1/row_softmax/build.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def make_bundle(tmp_path):
    def build(operation="implement", *, directory=None):
        return module.build_bundle(directory or tmp_path / operation, operation)
    return build
