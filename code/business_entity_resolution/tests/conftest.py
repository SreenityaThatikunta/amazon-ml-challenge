import pytest

from ber.config import load_config


@pytest.fixture
def cfg(tmp_path):
    """Default config rooted in a temp dir (artifacts/output land there)."""
    (tmp_path / "dataset").mkdir()
    return load_config(overrides=[f"paths.root={tmp_path}"], run_id="test")
