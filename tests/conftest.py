import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import simulate  # noqa: E402


@pytest.fixture(scope="session")
def counts():
    return simulate.counts()


@pytest.fixture(scope="session")
def genes():
    return simulate.gene_names(1500)
