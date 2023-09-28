import pytest
from epmatools import Oxides


@pytest.fixture
def data():
    return Oxides.from_examples("minerals").set_index("Comment")
