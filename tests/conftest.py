import pytest
from empatools import Oxides


@pytest.fixture
def data():
    return Oxides.example_data().set_index("Comment")
