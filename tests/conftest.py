import pytest

from mathgrades.dataset import features_and_bands, hold_out_test, load_students
from mathgrades.training import make_deterministic


@pytest.fixture(scope="session")
def students():
    make_deterministic()
    return load_students()


@pytest.fixture(scope="session")
def split(students):
    """X_train, X_test, y_train, y_test exactly as training uses them."""
    X, y = features_and_bands(students)
    return hold_out_test(X, y)
