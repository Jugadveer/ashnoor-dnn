"""Get the Mathematics data, turn final grades into bands, split off a test set."""

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from mathgrades.settings import SETTINGS


def fetch_csv() -> None:
    """Save student-mat.csv into data/ unless it is already there.

    UCI ships a zip that holds another zip (student.zip) with the CSV files.
    """
    if SETTINGS.csv_path.exists():
        return
    print(f"Downloading {SETTINGS.dataset_url}")
    with urllib.request.urlopen(SETTINGS.dataset_url) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    inner = zipfile.ZipFile(io.BytesIO(archive.read("student.zip")))
    SETTINGS.csv_path.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.csv_path.write_bytes(inner.read(SETTINGS.csv_path.name))


def load_students() -> pd.DataFrame:
    fetch_csv()
    return pd.read_csv(SETTINGS.csv_path, sep=";")  # fields are separated by ';'


def to_band(grades) -> np.ndarray:
    """Final grade (0-20) -> band index 0..3.

    np.digitize counts how many band starts each grade has reached:
    grade 8 has reached none (0, At risk), grade 14 has reached 10 and 13 (2, Good).
    """
    return np.digitize(grades, SETTINGS.band_starts)


def features_and_bands(students: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """All 32 columns except G3 are inputs (G1 and G2 included); the band of G3 is the label."""
    bands = pd.Series(to_band(students[SETTINGS.target]), index=students.index, name="band")
    return students.drop(columns=SETTINGS.target), bands


def hold_out_test(X: pd.DataFrame, y: pd.Series):
    """75 % training, 25 % test, with the same band proportions in both."""
    return train_test_split(X, y, test_size=SETTINGS.test_share, stratify=y,
                            random_state=SETTINGS.seed)
