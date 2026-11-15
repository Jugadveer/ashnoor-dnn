"""Everything the web app needs: the input form description and a predictor."""

import json

import joblib
import numpy as np
import pandas as pd

from mathgrades.attributes import BY_NAME
from mathgrades.explain import prediction_drivers
from mathgrades.selection import is_numeric
from mathgrades.settings import SETTINGS


def build_input_form(X: pd.DataFrame, columns: list[str]) -> list[dict]:
    """Describe each raw column the model reads, so the app can draw the right widget.

    kind "number":  a count or grade -> slider between the training min and max
    kind "ordered": a coded scale such as study hours 1-4 -> labelled slider
    kind "choice":  text categories such as the mother's occupation -> dropdown
    Ranges and options come from the training students, so the app never asks
    the model about a value it has not seen. The starting value is the typical
    training student (median or most common answer).
    """
    form = []
    for name in columns:
        attribute = BY_NAME[name]
        field = {"name": name, "group": attribute.group, "question": attribute.question}
        if attribute.choices is None:
            field.update(kind="number", low=int(X[name].min()), high=int(X[name].max()),
                         start=int(X[name].median()))
        else:
            seen = set(X[name].unique().tolist())
            field.update(kind="ordered" if is_numeric(X[name]) else "choice",
                         options=[[value, wording] for value, wording in attribute.choices.items()
                                  if value in seen],
                         start=X[name].mode().tolist()[0])
        form.append(field)
    return form


class GradePredictor:
    """Loads the saved pipeline and form once; predicts many times."""

    def __init__(self):
        self.pipeline = joblib.load(SETTINGS.pipeline_path)
        self.form = json.loads(SETTINGS.input_form_path.read_text())
        self.typical = {field["name"]: field["start"] for field in self.form}

    def probabilities(self, student: dict) -> np.ndarray:
        """One probability per band for one student {column: value}."""
        return self.pipeline.predict_proba(pd.DataFrame([student]))[0]

    def band(self, student: dict) -> int:
        """The predicted band (for the ordinal network: the median band)."""
        return int(self.pipeline.predict(pd.DataFrame([student]))[0])

    def drivers(self, student: dict, band: int) -> pd.DataFrame:
        return prediction_drivers(self.probabilities, student, self.typical, band)
