"""All tunable numbers and file locations, held in one frozen dataclass.

Frozen means the values cannot be changed while the program runs, so every
module sees the same settings. To experiment, edit the defaults here.
"""

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Settings:
    # ---- data
    dataset_url: str = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
    csv_path: Path = ROOT / "data" / "student-mat.csv"   # Mathematics course, 395 students
    target: str = "G3"                                     # final grade, 0-20

    # A final grade below 10 is "At risk", 10-12 "Average", 13-15 "Good",
    # 16 and above "Excellent". Each number is where a band starts.
    band_starts: tuple[int, ...] = (10, 13, 16)
    band_names: tuple[str, ...] = ("At risk", "Average", "Good", "Excellent")

    # ---- feature selection
    significance_level: float = 0.05   # keep a column if its test p-value is below this

    # ---- splitting
    seed: int = 7
    test_share: float = 0.25    # held-out test students
    stop_share: float = 0.2     # part of each training set used to decide when to stop
    cv_folds: int = 3           # folds used to score every Optuna trial

    # ---- network training
    max_epochs: int = 300
    stop_patience: int = 20     # epochs without improvement before stopping
    lr_patience: int = 8        # epochs without improvement before halving the learning rate
    label_smoothing: float = 0.1

    # ---- tuning (one Optuna study per model family, same folds and objective)
    network_trials: int = 40
    svm_trials: int = 40
    boosting_trials: int = 40

    # ---- final comparison and statistics
    compare_folds: int = 5      # repeated stratified 5-fold CV ...
    compare_repeats: int = 3    # ... repeated 3 times = 15 scores per model
    bootstrap_resamples: int = 2000
    permutation_repeats: int = 20
    ensemble_size: int = 5      # networks with different seeds in the seed-ensemble experiment
    ablation_repeats: int = 3   # repeats of the cv_folds split in the ablation study

    # ---- outputs
    pipeline_path: Path = ROOT / "models" / "grade_pipeline.joblib"
    input_form_path: Path = ROOT / "models" / "input_form.json"
    results_path: Path = ROOT / "reports" / "results.json"
    experiments_path: Path = ROOT / "reports" / "experiments.json"
    feature_tests_path: Path = ROOT / "reports" / "feature_tests.csv"
    trials_dir: Path = ROOT / "reports" / "trials"
    worst_mistakes_path: Path = ROOT / "reports" / "worst_mistakes.csv"
    figures_dir: Path = ROOT / "reports" / "figures"
    theme_path: Path = ROOT / ".streamlit" / "config.toml"
    style_path: Path = ROOT / ".streamlit" / "exercise_book.css"   # the app's page styling

    @property
    def n_classes(self) -> int:
        return len(self.band_names)

    def grade_range(self, band: int) -> str:
        """'13-15' for band 2, built from band_starts."""
        lows = (0, *self.band_starts)
        highs = (*(start - 1 for start in self.band_starts), 20)
        return f"{lows[band]}-{highs[band]}"


SETTINGS = Settings()
