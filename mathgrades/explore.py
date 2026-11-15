"""Exploratory data analysis. Run:  python -m mathgrades.explore"""

import matplotlib.pyplot as plt
import pandas as pd

from mathgrades.attributes import ATTRIBUTES, BY_NAME, GROUPS
from mathgrades.charts import RED_PEN, INK_BLUE, GREY, PAPER, save
from mathgrades.dataset import load_students, to_band
from mathgrades.settings import SETTINGS

# Columns compared against the final grade in the box-plot chart.
BOX_PLOT_COLUMNS = ["failures", "higher", "studytime", "schoolsup", "Medu", "goout"]


def describe(students: pd.DataFrame) -> None:
    """Print the facts that the project brief states, so they can be checked."""
    inputs = [a.name for a in ATTRIBUTES if a.name not in ("G1", "G2")]
    print(f"Rows: {len(students)}   (brief: 395 in the Mathematics file)")
    print(f"Columns: {students.shape[1]} = {len(inputs)} attributes + G1, G2 + target G3")
    for group in GROUPS:
        print(f"  {group:<12} {', '.join(a.name for a in ATTRIBUTES if a.group == group)}")
    print(f"Missing values: {students.isna().sum().sum()}   Duplicate rows: {students.duplicated().sum()}")
    print(f"Students with G3 = 0: {(students[SETTINGS.target] == 0).sum()}")

    counts = pd.Series(to_band(students[SETTINGS.target])).value_counts().sort_index()
    print("\nStudents per band:")
    for band, count in counts.items():
        print(f"  {SETTINGS.band_names[band]:<10} G3 {SETTINGS.grade_range(band):<6}"
              f"{count:>4}  ({count / len(students):.0%})")


def grade_histogram(students: pd.DataFrame) -> None:
    """Every final grade, with dashed lines where the bands start."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(students[SETTINGS.target], bins=range(0, 22), color=INK_BLUE, edgecolor=PAPER)
    for start in SETTINGS.band_starts:
        ax.axvline(start, color=RED_PEN, linestyle="--")
    ax.set(title="Final maths grade (G3); dashed lines start a new band",
           xlabel="G3", ylabel="Students")
    save(fig, "eda_grade_histogram.png")


def grade_by_column(students: pd.DataFrame) -> None:
    """Distribution of G3 for each value of a few columns we expect to matter."""
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for name, ax in zip(BOX_PLOT_COLUMNS, axes.flat):
        values = sorted(students[name].unique())
        groups = [students.loc[students[name] == value, SETTINGS.target] for value in values]
        ax.boxplot(groups, tick_labels=[str(v) for v in values], patch_artist=True,
                   boxprops={"facecolor": INK_BLUE, "color": INK_BLUE},
                   medianprops={"color": "white"}, flierprops={"markeredgecolor": GREY})
        ax.set(title=BY_NAME[name].question, ylabel="G3")
    save(fig, "eda_grade_by_column.png")


def rank_correlations(students: pd.DataFrame) -> None:
    """Spearman rank correlation of every numeric column with G3.
    Spearman works on ranks, so it also captures monotonic curved links."""
    numeric = students.select_dtypes("number")
    rho = numeric.corr(method="spearman")[SETTINGS.target].drop(SETTINGS.target).sort_values()
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(rho.index, rho.values, color=[INK_BLUE if r > 0 else RED_PEN for r in rho.values])
    ax.axvline(0, color=GREY, linewidth=1)
    ax.set(title="Spearman correlation with the final grade", xlabel="rho")
    save(fig, "eda_spearman.png")
    print("\nSpearman correlation with G3:")
    print(rho.round(2).to_string())


def period_grades(students: pd.DataFrame) -> None:
    """G1 and G2 against G3. Points at G3 = 0 are students with no final grade."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, ax in zip(["G1", "G2"], axes):
        ax.scatter(students[name], students[SETTINGS.target], color=INK_BLUE, alpha=0.35)
        ax.set(xlabel=name, ylabel="G3", title=f"{name} against the final grade")
    save(fig, "eda_period_grades.png")


def main() -> None:
    students = load_students()
    describe(students)
    grade_histogram(students)
    grade_by_column(students)
    rank_correlations(students)
    period_grades(students)
    print(f"\nCharts saved in {SETTINGS.figures_dir}")


if __name__ == "__main__":
    main()
