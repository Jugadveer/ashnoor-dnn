"""Maths grade band predictor. Run (after training):  streamlit run streamlit_app.py"""

import json

import altair as alt
import pandas as pd
import streamlit as st

from mathgrades.attributes import ATTRIBUTES, BY_NAME, GROUPS
from mathgrades.charts import confusion_file_name
from mathgrades.inference import GradePredictor
from mathgrades.search import NETWORK_NAME
from mathgrades.settings import SETTINGS

INK_BLUE = st.get_option("theme.primaryColor")
INK = st.get_option("theme.textColor")
RED_PEN = "#c8312b"  # the predicted band, marked as a teacher would (same as exercise_book.css)
POSITION = {attribute.name: index for index, attribute in enumerate(ATTRIBUTES)}
REQUIRED = (SETTINGS.pipeline_path, SETTINGS.input_form_path, SETTINGS.results_path)


@st.cache_resource  # one copy of the model for the whole server, loaded once
def get_predictor() -> GradePredictor:
    return GradePredictor()


@st.cache_data
def read_json(path_text: str) -> dict:
    with open(path_text) as handle:
        return json.load(handle)


def percent(value: float) -> str:
    return f"{value:.0%}"


# ---------------------------------------------------------------- inputs
def ask(field: dict):
    """Draw the widget for one field. Every widget only offers values seen in
    the training data, so every answer is one the model has learnt from."""
    question = field["question"]
    if field["kind"] == "number":
        if field["high"] - field["low"] <= 4:  # a few whole numbers: tick one
            choices = list(range(field["low"], field["high"] + 1))
            return st.radio(question, choices, index=choices.index(field["start"]), horizontal=True)
        return st.number_input(question, field["low"], field["high"], field["start"], step=1)
    values = [value for value, _ in field["options"]]
    wording = {value: words for value, words in field["options"]}
    return st.selectbox(question, values, index=values.index(field["start"]),
                        format_func=wording.get)


def given_section(form: list[dict]) -> dict:
    """The student's details, written out like the 'given' part of an exercise:
    grades on the left, background on the right."""
    st.subheader("Given", anchor=False)
    asked_groups = [group for group in GROUPS if any(field["group"] == group for field in form)]
    asked_groups.sort(key=lambda group: group != "Academic")  # grades first
    student = {}
    for group, column in zip(asked_groups, st.columns(len(asked_groups), gap="large")):
        # Natural order (G1 before G2), not the order of the feature tests.
        fields = sorted((field for field in form if field["group"] == group),
                        key=lambda field: POSITION[field["name"]])
        with column:
            for field in fields:
                student[field["name"]] = ask(field)
    return student


# ---------------------------------------------------------------- report card
def probability_chart(probabilities, band: int) -> alt.LayerChart:
    """One column per band, drawn in blue outline; the predicted band is filled in red."""
    table = pd.DataFrame({"band": SETTINGS.band_names, "probability": probabilities})
    table["predicted"] = table.index == band
    colour = alt.condition("datum.predicted", alt.value(RED_PEN), alt.value(INK_BLUE))
    columns = alt.Chart(table).mark_bar(size=46, strokeWidth=1.5).encode(
        x=alt.X("band:N", sort=list(SETTINGS.band_names), title=None,
                axis=alt.Axis(labelAngle=0, labelLimit=0, labelFontSize=14)),
        y=alt.Y("probability:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, 0.5, 1], grid=False)),
        stroke=colour,
        fill=alt.condition("datum.predicted", alt.value(RED_PEN), alt.value("transparent")),
        tooltip=[alt.Tooltip("band:N", title="Band"),
                 alt.Tooltip("probability:Q", title="Probability", format=".1%")],
    )
    values = columns.mark_text(dy=-9, fontSize=14).encode(
        text=alt.Text("probability:Q", format=".0%"), color=colour)
    return (columns + values).properties(height=200, background="transparent").configure_view(
        strokeWidth=0)


def answer_block(band: int, probabilities) -> None:
    """The predicted band in red pen, double-underlined like a final answer."""
    st.subheader("Answer", anchor=False)
    st.html(f"""
        <div class="answer">
          <div class="label">final band</div>
          <div class="band">{SETTINGS.band_names[band]}</div>
          <div class="detail">Final grade most likely <b>{SETTINGS.grade_range(band)}</b> out of 20
            &nbsp;·&nbsp; chance of passing (10 or more): <b>{percent(1 - probabilities[0])}</b></div>
        </div>""")


def what_if_table(predictor: GradePredictor, student: dict) -> pd.DataFrame:
    """The student as entered and with two realistic changes."""
    scenarios = {"As entered": student}
    if "G2" in student:
        scenarios["G2 two points higher"] = {**student, "G2": min(student["G2"] + 2, 20)}
    if student.get("failures", 0) > 0:
        scenarios["No past failures"] = {**student, "failures": 0}
    rows = []
    for label, case in scenarios.items():
        probabilities = predictor.probabilities(case)
        rows.append({"Scenario": label, "Band": SETTINGS.band_names[predictor.band(case)],
                     "Chance of passing": percent(1 - probabilities[0])})
    return pd.DataFrame(rows).set_index("Scenario")


def describe(name: str, value, form: list[dict]) -> str:
    """'Second-term maths grade (0-20): 14' with code values turned into words."""
    field = next(field for field in form if field["name"] == name)
    if field["kind"] != "number":
        value = dict((v, words) for v, words in field["options"])[value]
    return f"{BY_NAME[name].question}: {value}"


def working_section(predictor: GradePredictor, student: dict, band: int) -> None:
    """The working: each detail that differs from the typical student, and how
    far it moves the chance of the predicted band."""
    st.subheader("Working", anchor=False)
    drivers = predictor.drivers(student, band).head(4)
    if drivers.empty:
        st.write("Every detail is at the typical student's value, so nothing stands out.")
        return
    lines = []
    for row in drivers.itertuples():
        direction = "towards" if row.effect > 0 else "away from"
        lines.append(f"<li>{describe(row.input, row.value, predictor.form)} "
                     f"<span class='arrow'>&rarr;</span> {abs(row.effect) * 100:.0f} points "
                     f"{direction} {SETTINGS.band_names[band]}</li>")
    st.html(f"<ul class='working'>{''.join(lines)}</ul>")
    st.caption("Each detail is set back to the typical student's value on its own; "
               "the change in the predicted band's probability is its effect.")


def report_card_page() -> None:
    predictor = get_predictor()
    st.title("Final maths band", anchor=False)
    st.write("GradeNet, an ordinal neural network, reads a student's term grades and "
             "background and works out which band their final maths grade (G3, 0 to 20) "
             "will fall in: " + ", ".join(f"{name} {SETTINGS.grade_range(i)}"
                                           for i, name in enumerate(SETTINGS.band_names)) + ".")

    student = given_section(predictor.form)
    probabilities = predictor.probabilities(student)
    band = predictor.band(student)

    answer_block(band, probabilities)
    st.altair_chart(probability_chart(probabilities, band), width="stretch")
    st.caption("The predicted band is the middle of the model's distribution: the student "
               "is at least as likely to land at or below it as at or above it.")
    st.html("<div class='margin-note'>Copying the band of G2 is right as often as GradeNet "
            "(see Model results). The network's extra is the chance for every band.</div>")

    working_section(predictor, student, band)

    st.subheader("What if", anchor=False)
    st.table(what_if_table(predictor, student))
    if student.get("failures", 0) == 0:
        st.caption("Past failures is already 0, so there is no 'no failures' scenario.")


# ---------------------------------------------------------------- results page
def interval(model: dict, metric: str) -> str:
    low, high = model["test_ci"][metric]
    return f"{model['test'][metric]:.2f} ({low:.2f}-{high:.2f})"


def headline(results: dict) -> None:
    net, naive = results["models"][NETWORK_NAME], results["models"]["Naive: band of G2"]
    st.subheader("In short", anchor=False)
    rows = [
        ("Pass / fail correct", "pass_fail_accuracy"),
        ("Within one band", "within_one_band"),
        ("Exact band", "accuracy"),
        ("Weighted kappa (main score)", "weighted_kappa"),
    ]
    cv_label = f"CV ({SETTINGS.compare_folds}x{SETTINGS.compare_repeats})"
    st.table(pd.DataFrame(
        [{"Measure": label,
          f"GradeNet, {cv_label}": f"{net['cv'][metric][0]:.3f}",
          f"Copy G2 band, {cv_label}": f"{naive['cv'][metric][0]:.3f}",
          "GradeNet, test (95 % CI)": interval(net, metric),
          "Copy G2 band, test": f"{naive['test'][metric]:.2f}"}
         for label, metric in rows]).set_index("Measure"))
    auc = net["test_at_risk_auc"]
    st.write(f"Spotting failing students: GradeNet ranks a random failing student above a "
             f"random passing one {percent(auc)} of the time (ROC-AUC {auc:.2f}).")


def comparison_table(results: dict) -> pd.DataFrame:
    rows = []
    for name, model in results["models"].items():
        cv_mean, cv_std = model["cv"]["weighted_kappa"]
        auc = model["test_at_risk_auc"]
        rows.append({"Model": name, "CV kappa": f"{cv_mean:.3f} ± {cv_std:.3f}",
                     "Test kappa (95 % CI)": interval(model, "weighted_kappa"),
                     "Test accuracy (95 % CI)": interval(model, "accuracy"),
                     "Within one band": f"{model['test']['within_one_band']:.2f}",
                     "Pass/fail accuracy": f"{model['test']['pass_fail_accuracy']:.2f}",
                     "At-risk recall": f"{model['test']['at_risk_recall']:.2f}",
                     "At-risk AUC": "n/a" if auc is None else f"{auc:.2f}"})
    return pd.DataFrame(rows).set_index("Model")


def significance_text(results: dict) -> None:
    st.subheader("Is GradeNet really better?", anchor=False)
    for opponent, test in results["paired_tests"].items():
        mcnemar, paired = test["mcnemar"], test["five_by_two"]
        st.write(f"- **vs {opponent}.** Test set: GradeNet alone was right for "
                 f"{mcnemar['a_only_right']} students, {opponent} alone for "
                 f"{mcnemar['b_only_right']}; McNemar p = {mcnemar['p_value']:.3f}, "
                 f"{significance(mcnemar['p_value'])}. Cross-validation (5x2): kappa "
                 f"difference {paired['mean_difference']:+.3f}, t-test p = "
                 f"{paired['p_value']:.3f}, {significance(paired['p_value'])}.")


def significance(p_value: float) -> str:
    return "significant at 5 %" if p_value < 0.05 else "not significant"


def bar_chart(table: pd.DataFrame, value: str, label: str, title: str) -> alt.Chart:
    return alt.Chart(table).mark_bar(color=INK_BLUE).encode(
        x=alt.X(f"{value}:Q", title=title),
        y=alt.Y(f"{label}:N", sort="-x", title=None, axis=alt.Axis(labelLimit=0)),
        tooltip=[label, alt.Tooltip(f"{value}:Q", format=".3f")],
    ).properties(height=alt.Step(22), background="transparent")


def experiments_section() -> None:
    if not SETTINGS.experiments_path.exists():
        st.info("Ablation, learning curve, calibration and fairness appear here after "
                "`python -m mathgrades.experiments` has run (a few minutes).")
        return
    experiments = read_json(str(SETTINGS.experiments_path))

    st.subheader("Ablation: one design choice changed at a time", anchor=False)
    ablation = pd.DataFrame(experiments["ablation"])
    ablation["CV kappa"] = ablation.apply(lambda r: f"{r.cv_kappa:.3f} ± {r.cv_kappa_std:.3f}", axis=1)
    ablation["Change"] = ablation["change"].map(lambda c: f"{c:+.3f}")
    st.table(ablation[["variant", "CV kappa", "Change"]].set_index("variant"))

    st.subheader("Which grades the model may see", anchor=False)
    setups = pd.DataFrame(experiments["grade_setups"])
    table = setups.pivot(index="model", columns="setup", values="cv_kappa")
    st.table(table[["G1 and G2", "G1 only", "No grades"]].map(
        lambda kappa: "n/a" if pd.isna(kappa) else f"{kappa:.3f}"))

    st.subheader("Learning curve and calibration", anchor=False)
    st.image(str(SETTINGS.figures_dir / "learning_curve.png"))
    st.image(str(SETTINGS.figures_dir / "reliability.png"))
    calibration = pd.DataFrame(experiments["calibration"]["versions"]).T
    st.table(calibration.map(lambda score: f"{score:.3f}").rename(
        columns={"brier": "Brier score", "ece": "Expected calibration error",
                 "log_loss": "Log loss"}))

    st.subheader("Fairness: test scores by group", anchor=False)
    fairness = pd.DataFrame(experiments["fairness"])
    st.table(pd.DataFrame({
        "Group": fairness["column"] + " = " + fairness["value"],
        "Students": fairness["students"],
        "Kappa": fairness["weighted_kappa"].map(lambda k: f"{k:.2f}"),
        "Exact band": fairness["accuracy"].map(percent),
        "Actually At risk": fairness["actual_at_risk_share"].map(percent),
        "Predicted At risk": fairness["predicted_at_risk_share"].map(percent),
    }).set_index("Group"))
    st.caption("Groups with few test students (MS has 7) are too small to judge.")


def results_page() -> None:
    st.set_page_config(layout="wide")  # the comparison tables need the full width
    results = read_json(str(SETTINGS.results_path))
    st.title("How well the models do", anchor=False)
    st.write(f"{results['train_students']} students trained the models and "
             f"{results['test_students']} were held back as the test set. CV scores come from "
             f"repeated cross-validation on the training students; test scores carry 95 % "
             f"bootstrap intervals.")
    headline(results)

    st.subheader("Every model", anchor=False)
    st.table(comparison_table(results))
    significance_text(results)

    st.subheader("What GradeNet relies on", anchor=False)
    importance = pd.DataFrame(results["permutation_importance"])
    importance = importance[importance["kappa_drop"].abs() > 1e-9]
    st.altair_chart(bar_chart(importance, "kappa_drop", "column",
                              "Drop in test kappa when the column is shuffled"), width="stretch")

    experiments_section()

    st.subheader("Confusion matrices", anchor=False)
    model = st.selectbox("Model", list(results["models"]))
    st.image(str(SETTINGS.figures_dir / confusion_file_name(model)))


def main() -> None:
    st.set_page_config(page_title="Maths grade band", layout="centered")
    st.html(f"<style>{SETTINGS.style_path.read_text()}</style>")  # squared paper, red pen
    missing = [path.name for path in REQUIRED if not path.exists()]
    if missing:
        st.title("Maths grade band", anchor=False)
        st.info(f"The model has not been trained on this computer yet (missing: "
                f"{', '.join(missing)}). From the project folder run "
                f"`python -m mathgrades.train` (a few minutes), then reload this page.")
        st.stop()
    pages = [st.Page(report_card_page, title="Report card", default=True),
             st.Page(results_page, title="Model results", url_path="results")]
    st.navigation(pages, position="top").run()


main()
