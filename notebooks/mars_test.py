"""Reproduce the MARS comparison on the public SNAPS test set.

The notebook deliberately keeps orchestration and presentation here. Parsers,
downloaders, statistics, and plotting live in ``notebooks.modules`` so they can
be reused by other analyses.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from pathlib import Path

    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set
    from modules.results import assignment_table

    return (
        Path,
        assignment_table,
        compare_results,
        mo,
        pd,
        plot_comparisons,
        run_test_set,
    )


@app.cell
def _(Path):
    BMRB_IDS = [5471, 4354, 4384, 4022, 4106, 4457, 4402, 4341, 547, 4082, 4136]
    BMRB_NAMES = {
        5471: "Malate synthase G",
        4354: "Maltose binding protein",
        4384: "Rous Sarcoma Virus capsid",
        4022: "Human carbonic anhydrase I",
        4106: "N-terminal domain of enzyme I (EIN)",
        4457: "E-cadherin domains II and III",
        4402: "Human prion protein",
        4341: "Superoxide dismutase",
        547: "Calmodulin/M13",
        4082: "Profilin",
        4136: "E. coli EmrE",
    }
    DATA_DIR = Path(__file__).resolve().parent / "mars_test_set"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mars_test_result"
    STRATEGIES = {
        "I": "{bmrb_id}-test-result-direct.txt",
        "II": "{bmrb_id}-test-result-iterative.txt",
        "III": "{bmrb_id}-test-result-triplet.txt",
    }
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR, STRATEGIES


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # SNAPS vs MARS

    This walkthrough downloads the BMRB test inputs, runs the three SNAPS
    strategies, compares their assignment quality with the published MARS
    counts, and ends with one figure per SNAPS condition.
    """)
    return


@app.cell
def _(BMRB_IDS, DATA_DIR, OUTPUT_DIR, mo, run_test_set):
    mo.md("## 1. Prepare the test set\n\nExisting inputs and result files are reused.")
    run_test_set(BMRB_IDS, DATA_DIR, OUTPUT_DIR)
    test_set_ready = True
    return (test_set_ready,)


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    OUTPUT_DIR,
    STRATEGIES,
    assignment_table,
    test_set_ready,
):
    if not test_set_ready:
        raise RuntimeError("The test set was not prepared")
    snaps_results = assignment_table(BMRB_IDS, STRATEGIES, BMRB_NAMES, OUTPUT_DIR)
    return (snaps_results,)


@app.cell
def _(pd):
    # Published MARS counts used in the original benchmark.
    mars_data = [
        ("Malate synthase G", "C′, Cα, Cβ", (652, 639, 0), (652, 639, 0), (651, 623, 0)),
        ("Malate synthase G", "Cα, Cβ*", (500, 207, 0), (639, 584, 2), (622, 511, 0)),
        ("Maltose binding protein", "Cα, Cβ", (323, 303, 0), (333, 324, 0), (330, 313, 1)),
        ("Rous Sarcoma Virus capsid", "C′, Cα, Cβ", (214, 205, 0), (218, 207, 0), (218, 199, 0)),
        ("Human carbonic anhydrase I", "C′, Cα, Cβ", (242, 235, 0), (242, 237, 0), (242, 225, 0)),
        ("N-terminal domain of enzyme I (EIN)", "Cα, Cβ", (246, 232, 0), (246, 246, 0), (248, 245, 0)),
        ("E-cadherin domains II and III", "Cα, Cβ", (116, 77, 1), (134, 102, 0), (136, 70, 1)),
        ("Human prion protein", "Cα, Cβ", (138, 103, 0), (155, 127, 0), (154, 118, 0)),
        ("Superoxide dismutase", "C′, Cα, Cβ", (112, 101, 0), (112, 104, 0), (111, 100, 0)),
        ("Superoxide dismutase", "Cα, Cβ", (111, 101, 0), (112, 104, 0), (112, 103, 0)),
        ("Calmodulin/M13", "Cα, C′", (97, 37, 0), (144, 142, 0), (136, 119, 0)),
        ("Profilin", "Cα, Cβ", (130, 132, 2), (132, 132, 0), (132, 123, 0)),
        ("E. coli EmrE", "C′, Cα, Cβ", (61, 35, 0), (70, 58, 0), (64, 50, 0)),
        ("Human ubiquitin", "Cα, Cβ", (72, 72, 0), (72, 72, 0), (72, 70, 0)),
        ("Human ubiquitin", "Cα only", (32, 9, 0), (58, 18, 0), (58, 9, 0)),
        ("Z domain", "Cα, Cβ, Hα†", (65, 65, 0), None, None),
        ("Z domain", "Cα, Cβ‡", (57, 34, 0), None, None),
        ("Tir110", "C′, Cα, Cβ§", (91, 80, 0), None, None),
    ]
    conditions = ["I", "II", "III"]
    rows = []
    for protein, shifts_used, *values_by_condition in mars_data:
        for condition, values in zip(conditions, values_by_condition):
            assignments, reliable, errors = values or (pd.NA,) * 3
            rows.append(
                {
                    "protein": protein,
                    "chemical_shifts_used": shifts_used,
                    "condition": condition,
                    "assignments": assignments,
                    "reliable_assignments": reliable,
                    "errors": errors,
                }
            )
    mars_results = pd.DataFrame(rows)
    mars_results[["assignments", "reliable_assignments", "errors"]] = mars_results[
        ["assignments", "reliable_assignments", "errors"]
    ].astype("Int64")
    return (mars_results,)


@app.cell
def _(mars_results, mo, snaps_results):
    mo.md("## 2. Assignment counts\n\n### SNAPS\n\n" + snaps_results.to_markdown(index=False)
          + "\n\n### MARS\n\n" + mars_results.to_markdown(index=False))
    return


@app.cell
def _(compare_results, mars_results, snaps_results):
    comparison = compare_results(snaps_results, mars_results)
    return (comparison,)


@app.cell
def _(comparison, mo):
    mo.md("## 3. Comparison statistics\n\n" + comparison.to_markdown(index=False))
    return


@app.cell(hide_code=True)
def _(mo, plots):
    mo.md(rf"""
    ## 4. Figures

    Positive differences mean that SNAPS has a higher correct-assignment fraction.

    ### 4.1 Direct Strategy
    {mo.as_html(plots[0])}

    ### 4.2 Iterative Strategy
    {mo.as_html(plots[1])}

    ### 4.3 Triplet Strategy
    {mo.as_html(plots[2])}
    """)
    return


@app.cell
def _(comparison, plot_comparisons):
    plots = plot_comparisons(comparison)
    return (plots,)


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    OUTPUT_DIR,
    STRATEGIES,
    assignment_table,
    compare_results,
    mars_results,
    plot_comparisons,
):
    snaps_results_high_med = assignment_table(BMRB_IDS, STRATEGIES, BMRB_NAMES, OUTPUT_DIR, good_confidences=['High', 'Medium'])
    comparison_high_med = compare_results(snaps_results_high_med, mars_results)
    plots_high_med = plot_comparisons(comparison_high_med)
    return (plots_high_med,)


@app.cell(hide_code=True)
def _(mo, plots_high_med):
    mo.md(rf"""
    ## 5 Including High and Medium confidence results

    Slightly higher scores, but much higher error rates

    ### 5.1 Direct Strategy
    {mo.as_html(plots_high_med[0])}

    ### 5.2 Iterative Strategy
    {mo.as_html(plots_high_med[1])}

    ### 5.3 Triplet Strategy
    {mo.as_html(plots_high_med[2])}
    """)
    return


if __name__ == "__main__":
    app.run()
