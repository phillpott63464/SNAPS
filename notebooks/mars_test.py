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

    from modules.analysis import compare_results, plot_comparisons, analyse_mismatches
    from modules.bmrb import run_test_set
    from modules.results import assignment_table, all_mismatches
    # from modules.pdb import colour_mars_set
    return (
        Path,
        all_mismatches,
        analyse_mismatches,
        assignment_table,
        compare_results,
        mo,
        plot_comparisons,
        run_test_set,
    )


@app.cell
def _(Path):
    BMRB_IDS = [ 5471, 4354, 4384, 4022, 4106, 4457, 4402, 4341, 547, 4082, 4136, 5387 ]
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
        5387: "Human ubiquitin",
    }
    DATA_DIR = Path(__file__).resolve().parent / "mars_test_set"
    OUTPUT_DIR = Path(__file__).resolve().parent / "mars_test_result"
    STRATEGIES = {
        "direct": "{bmrb_id}-test-result-direct.txt",
        "iterative": "{bmrb_id}-test-result-iterative.txt",
        "triplet": "{bmrb_id}-test-result-triplet.txt",
    }
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


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
    snaps_test_files = run_test_set(bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR)
    return (snaps_test_files,)


@app.cell
def _(BMRB_NAMES, assignment_table, snaps_test_files):
    snaps_results = assignment_table(snaps_test_files, BMRB_NAMES, 'snaps')
    return (snaps_results,)


@app.cell
def _(BMRB_IDS, BMRB_NAMES, DATA_DIR):
    # Published MARS counts used in the original benchmark.

    from modules.mars import get_mars_data

    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    return (mars_results,)


@app.cell
def _(mars_results, mo, snaps_results):
    mo.md("## 2. Assignment counts\n\n### SNAPS\n\n" + snaps_results.to_markdown(index=False)
          + "\n\n### MARS\n\n" + mars_results.to_markdown(index=False))
    return


@app.cell
def _(compare_results, mars_results, snaps_results):
    print(snaps_results, mars_results)

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
    {mo.as_html(plots['direct'])}

    ### 4.2 Iterative Strategy
    {mo.as_html(plots['iterative'])}

    ### 4.3 Triplet Strategy
    {mo.as_html(plots['triplet'])}
    """)
    return


@app.cell
def _(comparison, plot_comparisons):
    plots = plot_comparisons(comparison)
    return (plots,)


@app.cell
def _(
    BMRB_NAMES,
    assignment_table,
    compare_results,
    plot_comparisons,
    snaps_results,
    snaps_test_files,
):
    snaps_results_high_med = assignment_table(snaps_test_files, BMRB_NAMES, 'snaps_high_med', good_confidences=['High', 'Medium'])
    comparison_high_med = compare_results(snaps_results_high_med, snaps_results)
    plots_high_med = plot_comparisons(comparison_high_med)
    return (plots_high_med,)


@app.cell(hide_code=True)
def _(mo, plots_high_med):
    mo.md(rf"""
    ## 5 Including High and Medium confidence results

    Slightly higher scores, but much higher error rates

    ### 5.1 Direct Strategy
    {mo.as_html(plots_high_med['direct'])}

    ### 5.2 Iterative Strategy
    {mo.as_html(plots_high_med['iterative'])}

    ### 5.3 Triplet Strategy
    {mo.as_html(plots_high_med['triplet'])}
    """)
    return


@app.cell
def _(DATA_DIR, OUTPUT_DIR, all_mismatches, snaps_test_files):
    mismatches = all_mismatches(snaps_test_files, DATA_DIR, OUTPUT_DIR)
    return (mismatches,)


@app.cell
def _(BMRB_NAMES, mismatches, mo):
    def _():
        condition_blocks = []

        for condition, protein_results in mismatches.items():
            no_error_proteins = []
            protein_tables = []

            for bmrb_id, df in protein_results.items():
                protein = BMRB_NAMES[bmrb_id]

                if df.empty:
                    no_error_proteins.append(protein)
                else:
                    protein_tables.extend(
                        [
                            mo.md(f"#### {protein} · {len(df)} rows"),
                            mo.ui.table(df),
                        ]
                    )

            content = []
            if no_error_proteins:
                names = ", ".join(no_error_proteins)
                content.append(mo.md(f"**{names} had no errors.**"))

            content.extend(protein_tables)

            condition_blocks.append(
                mo.vstack(
                    [
                        mo.md(f"## {condition}"),
                        mo.vstack(content) if content else mo.md("No results."),
                    ]
                )
            )
        return mo.vstack(condition_blocks)


    _()
    return


@app.cell
def _(analyse_mismatches, mismatches):
    mismatch_figs = analyse_mismatches(mismatches)
    return (mismatch_figs,)


@app.cell(hide_code=True)
def _(mismatch_figs, mo):
    mo.md(rf"""
    ## 6 Mismatch Figures

    ### 6.1 Direct Strategy
    {mo.as_html(mismatch_figs['direct'])}

    ### 6.2 Iterative Strategy
    {mo.as_html(mismatch_figs['iterative'])}

    ### 6.3 Triplet Strategy
    {mo.as_html(mismatch_figs['triplet'])}
    """)
    return


@app.cell
def _(BMRB_IDS, DATA_DIR, OUTPUT_DIR, colour_mars_set):
    colour_mars_set(BMRB_IDS, DATA_DIR, OUTPUT_DIR)
    return


if __name__ == "__main__":
    app.run()
