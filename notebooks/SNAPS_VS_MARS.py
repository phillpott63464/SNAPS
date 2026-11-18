"""Reproduce the MARS comparison on the public SNAPS test set.

The notebook deliberately keeps orchestration and presentation here. Parsers,
downloaders, statistics, and plotting live in ``notebooks.modules`` so they can
be reused by other analyses.
"""

import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set
    from modules.results import assignment_table

    # from modules.pdb import colour_mars_set
    return (
        Path,
        assignment_table,
        compare_results,
        mo,
        plot_comparisons,
        run_test_set,
    )


@app.cell
def _(Path):
    BMRB_IDS = [
        5471,
        4354,
        4384,
        4022,
        4106,
        4457,
        4402,
        4341,
        547,
        4082,
        4136,
        5387,
    ]
    BMRB_NAMES = {
        5471: 'Malate synthase G',
        4354: 'Maltose binding protein',
        4384: 'Rous Sarcoma Virus capsid',
        4022: 'Human carbonic anhydrase I',
        4106: 'N-terminal domain of enzyme I (EIN)',
        4457: 'E-cadherin domains II and III',
        4402: 'Human prion protein',
        4341: 'Superoxide dismutase',
        547: 'Calmodulin/M13',
        4082: 'Profilin',
        4136: 'E. coli EmrE',
        5387: 'Human ubiquitin',
    }
    DATA_DIR = Path(__file__).resolve().parent / 'mars_test_set'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'mars_test_result'
    return BMRB_IDS, BMRB_NAMES, DATA_DIR, OUTPUT_DIR


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    plot_comparisons,
    run_test_set,
):
    # Published MARS counts used in the original benchmark.

    from modules.mars import get_mars_data

    snaps_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    snaps_results = assignment_table(snaps_test_files, BMRB_NAMES, 'snaps')
    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    comparison = compare_results(snaps_results, mars_results)
    plots = plot_comparisons(comparison)
    return comparison, plots


@app.cell(hide_code=True)
def _(comparison, mo, plots):
    mo.md(
        rf"""
    ## SNAPS vs MARS

    MARS is a fairly good and well established automated assignment program. Let's try to get pretty close to it.

    Requirements: mars-podman on path (more on that later)

    ### Direct Strategy
    {mo.as_html(plots['direct'])}

    ### Iterative Strategy
    {mo.as_html(plots['iterative'])}

    ### Triplet Strategy
    {mo.as_html(plots['triplet'])}

    {comparison.to_markdown(index=False)}
    """
    )


@app.cell
def _():
    return


if __name__ == '__main__':
    app.run()
