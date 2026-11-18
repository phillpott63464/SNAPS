import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    from pathlib import Path

    from modules.analysis import compare_results, plot_comparisons
    from modules.mars import get_mars_data

    return Path, compare_results, get_mars_data, plot_comparisons


@app.cell
def _(Path):
    ## test_mars_download

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
    return BMRB_IDS, BMRB_NAMES, DATA_DIR


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    compare_results,
    get_mars_data,
    plot_comparisons,
):
    new_mars_data = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    original_mars_data = get_mars_data(
        BMRB_IDS, BMRB_NAMES, DATA_DIR, use_defaults=True
    )
    comparison = compare_results(new_mars_data, original_mars_data)
    plots = plot_comparisons(comparison)
    return comparison, plots


@app.cell(hide_code=True)
def _(comparison, mo, plots):
    mo.md(
        rf"""
    # Test mars the software against mars the results from the paper

    Hypothesis: The software is 20 years old and has probably changed a little. Test whether the results are the same.

    Conclusions: Yeah it's changed a little

    Requirements: mars-podman on path, more details on that later

    {mo.as_html(plots['II'])}

    {comparison.to_markdown(index=False)}
    """
    )


if __name__ == '__main__':
    app.run()
