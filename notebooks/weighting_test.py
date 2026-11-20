import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It's better if you remove sequential link weighting entirely

    Not sure why
    """)
    return


@app.cell
def _():
    from pathlib import Path

    import matplotlib.pyplot as plt
    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set, run_weighting_test
    from modules.mars import get_mars_data
    from modules.results import assignment_table

    return (
        Path,
        assignment_table,
        compare_results,
        get_mars_data,
        plot_comparisons,
        plt,
        run_test_set,
        run_weighting_test,
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
    run_test_set,
    run_weighting_test,
):
    weighting_test_files_10 = run_weighting_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        weighting_range=range(0, 11, 1),
    )

    default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )

    weighting_results_10 = assignment_table(
        weighting_test_files_10,
        BMRB_NAMES,
        'weighting',
    )
    default_results = assignment_table(
        default_test_files, BMRB_NAMES, 'default (1.0)'
    )
    default_results = default_results[
        default_results['condition'] == 'triplet'
    ]
    return default_results, weighting_results_10


@app.cell
def _(
    compare_results,
    default_results,
    plot_comparisons,
    weighting_results_10,
):
    comparison_10 = compare_results(weighting_results_10, default_results)

    plots_10 = plot_comparisons(comparison_10)
    return comparison_10, plots_10


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test weighting

    0 - 10 step 1
    """)
    return


@app.cell
def _(mo, plots_10):
    mo.vstack([mo.as_html(plot) for plot in plots_10.values()])
    return


@app.cell
def _(comparison_10, make_weighting_graphs, mo):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_weighting_graphs(comparison_10).values()
        ]
    )
    return


@app.cell
def _(plt):
    # weighting_results = a pandas dataframe

    def make_weighting_graphs(comparison):
        graph_vals = {'error': {}, 'reliable': {}, 'correct': {}}

        for condition in comparison['condition'].unique():
            # condition == float
            weighting_slice = comparison[
                comparison['condition'] == condition
            ]   # Only the condition we want
            error_average = weighting_slice['err_rate_weighting'].mean()
            reliable_average = weighting_slice[
                'reliable_frac_weighting'
            ].mean()
            correct_average = weighting_slice['correct_frac_weighting'].mean()

            graph_vals['error'][condition] = error_average
            graph_vals['reliable'][condition] = reliable_average
            graph_vals['correct'][condition] = correct_average

        weighting_graphs = {}

        for graph, vals in graph_vals.items():
            # Sort conditions so the curve is drawn in order
            conditions = sorted(vals)
            averages = [vals[condition] for condition in conditions]

            fig, ax = plt.subplots()
            ax.scatter(conditions, averages)
            ax.plot(conditions, averages)

            ax.set_xlabel(
                'Weighting of shifts vs sequential links (0 is no sequential links, 1 is default)'
            )
            ax.set_ylabel(f'Mean {graph} fraction')
            ax.set_title(f'{graph.capitalize()} vs. sequential link weight')
            ax.grid(True, alpha=0.3)

            weighting_graphs[graph] = fig
        return weighting_graphs

    return (make_weighting_graphs,)


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    default_results,
    plot_comparisons,
    run_weighting_test,
):
    weighting_test_files_1 = run_weighting_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        weighting_range=[x / 10 for x in range(0, 11, 1)],
    )
    weighting_results_1 = assignment_table(
        weighting_test_files_1,
        BMRB_NAMES,
        'weighting',
    )
    comparison_1 = compare_results(weighting_results_1, default_results)
    plots_1 = plot_comparisons(comparison_1)
    return comparison_1, weighting_results_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test weighting

    0 - 1 step 0.1
    """)
    return


@app.cell
def _(mo, plots_10):
    mo.vstack([mo.as_html(plot) for plot in plots_10.values()])
    return


@app.cell
def _(comparison_1, make_weighting_graphs, mo):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_weighting_graphs(comparison_1).values()
        ]
    )
    return


@app.cell
def _(comparison_10):
    comparison_10.to_clipboard()
    return


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    compare_results,
    get_mars_data,
    plot_comparisons,
    weighting_results_1,
):
    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    mars_comparison = compare_results(weighting_results_1, mars_results)
    mars_plots = plot_comparisons(mars_comparison)
    return (mars_plots,)


@app.cell
def _(mars_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in mars_plots.values()])
    return


if __name__ == "__main__":
    app.run()
