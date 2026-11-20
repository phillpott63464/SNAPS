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

    import matplotlib.pyplot as plt
    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_test_set, run_weighting_test
    from modules.results import assignment_table

    return (
        Path,
        assignment_table,
        compare_results,
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
    compare_results,
    plot_comparisons,
    run_test_set,
    run_weighting_test,
):
    weighting_test_files = run_weighting_test(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )

    weighting_results = assignment_table(
        weighting_test_files, BMRB_NAMES, 'weighting'
    )
    default_results = assignment_table(
        default_test_files, BMRB_NAMES, 'default'
    )
    default_results = default_results[
        default_results['condition'] == 'triplet'
    ]

    comparison = compare_results(weighting_results, default_results)

    print(comparison)

    plots = plot_comparisons(comparison)
    return plots, weighting_results


@app.cell
def _(mo, plots):
    mo.vstack([mo.as_html(plot) for plot in plots.values()])


@app.cell
def _(plt, weighting_results):
    # weighting_results = a pandas dataframe

    graph_vals = {'error': {}, 'reliable': {}, 'correct': {}}

    for condition in weighting_results['condition'].unique():
        # condition == float
        weighting_slice = weighting_results[
            weighting_results['condition'] == condition
        ]   # Only the condition we want
        error_average = weighting_slice['error_frac'].mean()
        reliable_average = weighting_slice['reliable_frac'].mean()
        correct_average = weighting_slice['correct_frac'].mean()

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

        ax.set_xlabel('Condition')
        ax.set_ylabel(f'Mean {graph} fraction')
        ax.set_title(f'{graph.capitalize()} vs. condition')
        ax.grid(True, alpha=0.3)

        weighting_graphs[graph] = fig
    return (weighting_graphs,)


@app.cell
def _(mo, weighting_graphs):
    mo.vstack([mo.as_html(plot) for plot in weighting_graphs.values()])


if __name__ == '__main__':
    app.run()
