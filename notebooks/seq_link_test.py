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
    from modules.bmrb import run_seq_link_test, run_test_set
    from modules.results import assignment_table

    return (
        Path,
        assignment_table,
        compare_results,
        plot_comparisons,
        plt,
        run_seq_link_test,
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
    run_seq_link_test,
    run_test_set,
):
    seq_link_test_files = run_seq_link_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        seq_range=[x / 100 for x in range(1, 11, 1)]
        + [x / 10 for x in range(1, 11, 1)],
    )

    default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )

    seq_link_test_result = assignment_table(
        seq_link_test_files,
        BMRB_NAMES,
        'sequential link threshold',
    )

    default_results = assignment_table(
        default_test_files, BMRB_NAMES, 'default (0.2)'
    )
    default_results = default_results[
        default_results['condition'] == 'triplet'
    ]

    comparison = compare_results(seq_link_test_result, default_results)

    plots = plot_comparisons(comparison)
    return comparison, default_results, plots


@app.cell
def _(mo, plots):
    mo.vstack([mo.as_html(plot) for plot in plots.values()][:10])


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
            error_average = weighting_slice[
                'err_rate_sequential link threshold'
            ].mean()
            reliable_average = weighting_slice[
                'reliable_frac_sequential link threshold'
            ].mean()
            correct_average = weighting_slice[
                'correct_frac_sequential link threshold'
            ].mean()

            graph_vals['error'][condition] = error_average
            graph_vals['reliable'][condition] = reliable_average
            graph_vals['correct'][condition] = correct_average

        weighting_graphs = {}

        for graph, vals in graph_vals.items():
            # Sort conditions so the curve is drawn in order
            conditions = sorted(vals)
            averages = [vals[condition] for condition in conditions]
            conditions = sorted([float(x) for x in vals.keys()])

            fig, ax = plt.subplots()
            ax.scatter(conditions, averages)
            ax.plot(conditions, averages)

            ax.set_xlabel('sequential link value')
            plt.xticks(rotation=45)
            ax.set_ylabel(f'Mean {graph} fraction')
            ax.set_title(f'{graph.capitalize()} vs. sequential link value')

            step = max(1, len(conditions) // 15)
            ax.set_xticks(conditions[::step])
            ax.grid(True, alpha=0.3)

            weighting_graphs[graph] = fig
        return weighting_graphs

    return (make_weighting_graphs,)


@app.cell
def _(comparison, make_weighting_graphs, mo):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_weighting_graphs(comparison).values()
        ]
    )


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
    run_seq_link_test,
):
    weighted_seq_link_test_files = run_seq_link_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        seq_range=[x / 100 for x in range(1, 11, 1)]
        + [x / 10 for x in range(1, 11, 1)],
        weight=0,
    )

    weighted_seq_link_test_result = assignment_table(
        weighted_seq_link_test_files,
        BMRB_NAMES,
        'sequential link threshold',
    )

    weighted_comparison = compare_results(
        weighted_seq_link_test_result, default_results
    )

    weighted_plots = plot_comparisons(weighted_comparison)
    return weighted_comparison, weighted_plots


@app.cell
def _(mo, weighted_plots):
    mo.vstack([mo.as_html(plot) for plot in weighted_plots.values()][:10])


@app.cell
def _(mo, weighted_plots):
    mo.vstack([mo.as_html(plot) for plot in weighted_plots.values()][10:])


@app.cell
def _(make_weighting_graphs, mo, weighted_comparison):
    mo.vstack(
        [
            mo.as_html(plot)
            for plot in make_weighting_graphs(weighted_comparison).values()
        ]
    )


if __name__ == '__main__':
    app.run()
