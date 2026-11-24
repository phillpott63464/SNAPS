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

    import optuna
    from modules.analysis import compare_results, plot_comparisons
    from modules.bmrb import run_seq_link_test, run_test_set
    from modules.mars import get_mars_data
    from modules.results import assignment_table
    from optuna.visualization import plot_optimization_history

    return (
        Path,
        assignment_table,
        compare_results,
        get_mars_data,
        optuna,
        plot_comparisons,
        plot_optimization_history,
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
    run_test_set,
):
    default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    default_results = assignment_table(
        default_test_files, BMRB_NAMES, 'default (1.0)'
    )
    default_results = default_results[
        default_results['condition'] == 'triplet'
    ]
    return (default_results,)


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    default_results,
    run_seq_link_test,
):
    def evaluate(threshold, weight):
        files = run_seq_link_test(
            bmrb_ids=BMRB_IDS,
            DATA_DIR=DATA_DIR,
            OUTPUT_DIR=OUTPUT_DIR,
            seq_range=[threshold],
            weight=weight,
        )
        results = assignment_table(files, BMRB_NAMES, 'candidate')
        comparison = compare_results(results, default_results)
        return comparison['err_rate_candidate'].mean()

    return (evaluate,)


@app.cell
def _(OUTPUT_DIR, evaluate, optuna):
    def objective(trial):
        threshold = trial.suggest_float('threshold', 0.01, 0.3)
        weight = trial.suggest_float('weight', 1.0, 5.0)
        return evaluate(threshold, weight)

    study = optuna.create_study(
        direction='minimize',
        study_name='Sequential link and weighting surface',
        storage=f'sqlite:///{OUTPUT_DIR}/optuna.db',
        load_if_exists=True,
    )

    current_trials = len(study.trials)

    study.optimize(objective, n_trials=50 - current_trials)

    print(study.best_params, study.best_value)
    return (study,)


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
    study,
):
    files = run_seq_link_test(
        bmrb_ids=BMRB_IDS,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR,
        seq_range=[round(study.best_params['threshold'], 3)],
        weight=round(study.best_params['weight'], 3),
    )

    results = assignment_table(files, BMRB_NAMES, 'candidate')
    comparison = compare_results(results, default_results)
    plots = plot_comparisons(comparison)
    return plots, results


@app.cell
def _(mo, plots):
    mo.vstack([mo.as_html(plot) for plot in plots.values()])


@app.cell
def _(
    BMRB_IDS,
    BMRB_NAMES,
    DATA_DIR,
    compare_results,
    default_results,
    get_mars_data,
    plot_comparisons,
    results,
):
    mars_results = get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR)
    mars_default_comparison = compare_results(default_results, mars_results)

    mars_new_comparison = compare_results(results, mars_results)

    mars_plots = {
        **plot_comparisons(mars_default_comparison),
        **plot_comparisons(mars_new_comparison),
    }
    return (mars_plots,)


@app.cell
def _(mars_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in mars_plots.values()])


@app.cell
def _(plot_optimization_history, study):
    fig = plot_optimization_history(study)
    fig.show()


if __name__ == '__main__':
    app.run()
