import marimo

__generated_with = '0.25.2'
app = marimo.App(width='medium')


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import contextlib
    import os
    from pathlib import Path

    import optuna
    from modules.analysis import compare_results, plot_comparisons_big
    from modules.bmrb import (
        run_test_set,
        with_psipred_dir,
    )
    from modules.mars import get_mars_data
    from modules.rbmrb import get_rbmrb
    from modules.results import assignment_table
    from modules.test_sets import run_tripletm2
    from optuna.visualization import plot_optimization_history
    from tqdm.auto import tqdm

    return (
        Path,
        assignment_table,
        compare_results,
        contextlib,
        get_mars_data,
        get_rbmrb,
        optuna,
        os,
        plot_comparisons_big,
        plot_optimization_history,
        run_test_set,
        run_tripletm2,
        tqdm,
        with_psipred_dir,
    )


@app.cell
def _(Path, get_rbmrb):
    DATA_DIR = Path(__file__).resolve().parent / 'mars_test_set'
    OUTPUT_DIR = Path(__file__).resolve().parent / 'mars_test_result'
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
    BMRB_IDS += get_rbmrb(DATA_DIR, 200)
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
    for key in BMRB_IDS:
        BMRB_NAMES.setdefault(key, str(key))
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
        default_test_files, BMRB_NAMES, 'default (1.0/0.2 all)'
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
    contextlib,
    default_results,
    optuna,
    os,
    plot_optimization_history,
    run_tripletm2,
    tqdm,
    with_psipred_dir,
):
    @with_psipred_dir
    def _(psipred_dir):
        MAX_ERROR_HIGH = 0.01
        MAX_ERROR_MEDIUM = 0.05
        MEDIUM_TIEBREAK = 0.001  # assumes both coverage metrics are in [0, 1]

        def objective(trial):
            cutoff_2 = trial.suggest_float('cutoff_2', 0, 10)
            cutoff_1 = trial.suggest_float('cutoff_1', 0, cutoff_2)

            files = run_tripletm2(
                bmrb_ids=BMRB_IDS,
                DATA_DIR=DATA_DIR,
                OUTPUT_DIR=OUTPUT_DIR / 'tripletm',
                cutoff_1=cutoff_1,
                cutoff_2=cutoff_2,
                psipred_dir=psipred_dir,
            )

            results_high = assignment_table(
                files, BMRB_NAMES, 'candidate', good_confidences=['High']
            )
            results_medium = assignment_table(
                files,
                BMRB_NAMES,
                'candidate',
                good_confidences=['High', 'Medium'],
            )

            comparison_high = compare_results(results_high, default_results)
            comparison_medium = compare_results(
                results_medium, default_results
            )

            error_high = comparison_high['err_rate_candidate'].mean()
            error_medium = comparison_medium['err_rate_candidate'].mean()

            coverage_high = comparison_high['reliable_frac_candidate'].mean()

            coverage_medium = comparison_medium[
                'reliable_frac_candidate'
            ].mean()

            trial.set_user_attr('error_high', error_high)
            trial.set_user_attr('error_medium', error_medium)
            trial.set_user_attr('coverage_high', coverage_high)
            trial.set_user_attr('coverage_medium', coverage_medium)
            trial.set_constraint(
                'error_high',
                error_high - MAX_ERROR_HIGH,
            )
            trial.set_constraint(
                'error_medium',
                error_medium - MAX_ERROR_MEDIUM,
            )

            # Optuna minimizes: maximize high coverage, with medium as a small tie-breaker.
            return -(coverage_high + MEDIUM_TIEBREAK * coverage_medium)

        def silent_objective(trial):
            with open(os.devnull, 'w') as f:
                with contextlib.redirect_stdout(f), contextlib.redirect_stderr(
                    f
                ):
                    try:
                        return objective(trial)
                    except:
                        return 5

        study = optuna.create_study(
            direction='minimize',
            study_name='Make tripletm2 as high coverage as possible whilst remaining within 1% (high) and 5% (medium) error rate',
            storage=f'sqlite:///{OUTPUT_DIR}/tripletm.db',
            load_if_exists=True,
        )

        n_trials = 300 - len(study.trials)

        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        with tqdm(total=n_trials, desc='Optuna') as pbar:

            def callback(study, trial):
                pbar.update(1)
                try:
                    best = study.best_trial
                except ValueError:  # No feasible trial yet
                    pbar.set_postfix(
                        best=f'Trial {trial.number} is complete but no feasible trial exists yet.'
                    )
                else:
                    string = f'Best params: {best.params}, best value: {best.value}, error_high={best.user_attrs.get("error_high")}, error_medium={best.user_attrs.get("error_medium")}, coverage_high={best.user_attrs.get("coverage_high")}, coverage_medium={best.user_attrs.get("coverage_medium")}'
                    print(string)

                fig = plot_optimization_history(study)
                fig.show()

            study.optimize(
                silent_objective,
                n_trials=n_trials,
                callbacks=[callback],
                show_progress_bar=False,
            )

        return study

    study = _()
    return (study,)


@app.cell
def _(study):
    print(study.best_params, study.best_value)


@app.cell
def _(DATA_DIR, get_rbmrb):
    BMRB_IDS_TEST = get_rbmrb(DATA_DIR, 200, 'test')
    BMRB_NAMES_TEST = {}

    for key_2 in BMRB_IDS_TEST:
        BMRB_NAMES_TEST.setdefault(key_2, str(key_2))

    print(len(BMRB_IDS_TEST))
    print(len(BMRB_NAMES_TEST))

    for name in BMRB_NAMES_TEST:
        assert name in BMRB_IDS_TEST

    for val in BMRB_IDS_TEST:
        assert name in BMRB_NAMES_TEST
    return BMRB_IDS_TEST, BMRB_NAMES_TEST


@app.cell
def _(
    BMRB_IDS_TEST,
    BMRB_NAMES_TEST,
    DATA_DIR,
    OUTPUT_DIR,
    assignment_table,
    compare_results,
    plot_comparisons_big,
    run_test_set,
    run_tripletm2,
    study,
):
    test_default_test_files = run_test_set(
        bmrb_ids=BMRB_IDS_TEST, DATA_DIR=DATA_DIR, OUTPUT_DIR=OUTPUT_DIR
    )
    test_default_results = assignment_table(
        test_default_test_files, BMRB_NAMES_TEST, 'default'
    )
    test_default_results = test_default_results[
        test_default_results['condition'] == 'triplet'
    ]

    files = run_tripletm2(
        bmrb_ids=BMRB_IDS_TEST,
        DATA_DIR=DATA_DIR,
        OUTPUT_DIR=OUTPUT_DIR / 'tripletm',
        cutoff_1=study.best_params['cutoff_1'],
        cutoff_2=study.best_params['cutoff_2'],
    )

    results_high = assignment_table(
        files, BMRB_NAMES_TEST, 'candidate', good_confidences=['High']
    )
    comparison_high = compare_results(results_high, test_default_results)
    plots_high = plot_comparisons_big(comparison_high)

    results_medium = assignment_table(
        files,
        BMRB_NAMES_TEST,
        'candidate',
        good_confidences=['High', 'Medium'],
    )
    comparison_medium = compare_results(results_medium, test_default_results)
    plots_medium = plot_comparisons_big(comparison_medium)
    return (
        comparison_high,
        comparison_medium,
        plots_high,
        plots_medium,
        results_medium,
    )


@app.cell
def _(mo, plots_medium):
    mo.vstack([mo.as_html(plot) for plot in plots_medium.values()])


@app.cell
def _(mo, plots_high):
    mo.vstack([mo.as_html(plot) for plot in plots_high.values()])


@app.cell
def _(comparison_high, comparison_medium):
    print(comparison_high['err_rate_candidate'].mean())
    print(comparison_high['err_rate_default'].mean())
    print(comparison_medium['err_rate_candidate'].mean())
    print(comparison_medium['err_rate_default'].mean())


@app.cell
def _(plot_optimization_history, study):
    fig = plot_optimization_history(study)
    fig.show()


@app.cell
def _(
    BMRB_IDS_TEST,
    BMRB_NAMES_TEST,
    DATA_DIR,
    compare_results,
    get_mars_data,
    plot_comparisons_big,
    results_medium,
):
    mars_results = get_mars_data(BMRB_IDS_TEST, BMRB_NAMES_TEST, DATA_DIR)

    mars_new_comparison = compare_results(results_medium, mars_results)

    mars_plots = {
        **plot_comparisons_big(mars_new_comparison),
    }
    return (mars_plots,)


@app.cell
def _(mars_plots, mo):
    mo.vstack([mo.as_html(plot) for plot in mars_plots.values()])


if __name__ == '__main__':
    app.run()
