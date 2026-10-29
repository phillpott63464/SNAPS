from refactor.dataclasses import AnalysisResult
from refactor.SNAPSconfig import SNAPSConfig
from refactor.SNAPSimporter import SNAPSImporter
from refactor.SNAPSoutput import create_output
from refactor.SNAPSstrategy import create_strategy


def run_pipeline(
    config: SNAPSConfig,
    verbosity: int = 0,
) -> AnalysisResult:
    """
    Main pipeline - can be called from CLI or web service.

    Args:
        config: SNAPS configuration
        verbosity: Logging level (0=quiet, 1=info, 2=debug)

    Returns:
        Analysis results with assignments and metrics
    """

    if config.obs_file is None or config.pred_file is None:
        raise ValueError(
            "obs_file and pred_file must be configured before running the pipeline"
        )
    if config.output_file is None:
        raise ValueError("output_file must be configured before running the pipeline")

    # Setup logging
    config.logger.info("Starting SNAPS pipeline")

    # Stage 1: Import
    config.logger.info("Importing data...")
    importer = SNAPSImporter(config=config)
    data = importer.import_data(
        obs_file=config.obs_file,
        pred_file=config.pred_file,
        obs_format=config.obs_format,
        pred_format=config.pred_format,
        obs_frame=config.obs_frame,
        pred_frame=config.pred_frame,
        obs_chain=config.obs_chain,
        pred_chain=config.pred_chain,
        aa_restraints_file=config.aa_restraints_file,
        aa_restraints_frame=config.aa_restraints_frame,
    )
    config.logger.info(f"  {len(data.obs_shifts)} observed spin systems")
    config.logger.info(f"  {len(data.pred_shifts)} predicted residues")

    # Stage 2: Assignment
    config.logger.info(f"Running {config.strategy} assignment...")
    strategy = create_strategy(config.strategy, config)
    assignment_result = strategy.assign(data)
    config.logger.info("  Assignments complete")

    # Stage 3: Analysis
    # This was not in the original and thus is not part of the refactor. Revisit when the refactor is done.
    # logger.info("Analyzing results...")
    # analyzers = [
    #     ConsistencyAnalyzer(threshold=config.seq_link_threshold),
    #     # QualityAnalyzer(),
    # ]
    # analysis_pipeline = AnalysisPipeline(analyzers)
    # analysis_result = analysis_pipeline.run(assignment_result, data)
    # logger.info(f"  Quality: {analysis_result.quality_metrics}")

    # Stage 4: Output
    # Stage 5: Plots (optional)
    # if output_spec.hsqc_plot or output_spec.strip_plot:
    #     logger.info("Creating plots...")
    #     plot_manager = PlotManager()
    #
    #     if output_spec.hsqc_plot:
    #         plot_manager.create_plot('hsqc', analysis_result, data, output_spec.hsqc_plot)
    #         logger.info(f"  HSQC → {output_spec.hsqc_plot}")
    #
    #     if output_spec.strip_plot:
    #         plot_manager.create_plot('strip', analysis_result, data, output_spec.strip_plot)
    #         logger.info(f"  Strips → {output_spec.strip_plot}")
    output = create_output(config.output_type, config)
    output.output(assignment_result, config)
    output.plot_graphs(assignment_result=assignment_result, data=data)

    config.logger.info("Pipeline complete")
    return assignment_result
