from pathlib import Path
import sys
from enum import Enum
from typing import Dict, List, Optional

import click
import typer

from refactor.SNAPSconfig import SNAPSConfig, build_config


DEFAULT_CONFIG = build_config()


def _display_default(value: object, none_label: Optional[str] = None) -> str:
    """Render a config default for CLI help."""
    if value is None and none_label is not None:
        return none_label
    return str(value)


app = typer.Typer(
    help="SNAPS (Simple NMR Assignments from Predicted Shifts)",
    no_args_is_help=True,
)


class Strategy(str, Enum):
    direct = "direct"
    iterative = "iterative"


def _show_advanced_help(
    ctx: click.Context, param: click.Parameter, value: bool
) -> None:
    """Show the full option reference without cluttering normal help."""
    if value:
        for option in ctx.command.params:
            if getattr(option, "rich_help_panel", None) == "Advanced":
                option.hidden = False
        typer.echo(ctx.get_help())
        ctx.exit()


def run_pipeline(config: SNAPSConfig, verbosity: int = 0):
    """Load the pipeline lazily so CLI help works without optional readers."""
    from refactor.SNAPSpipeline import run_pipeline as pipeline_run

    return pipeline_run(config, verbosity)


def _parse_atom_sd(values: Optional[List[str]]) -> Optional[Dict[str, float]]:
    """Convert repeated ``ATOM=SD`` options to the config mapping."""
    if values is None:
        return None

    result: Dict[str, float] = {}
    for value in values:
        atom, separator, standard_deviation = value.partition("=")
        if not separator or not atom or not standard_deviation:
            raise typer.BadParameter(
                "atom standard deviations must use ATOM=VALUE (for example H=0.1726)"
            )
        try:
            result[atom] = float(standard_deviation)
        except ValueError as error:
            raise typer.BadParameter(
                f"standard deviation for {atom!r} must be a number"
            ) from error
    return result


@app.command()
def main(
    shift_file: Optional[Path] = typer.Argument(
        None, help="Observed shifts file (or a NEF file containing both lists)."
    ),
    pred_file: Optional[Path] = typer.Argument(
        None, help="Predicted shifts file."
    ),
    output_file: Optional[Path] = typer.Argument(
        None, help="Assignment output file (default: -)."
    ),
    shift_file_option: Optional[Path] = typer.Option(
        None,
        "--shift_file",
        "--shift-file",
        help="Observed shifts file.",
        rich_help_panel="Common",
    ),
    pred_file_option: Optional[Path] = typer.Option(
        None,
        "--pred_file",
        "--pred-file",
        help="Predicted shifts file.",
        rich_help_panel="Common",
    ),
    output_file_option: Path = typer.Option(
        Path("-"),
        "--output_file",
        "--output-file",
        help="Assignment output file (default: -).",
        rich_help_panel="Common",
    ),
    config_file: Optional[Path] = typer.Option(
        None,
        "--config_file",
        "--config-file",
        "-c",
        help="YAML configuration file. CLI options override file values.",
        rich_help_panel="Common",
    ),
    strategy: Optional[Strategy] = typer.Option(
        None,
        "--strategy",
        help="Assignment strategy.",
        show_default=str(DEFAULT_CONFIG.strategy),
        rich_help_panel="Common",
    ),
    obs_format: Optional[str] = typer.Option(
        None,
        "--obs_format",
        "--obs-format",
        help="Observed shifts format.",
        show_default=_display_default(DEFAULT_CONFIG.obs_format, "autodetected"),
        rich_help_panel="Common",
    ),
    pred_format: Optional[str] = typer.Option(
        None,
        "--pred_format",
        "--pred-format",
        help="Predicted shifts format.",
        show_default=_display_default(DEFAULT_CONFIG.pred_format, "autodetected"),
        rich_help_panel="Common",
    ),
    obs_frame: Optional[str] = typer.Option(
        None, "--obs_frame", "--obs-frame", help="Observed shifts frame.", rich_help_panel="Common"
    ),
    pred_frame: Optional[str] = typer.Option(
        None, "--pred_frame", "--pred-frame", help="Predicted shifts frame.", rich_help_panel="Common"
    ),
    advanced: bool = typer.Option(
        False,
        "--advanced",
        help="Show advanced options and exit.",
        is_eager=True,
        callback=_show_advanced_help,
    ),
    obs_chain: Optional[str] = typer.Option(
        None,
        "--obs_chain",
        "--obs-chain",
        help="Observed chain identifier.",
        show_default=str(DEFAULT_CONFIG.obs_chain),
        rich_help_panel="Common",
    ),
    pred_chain: Optional[str] = typer.Option(
        None,
        "--pred_chain",
        "--pred-chain",
        help="Predicted chain identifier.",
        show_default=str(DEFAULT_CONFIG.pred_chain),
        rich_help_panel="Common",
    ),
    aa_restraints_file: Optional[Path] = typer.Option(
        None,
        "--aa_restraints_file",
        "--aa-restraints-file",
        help="Amino acid restraints file.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    aa_restraints_frame: Optional[str] = typer.Option(
        None,
        "--aa_restraints_frame",
        "--aa-restraints-frame",
        help="Amino acid restraints frame.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    output_type: Optional[str] = typer.Option(
        None,
        "--output_type",
        "--output-type",
        help="Assignment output format.",
        show_default=str(DEFAULT_CONFIG.output_type),
        rich_help_panel="Common",
    ),
    shift_output_file: Optional[Path] = typer.Option(
        None, "--shift_output_file", "--shift-output-file", help="Chemical shift output file.", rich_help_panel="Advanced", hidden=True
    ),
    shift_output_type: Optional[str] = typer.Option(
        None, "--shift_output_type", "--shift-output-type", help="Chemical shift output format.", rich_help_panel="Advanced", hidden=True
    ),
    shift_output_confidence: Optional[str] = typer.Option(
        None,
        "--shift_output_confidence",
        "--shift-output-confidence",
        help="Chemical shift output confidence.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    atom_set: Optional[List[str]] = typer.Option(
        None,
        "--atom_set",
        "--atom-set",
        help="Atom type to use; repeat for multiple atom types.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    atom_sd: Optional[List[str]] = typer.Option(
        None,
        "--atom_sd",
        "--atom-sd",
        help="Atom standard deviation as ATOM=VALUE; repeatable.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    supported_atom_types: Optional[List[str]] = typer.Option(
        None,
        "--supported_atom_types",
        "--supported-atom-types",
        help="Accepted atom type; repeat for multiple atom types.",
        rich_help_panel="Advanced",
        hidden=True,
    ),
    default_probability: Optional[float] = typer.Option(
        None,
        "--default_probability",
        "--default-probability",
        help="Probability assigned to a missing shift.",
        show_default=str(DEFAULT_CONFIG.default_probability),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    shift_scale_factor: Optional[float] = typer.Option(
        None,
        "--shift_scale_factor",
        "--shift-scale-factor",
        help="Scale factor for atom standard deviations.",
        show_default=str(DEFAULT_CONFIG.shift_scale_factor),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    pred_offset: Optional[int] = typer.Option(
        None,
        "--pred_offset",
        "--pred-offset",
        help="Offset applied to predicted residue numbers.",
        show_default=str(DEFAULT_CONFIG.pred_offset),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    maximise_assignments: Optional[bool] = typer.Option(
        None,
        "--maximise_assignments/--minimise_assignments",
        "--maximise-assignments/--minimise-assignments",
        help="Maximise assignment scores rather than minimise them.",
        show_default=str(DEFAULT_CONFIG.maximise_assignments),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    dummy_predicted_prefix: Optional[str] = typer.Option(
        None,
        "--dummy_predicted_prefix",
        "--dummy-predicted-prefix",
        help="Prefix for dummy predicted residues.",
        show_default=str(DEFAULT_CONFIG.dummy_predicted_prefix),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    dummy_observed_prefix: Optional[str] = typer.Option(
        None,
        "--dummy_observed_prefix",
        "--dummy-observed-prefix",
        help="Prefix for dummy observed spin systems.",
        show_default=str(DEFAULT_CONFIG.dummy_observed_prefix),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    seq_link_threshold: Optional[float] = typer.Option(
        None,
        "--seq_link_threshold",
        "--seq-link-threshold",
        help="Carbon shift threshold for sequential links.",
        show_default=str(DEFAULT_CONFIG.seq_link_threshold),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    iterate_until_consistent: Optional[bool] = typer.Option(
        None,
        "--iterate_until_consistent/--no-iterate_until_consistent",
        "--iterate-until-consistent/--no-iterate-until-consistent",
        help="Iterate assignments until sequentially consistent.",
        show_default=str(DEFAULT_CONFIG.iterate_until_consistent),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    use_ss_class_info: Optional[bool] = typer.Option(
        None,
        "--use_ss_class_info/--no-use_ss_class_info",
        "--use-ss-class-info/--no-use-ss-class-info",
        help="Use amino acid class restraints.",
        show_default=str(DEFAULT_CONFIG.use_ss_class_info),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    delta_correlation: Optional[bool] = typer.Option(
        None,
        "--delta_correlation/--no-delta_correlation",
        "--delta-correlation/--no-delta-correlation",
        help="Account for correlations in prediction errors.",
        show_default=str(DEFAULT_CONFIG.delta_correlation),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    pred_correction: Optional[bool] = typer.Option(
        None,
        "--pred_correction/--no_pred_correction",
        "--pred-correction/--no-pred-correction",
        help="Use prediction correction.",
        show_default=str(DEFAULT_CONFIG.pred_correction),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    pred_correction_file: Optional[str] = typer.Option(
        None,
        "--pred_correction_file",
        "--pred-correction-file",
        help="Linear prediction correction parameters.",
        show_default=str(DEFAULT_CONFIG.pred_correction_file),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    delta_correlation_mean_corrected_file: Optional[str] = typer.Option(
        None,
        "--delta_correlation_mean_corrected_file",
        "--delta-correlation-mean-corrected-file",
        help="Corrected prediction error means.",
        show_default=str(DEFAULT_CONFIG.delta_correlation_mean_corrected_file),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    delta_correlation_cov_corrected_file: Optional[str] = typer.Option(
        None,
        "--delta_correlation_cov_corrected_file",
        "--delta-correlation-cov-corrected-file",
        help="Corrected prediction error covariances.",
        show_default=str(DEFAULT_CONFIG.delta_correlation_cov_corrected_file),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    delta_correlation_mean_file: Optional[str] = typer.Option(
        None,
        "--delta_correlation_mean_file",
        "--delta-correlation-mean-file",
        help="Prediction error means.",
        show_default=str(DEFAULT_CONFIG.delta_correlation_mean_file),
        rich_help_panel="Advanced",
        hidden=True,
    ),
    delta_correlation_cov_file: Optional[str] = typer.Option(
        None,
        "--delta_correlation_cov_file",
        "--delta-correlation-cov-file",
        help="Prediction error covariances.",
        show_default=str(DEFAULT_CONFIG.delta_correlation_cov_file),
        rich_help_panel="Advanced",
        hidden=True,
    ),
) -> None:
    """Run the SNAPS assignment pipeline."""
    shift_file = shift_file_option or shift_file
    pred_file = pred_file_option or pred_file
    if output_file is None:
        output_file = output_file_option
    elif output_file_option != Path("-"):
        output_file = output_file_option

    if shift_file is None:
        raise typer.BadParameter(
            "a shift_file must be provided as a positional argument or with --shift_file"
        )
    if pred_file is None:
        if shift_file.suffix.lower() == ".nef":
            pred_file = shift_file
        else:
            raise typer.BadParameter(
                "a pred_file must be provided unless shift_file is a NEF file"
            )

    overrides = {
        "obs_file": shift_file,
        "pred_file": pred_file,
        "obs_format": obs_format,
        "pred_format": pred_format,
        "obs_frame": obs_frame,
        "pred_frame": pred_frame,
        "obs_chain": obs_chain,
        "pred_chain": pred_chain,
        "aa_restraints_file": aa_restraints_file,
        "aa_restraints_frame": aa_restraints_frame,
        "output_file": output_file,
        "output_type": output_type,
        "shift_output_file": shift_output_file,
        "shift_output_type": shift_output_type,
        "shift_output_confidence": shift_output_confidence,
        "atom_set": atom_set,
        "atom_sd": _parse_atom_sd(atom_sd),
        "supported_atom_types": supported_atom_types,
        "default_probability": default_probability,
        "shift_scale_factor": shift_scale_factor,
        "pred_offset": pred_offset,
        "maximise_assignments": maximise_assignments,
        "dummy_predicted_prefix": dummy_predicted_prefix,
        "dummy_observed_prefix": dummy_observed_prefix,
        "seq_link_threshold": seq_link_threshold,
        "iterate_until_consistent": iterate_until_consistent,
        "use_ss_class_info": use_ss_class_info,
        "delta_correlation": delta_correlation,
        "pred_correction": pred_correction,
        "pred_correction_file": pred_correction_file,
        "delta_correlation_mean_corrected_file": delta_correlation_mean_corrected_file,
        "delta_correlation_cov_corrected_file": delta_correlation_cov_corrected_file,
        "delta_correlation_mean_file": delta_correlation_mean_file,
        "delta_correlation_cov_file": delta_correlation_cov_file,
        "strategy": strategy,
    }
    config = build_config(config_file=config_file, cli_overrides=overrides)
    run_pipeline(config)

def cli() -> None:
    """Run the CLI, showing help when no arguments are supplied."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    cli()