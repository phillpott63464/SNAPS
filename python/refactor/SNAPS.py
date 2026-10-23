from pathlib import Path
import inspect
import sys
from typing import Any, Dict, Optional, Set

import click
import typer

from refactor.SNAPSconfig import SNAPSConfig, build_config
from refactor.cli_adapter import convert_model_value, signature_from_model


app = typer.Typer(
    help="SNAPS (Simple NMR Assignments from Predicted Shifts)",
    no_args_is_help=True,
)

COMMON_OPTIONS = {
    "obs_format",
    "pred_format",
    "obs_frame",
    "pred_frame",
    "obs_chain",
    "pred_chain",
    "strategy",
    "output_type",
}
EXCLUDED_FIELDS = {"obs_file", "pred_file", "output_file", "logger"}


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


def _extra_options() -> Dict[str, Any]:
    return {
        "shift_file_option": (
            Optional[Path],
            typer.Option(None, "--shift_file", "--shift-file", help="Observed shifts file.", rich_help_panel="Common"),
        ),
        "pred_file_option": (
            Optional[Path],
            typer.Option(None, "--pred_file", "--pred-file", help="Predicted shifts file.", rich_help_panel="Common"),
        ),
        "output_file_option": (
            Path,
            typer.Option(Path("-"), "--output_file", "--output-file", help="Assignment output file (default: -).", rich_help_panel="Common"),
        ),
        "config_file": (
            Optional[Path],
            typer.Option(None, "--config_file", "--config-file", "-c", help="YAML configuration file. CLI options override file values.", rich_help_panel="Common"),
        ),
        "advanced": (
            bool,
            typer.Option(False, "--advanced", help="Show advanced options and exit.", is_eager=True, callback=_show_advanced_help),
        ),
    }


def _main_impl(**kwargs: Any) -> None:
    """Run the SNAPS assignment pipeline."""
    shift_file = kwargs.pop("shift_file_option") or kwargs.pop("shift_file")
    pred_file = kwargs.pop("pred_file_option") or kwargs.pop("pred_file")
    output_file = kwargs.pop("output_file")
    output_option = kwargs.pop("output_file_option")
    kwargs.pop("advanced")
    config_file = kwargs.pop("config_file")

    if output_file is None:
        output_file = output_option
    elif output_option != Path("-"):
        output_file = output_option

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
        name: convert_model_value(SNAPSConfig, name, value)
        for name, value in kwargs.items()
        if value is not None
    }
    overrides.update({"obs_file": shift_file, "pred_file": pred_file, "output_file": output_file})
    config = build_config(config_file=config_file, cli_overrides=overrides)
    run_pipeline(config)


_main_impl.__signature__ = signature_from_model(
    SNAPSConfig,
    positional={
        "shift_file": (
            Optional[Path],
            typer.Argument(None, help="Observed shifts file (or a NEF file containing both lists)."),
        ),
        "pred_file": (
            Optional[Path],
            typer.Argument(None, help="Predicted shifts file."),
        ),
        "output_file": (
            Optional[Path],
            typer.Argument(None, help="Assignment output file (default: -)."),
        ),
    },
    excluded=EXCLUDED_FIELDS,
    common_options=COMMON_OPTIONS,
    extra_options=_extra_options(),
)
app.command(name="main")(_main_impl)


def cli() -> None:
    """Run the CLI, showing help when no arguments are supplied."""
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    app()


if __name__ == "__main__":
    cli()
