"""Build Typer parameters from a Pydantic configuration model."""

from __future__ import annotations

import inspect
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union, get_args, get_origin

import typer
from pydantic.fields import FieldInfo

try:
    from types import UnionType
except ImportError:  # Python 3.8 and 3.9
    UnionType = None


def _is_optional(annotation: Any) -> bool:
    origin = get_origin(annotation)
    union_origins = (Union, UnionType) if UnionType is not None else (Union,)
    return origin in union_origins and type(None) in get_args(annotation)


def _is_bool(annotation: Any) -> bool:
    types = [item for item in get_args(annotation) if item is not type(None)]
    return annotation is bool or types == [bool]


def _option_names(name: str) -> tuple[str, str]:
    hyphenated = name.replace("_", "-")
    return (f"--{name}", f"--{hyphenated}") if name != hyphenated else (f"--{name}",)


def _field_cli_type(annotation: Any) -> Any:
    """Return the Typer input type, using a list for repeatable collections."""
    origin = get_origin(annotation)
    if origin in (set, Set, list, List):
        item_type = get_args(annotation)[0] if get_args(annotation) else str
        return Optional[List[item_type]]
    if origin in (dict, Dict):
        return Optional[List[str]]
    return annotation if _is_optional(annotation) else Optional[annotation]


def option_for_field(
    name: str,
    field: FieldInfo,
    default: Any,
    *,
    common_options: Set[str],
) -> inspect.Parameter:
    """Create an inspect parameter whose default is a Typer option."""
    annotation = field.annotation
    option_names = _option_names(name)
    is_bool = _is_bool(annotation)
    if is_bool:
        hyphenated = name.replace("_", "-")
        option_names = (
            f"--{name}/--no-{name}",
            f"--{hyphenated}/--no-{hyphenated}",
        ) if name != hyphenated else (f"--{name}/--no-{name}",)

    option = typer.Option(
        None,
        *option_names,
        help=field.description or name.replace("_", " ").capitalize(),
        show_default=str(default),
        rich_help_panel="Common" if name in common_options else "Advanced",
        hidden=name not in common_options,
    )
    return inspect.Parameter(
        name,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        annotation=_field_cli_type(annotation),
        default=option,
    )


def signature_from_model(
    model: Any,
    *,
    positional: Dict[str, Any],
    excluded: Set[str],
    common_options: Set[str],
    extra_options: Dict[str, Any],
) -> inspect.Signature:
    """Construct a Typer-compatible signature from model fields and extras."""
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
            default=default,
        )
        for name, (annotation, default) in positional.items()
    ]
    parameters.extend(
        inspect.Parameter(
            name,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
            default=default,
        )
        for name, (annotation, default) in extra_options.items()
    )
    defaults = model()
    for name, field in model.model_fields.items():
        if name in excluded:
            continue
        parameters.append(
            option_for_field(
                name,
                field,
                getattr(defaults, name),
                common_options=common_options,
            )
        )
    return inspect.Signature(parameters)


def convert_model_value(model: Any, name: str, value: Any) -> Any:
    """Convert repeatable CLI values into the collection shape expected by Pydantic."""
    annotation = model.model_fields[name].annotation
    origin = get_origin(annotation)
    if value is None:
        return None
    if origin in (set, Set):
        return set(value)
    if origin in (dict, Dict):
        result: Dict[str, float] = {}
        for item in value:
            key, separator, raw_value = item.partition("=")
            if not separator or not key or not raw_value:
                raise typer.BadParameter(
                    f"{name} values must use KEY=VALUE (for example H=0.1726)"
                )
            try:
                result[key] = float(raw_value)
            except ValueError as error:
                raise typer.BadParameter(
                    f"value for {key!r} in {name} must be a number"
                ) from error
        return result
    return value
