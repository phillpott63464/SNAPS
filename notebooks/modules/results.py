"""Readers and summary-table builders for SNAPS and MARS results."""

from __future__ import annotations

import shlex
from pathlib import Path

import pandas as pd

RELIABLE_MERIT = 0.01
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))



def read_test_result(path: Path) -> pd.DataFrame:
    lines = path.read_text().splitlines()
    required = {"Res_name", "Res_N", "SS_name", "Log_prob", "Confidence"}
    for index, line in enumerate(lines):
        headings = shlex.split(line.strip())
        if required.issubset(headings):
            break
    else:
        raise ValueError(f"No SNAPS assignment table found in {path}")

    rows = []
    for line in lines[index + 1:]:
        if not line.strip():
            continue
        fields = shlex.split(line.strip())
        if len(fields) != len(headings):
            if rows:
                break
            continue
        rows.append(fields)
    if not rows:
        raise ValueError(f"No assignment rows found in {path}")

    result = pd.DataFrame(rows, columns=headings)
    for column in ("Res_N", "Log_prob", "Max_mismatch_m1", "Max_mismatch_p1", "Max_mismatch",
                   "Num_good_links_m1", "Num_good_links_p1", "Num_good_links"):
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    for column in ("Dummy_res", "Dummy_SS"):
        if column in result:
            result[column] = result[column].str.lower().map({"true": True, "false": False})
    result["assigned"] = ~result["Dummy_res"]
    result["chain_code"] = "A"
    result["residue_name"] = result["Res_type"].str.upper().map(AA3).fillna(result["Res_type"])
    result["unassigned_sequence_code"] = result["SS_name"]
    result["sequence_code"] = result["Res_N"]
    result["spin_system"] = pd.to_numeric(result["SS_name"].str.extract(r"^\s*(-?\d+)")[0], errors="coerce")
    result["residue_number"] = pd.to_numeric(result["Res_N"], errors="coerce")

    # For residue fingerprinting
    observed_shift_columns = [
        "C", "CA", "CB", "H", "HA", "N"
    ]
    observed_shift_columns = [
        column for column in observed_shift_columns if column in result.columns
    ]

    for column in observed_shift_columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result["observed_shifts"] = result[observed_shift_columns].apply(
        lambda row: {
            atom: float(value)
            for atom, value in row.items()
            if pd.notna(value)
        },
        axis=1,
    )
    return result


def read_reference_assignment(reference_file: Path) -> pd.DataFrame:
    prefix = "_nef_chemical_shift."
    headings = []
    rows = []
    in_loop = False
    in_shift_loop = False

    for line in reference_file.read_text().splitlines():
        value = line.strip()

        if value == "loop_":
            in_loop = True
            in_shift_loop = False
            headings = []
            continue

        if not in_loop:
            continue

        if value.startswith(prefix):
            in_shift_loop = True
            headings.append(value[len(prefix):].split()[0])
            continue

        if value in {"stop_", "save_"}:
            if in_shift_loop:
                break
            in_loop = False
            continue

        if in_shift_loop and value:
            fields = shlex.split(value)
            if len(fields) != len(headings):
                raise ValueError(
                    f"Malformed NEF chemical-shift row in {reference_file}: {value}"
                )
            rows.append(fields)

    if not rows:
        raise ValueError(f"No NEF chemical-shift rows found in {reference_file}")

    shifts = pd.DataFrame(rows, columns=headings)
    shifts = shifts.rename(
        columns={
            "sequence_code": "sequence_code",
            "residue_name": "residue_name",
            "atom_name": "atom_name",
            "value": "value",
        }
    )

    shifts["sequence_code"] = shifts["sequence_code"].replace({".": pd.NA, "?": pd.NA})
    shifts["value"] = pd.to_numeric(
        shifts["value"].replace({".": pd.NA, "?": pd.NA}),
        errors="coerce",
    )

    shifts["value"] = pd.to_numeric(
        shifts["value"].replace({".": pd.NA, "?": pd.NA}),
        errors="coerce",
    )

    shifts = shifts.dropna(subset=["sequence_code", "value"])

    grouped = []
    for (sequence_code, residue_name), group in shifts.groupby(
        ["sequence_code", "residue_name"], sort=False
    ):
        atom_shifts = dict(zip(group["atom_name"], group["value"]))
        grouped.append({
            "sequence_code": sequence_code,
            "residue_name": residue_name,
            "shifts": atom_shifts,
        })

    return pd.DataFrame(grouped)

def match_reference_residue(
    observed_shifts: dict,
    reference: pd.DataFrame,
    tolerance: float = 0.01,
    min_atoms: int = 2,
):
    """Return the unique reference sequence code matching the observed shifts."""

    observed_shift_columns = [
        "C", "CA", "CB", "H", "HA", "N"
    ]

    observed = {
        atom: float(value)
        for atom, value in observed_shifts.items()
        if atom in observed_shift_columns and pd.notna(value)
    }

    matches = []
    for row in reference.itertuples(index=False):
        reference_shifts = row.shifts
        common_atoms = observed.keys() & reference_shifts.keys()

        if len(common_atoms) < min_atoms:
            continue

        if all(
            abs(observed[atom] - float(reference_shifts[atom])) <= tolerance
            for atom in common_atoms
        ):
            matches.append(row.sequence_code)

    return matches[0] if len(matches) == 1 else pd.NA


def assignment_counts(
    result: pd.DataFrame,
    reference_file: Path,
    chain: str = "A",
    good_confidences: list[str] = ["High"],
) -> dict:
    assigned = result.loc[
        result["assigned"]
        & result["chain_code"].eq(chain)
        & result["residue_name"].ne("PRO")
    ].copy()

    if "merit" in assigned:
        reliable = assigned.loc[assigned["merit"].le(RELIABLE_MERIT)].copy()
    elif "Confidence" in assigned:
        reliable = assigned.loc[
            assigned["Confidence"].isin(good_confidences)
        ].copy()
    else:
        raise ValueError("Result has neither merit nor Confidence")

    reference = read_reference_assignment(reference_file)

    reliable["reference_sequence_code"] = reliable["observed_shifts"].apply(
        lambda shifts: match_reference_residue(shifts, reference)
    )

    source = pd.to_numeric(reliable["reference_sequence_code"], errors="coerce")
    predicted = pd.to_numeric(reliable["residue_number"], errors="coerce")
    evaluable = source.notna() & predicted.notna()
    errors = int(source[evaluable].ne(predicted[evaluable]).sum())

    return {
        "assignments": int(len(assigned)),
        "reliable_assignments": int(len(reliable)),
        "errors": errors,
    }


def assignment_table(bmrb_ids, strategies, protein_names, data_dir: Path, output_dir: Path, good_confidences: list[str] = ["High"]) -> pd.DataFrame:
    rows = []
    for bmrb_id in bmrb_ids:
        for condition, filename in strategies.items():
            result_path = output_dir / filename.format(bmrb_id=bmrb_id)
            reference_path = data_dir / f'{bmrb_id}.nef'
            counts = assignment_counts(read_test_result(result_path), reference_path, good_confidences=good_confidences) if result_path.is_file() else {
                "assignments": float("nan"), "reliable_assignments": float("nan"), "errors": float("nan")}
            rows.append(
                {
                    "bmrb_id": bmrb_id,
                    "protein": protein_names[bmrb_id],
                    "chemical_shifts_used": "—",
                    "condition": condition,
                    **counts,
                }
            )
    result = pd.DataFrame(rows)
    result[["assignments", "reliable_assignments"]] = result[["assignments", "reliable_assignments"]].astype("Int64")
    return result
