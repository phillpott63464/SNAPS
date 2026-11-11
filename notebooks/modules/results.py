"""Readers and summary-table builders for SNAPS and MARS results."""

from __future__ import annotations

import shlex
from pathlib import Path

import pandas as pd

RELIABLE_MERIT = 0.01
AA3 = dict(zip("ACDEFGHIKLMNPQRSTVWY", "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()))


def read_test_result(path: Path) -> pd.DataFrame:
    return read_test_result_nef(path) if path.suffix == ".nef" else read_test_result_snaps(path)


def read_test_result_snaps(path: Path) -> pd.DataFrame:
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
    return result


def read_test_result_nef(path: Path) -> pd.DataFrame:
    prefix = "_nefpls_assignments."
    headings, rows = [], []
    in_loop = False
    for line in path.read_text().splitlines():
        value = line.strip()
        if value == "loop_":
            in_loop, headings, rows = True, [], []
        elif in_loop and value.startswith(prefix):
            headings.append(value[len(prefix):].split()[0])
        elif in_loop and value == "stop_":
            if headings:
                break
            in_loop = False
        elif in_loop and value and headings:
            rows.append(shlex.split(value))
    if not rows:
        raise ValueError(f"No {prefix} loop found in {path}")
    result = pd.DataFrame(rows, columns=headings)
    result["assigned"] = result["assigned"].str.lower() == "true"
    result["merit"] = result["merit"].astype(float)
    result["spin_system"] = pd.to_numeric(result["unassigned_sequence_code"].str.strip(), errors="coerce")
    result["residue_number"] = pd.to_numeric(result["sequence_code"], errors="coerce")
    return result


def assignment_counts(result: pd.DataFrame, chain: str = "A", label_to_residue=None) -> dict:
    assigned = result.loc[result["assigned"] & result["chain_code"].eq(chain) & result["residue_name"].ne("PRO")].copy()
    if "merit" in assigned:
        reliable = assigned.loc[assigned["merit"].le(RELIABLE_MERIT)]
    elif "Confidence" in assigned:
        reliable = assigned.loc[assigned["Confidence"].isin(["High", "Medium"])]
    else:
        raise ValueError("Result has neither merit nor Confidence")
    reference = reliable["spin_system"].copy()
    if label_to_residue is not None:
        labels = reliable["unassigned_sequence_code"].astype(str).str.strip()
        reference = pd.to_numeric(labels.map(label_to_residue), errors="coerce").combine_first(reference)
    predicted = pd.to_numeric(reliable["residue_number"], errors="coerce")
    reference = pd.to_numeric(reference, errors="coerce")
    evaluable = reference.notna() & predicted.notna() & reference.mod(1).eq(0) & predicted.mod(1).eq(0)
    errors = int(reference[evaluable].ne(predicted[evaluable]).sum()) if evaluable.all() else float("nan")
    return {"assignments": int(len(assigned)), "reliable_assignments": int(len(reliable)), "errors": errors}


def assignment_table(bmrb_ids, strategies, protein_names, output_dir: Path) -> pd.DataFrame:
    rows = []
    for bmrb_id in bmrb_ids:
        for condition, filename in strategies.items():
            path = output_dir / filename.format(bmrb_id=bmrb_id)
            counts = assignment_counts(read_test_result(path)) if path.is_file() else {
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
