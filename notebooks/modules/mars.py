
import re

import pandas as pd

from modules.bmrb import run_mars_set
from modules.results import read_reference_assignment

# BMRB_NAMES = {
#     5471: "Malate synthase G",
# } example
def get_mars_data(BMRB_IDS, BMRB_NAMES, DATA_DIR, use_defaults: bool = False):

    if use_defaults:
        out_name = 'mars-orig'
        mars_data = {
            "I": {
                (5471, "C′, Cα, Cβ"): (652, 639, 0),
                (5471, "Cα, Cβ"): (500, 207, 0),
                (4354, "Cα, Cβ"): (323, 303, 0),
                (4384, "C′, Cα, Cβ"): (214, 205, 0),
                (4022, "C′, Cα, Cβ"): (242, 235, 0),
                (4106, "Cα, Cβ"): (246, 232, 0),
                (4457, "Cα, Cβ"): (116, 77, 1),
                (4402, "Cα, Cβ"): (138, 103, 0),
                (4341, "C′, Cα, Cβ"): (112, 101, 0),
                (4341, "Cα, Cβ"): (111, 101, 0),
                (547, "Cα, C′"): (97, 37, 0),
                (4082, "Cα, Cβ"): (130, 132, 2),
                (4136, "C′, Cα, Cβ"): (61, 35, 0),
                (5387, "Cα, Cβ"): (72, 72, 0),
                (5387, "Cα only"): (32, 9, 0),
                ("N/A", "Cα, Cβ, Hα"): (65, 65, 0),
                ("N/A", "Cα, Cβ"): (57, 34, 0),
                ("N/A", "C′, Cα, Cβ"): (91, 80, 0),
            },
            "II": {
                (5471, "C′, Cα, Cβ"): (652, 639, 0),
                (5471, "Cα, Cβ"): (639, 584, 2),
                (4354, "Cα, Cβ"): (333, 324, 0),
                (4384, "C′, Cα, Cβ"): (218, 207, 0),
                (4022, "C′, Cα, Cβ"): (242, 237, 0),
                (4106, "Cα, Cβ"): (246, 246, 0),
                (4457, "Cα, Cβ"): (134, 102, 0),
                (4402, "Cα, Cβ"): (155, 127, 0),
                (4341, "C′, Cα, Cβ"): (112, 104, 0),
                (4341, "Cα, Cβ"): (112, 104, 0),
                (547, "Cα, C′"): (144, 142, 0),
                (4082, "Cα, Cβ"): (132, 132, 0),
                (4136, "C′, Cα, Cβ"): (70, 58, 0),
                (5387, "Cα, Cβ"): (72, 72, 0),
                (5387, "Cα only"): (58, 18, 0),
                ("N/A", "Cα, Cβ, Hα"): None,
                ("N/A", "Cα, Cβ"): None,
                ("N/A", "C′, Cα, Cβ"): None,
            },
            "III": {
                (5471, "C′, Cα, Cβ"): (651, 623, 0),
                (5471, "Cα, Cβ"): (622, 511, 0),
                (4354, "Cα, Cβ"): (330, 313, 1),
                (4384, "C′, Cα, Cβ"): (218, 199, 0),
                (4022, "C′, Cα, Cβ"): (242, 225, 0),
                (4106, "Cα, Cβ"): (248, 245, 0),
                (4457, "Cα, Cβ"): (136, 70, 1),
                (4402, "Cα, Cβ"): (154, 118, 0),
                (4341, "C′, Cα, Cβ"): (111, 100, 0),
                (4341, "Cα, Cβ"): (112, 103, 0),
                (547, "Cα, C′"): (136, 119, 0),
                (4082, "Cα, Cβ"): (132, 123, 0),
                (4136, "C′, Cα, Cβ"): (64, 50, 0),
                (5387, "Cα, Cβ"): (72, 70, 0),
                (5387, "Cα only"): (58, 9, 0),
                ("N/A", "Cα, Cβ, Hα"): None,
                ("N/A", "Cα, Cβ"): None,
                ("N/A", "C′, Cα, Cβ"): None,
            },
        }

        rows = []

        for condition, results_by_bmrb in mars_data.items():
            for (bmrb_id, chemical_shifts_used), result in results_by_bmrb.items():
                if result is None:
                    continue

                assignments, reliable, errors = result
                rows.append({
                    "protein": lookup_name(bmrb_id, BMRB_NAMES) if bmrb_id != "N/A" else "N/A",
                    "bmrb_id": str(bmrb_id),
                    "chemical_shifts_used": chemical_shifts_used,
                    "condition": str(condition),
                    "assignments": assignments,
                    "reliable_assignments": reliable,
                    "errors": errors,
                })
    else:
        out_name = 'MARS-run'
        mars_data_files = run_mars_set(BMRB_IDS, DATA_DIR)
        mars_data = format_mars(mars_data_files)



        rows = []
        for condition, results_by_bmrb in mars_data.items():
            for bmrb_id, (assignments, reliable, errors) in results_by_bmrb.items():
                rows.append(
                    {
                        "protein": lookup_name(bmrb_id, BMRB_NAMES),
                        'bmrb_id': str(bmrb_id),
                        "chemical_shifts_used": 'all',  # Not changing in this output
                        "condition": str(condition),
                        "assignments": assignments,
                        "reliable_assignments": reliable,
                        "errors": errors,
                    }
                )

    mars_results = pd.DataFrame(rows)
    mars_results[["assignments", "reliable_assignments", "errors"]] = mars_results[
        ["assignments", "reliable_assignments", "errors"]
    ].astype("Int64")
    mars_results.attrs['name'] = out_name

    return mars_results


def format_mars(mars_files: dict):
    output = {}

    for condition, proteins in mars_files.items():
        output[condition] = {}

        for bmrb_id, files in proteins.items():
            reference = read_reference_assignment(files["nef_file"])
            output[condition][bmrb_id] = parse_assignment_aa(
                files["assignment_AA"],
                reference,
            )

    return output


def parse_assignment_aa(path, reference):
    reference_residues = dict(
        zip(
            reference["sequence_code"].astype(str),
            reference["residue_name"].str.upper(),
        )
    )

    assignments = 0
    reliable = 0
    errors = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 2:
                continue

            target = re.fullmatch(r"([A-Z]{3})_(\d+)", fields[0])
            assigned = re.fullmatch(r"[A-Z]+_(\d+)_([A-Z]{3})", fields[1])
            if not target or not assigned:
                continue

            target_name, target_code = target.groups()
            assigned_code, assigned_name = assigned.groups()

            assignments += 1
            if len(fields) >= 3 and fields[2] == "(H)":
                reliable += 1

            # Only score assignments that can be checked against the NEF.
            reference_name = reference_residues.get(assigned_code)
            if reference_name is None:
                continue

            if (
                    (assigned_code != target_code
                or assigned_name != target_name
                or assigned_name != reference_name)
                and len(fields) >= 3
                and fields[2] == "(H)"
            ):
                errors += 1

    return assignments, reliable, errors

def lookup_name(bmrb_id, BMRB_NAMES):
    # Handle IDs stored as either integers or strings in BMRB_NAMES.
    name = BMRB_NAMES.get(bmrb_id)
    if name is None:
        name = BMRB_NAMES.get(str(bmrb_id))
    if name is None:
        try:
            name = BMRB_NAMES.get(int(bmrb_id))
        except (TypeError, ValueError):
            pass
    if name is None:
        raise KeyError(f"No protein name found for BMRB ID {bmrb_id!r}")
    return name
