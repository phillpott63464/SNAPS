from typing import Optional

import pandas as pd
import logging
import numpy as np

def import_predicted_shifts(
        filename: str,
        filetype: str,
        seq_df: Optional[pd.DataFrame]=None,
        logger: logging.Logger = None,
        offset=0):
    """Import predicted chemical shifts from a ShiftX2 results file.

    Returns
    A DataFrame containing the predicted shifts, or None if the import failed.

    Parameters
    filename: path to file containing predicted shifts
    filetype: either "shiftx2" or "sparta+"
    offset: an optional integer to add to the residue number.
    """

    #### Import the raw data
    if filetype == 'shiftx2':
        preds_long = pd.read_csv(filename)
        if any(preds_long.columns == 'CHAIN'):
            if len(preds_long['CHAIN'].unique()) > 1:
                logger.warning(
                    """Chain identifier dropped - if multiple chains are
                        present in the predictions, they will be merged."""
                )
            preds_long = preds_long.drop('CHAIN', axis=1)
        preds_long = preds_long.reindex(
            columns=['NUM', 'RES', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']
    elif filetype == 'sparta+':
        # Work out where the column names and data are
        with open(filename, 'r') as f:
            for num, line in enumerate(f, 1):
                if line.find('VARS') > -1:
                    colnames_line = num
                    colnames = line.split()[1:]
                    break

        preds_long = pd.read_table(
            filename, sep='\s+', names=colnames, skiprows=colnames_line + 1
        )
        preds_long = preds_long.reindex(
            columns=['RESID', 'RESNAME', 'ATOMNAME', 'SHIFT']
        )
        preds_long.columns = ['Res_N', 'Res_type', 'Atom_type', 'Shift']

        # Sparta+ uses HN for backbone amide proton - convert to H
        preds_long.loc[preds_long['Atom_type'] == 'HN', 'Atom_type'] = 'H'
    else:
        logger.error(
            """Invalid predicted shift type: '%s'. Allowed
                          options are 'shiftx2' or 'sparta+'"""
            % (filetype)
        )
        return None

    logger.info(
        'Imported %d predicted chemical shifts from %s'
        % (len(preds_long.index), filename)
    )

    #### Initial processing and conversion from long to wide
    # Add sequence number offset and create residue names
    preds_long['Res_N'] = preds_long['Res_N'] + offset
    preds_long.insert(
        1,
        'Res_name',
        (preds_long['Res_N'].astype(str) + preds_long['Res_type']),
    )
    # Left pad with spaces to a constant length (helps with sorting)
    preds_long['Res_name'] = preds_long['Res_name'].str.rjust(5)

    # Convert from long to wide format
    preds = preds_long.pivot(
        index='Res_N', columns='Atom_type', values='Shift'
    )
    preds.index.name = None

    # Add the other residue name and type back in
    tmp = preds_long[['Res_N', 'Res_type', 'Res_name']]
    tmp = tmp.drop_duplicates(subset='Res_name')
    tmp.index = tmp['Res_N']
    tmp.index.name = None
    preds = pd.concat([tmp, preds], axis=1)

    #### Make consistent with seq_df (and create if it doesn't already exist)
    # TODO: Maybe this should be split off into a separate function?
    # If seq_df is missing, create it based on preds
    if seq_df is None:
        seq_df = preds.copy()[['Res_N', 'Res_type', 'Res_name']]

        # If there are any missing residue numbers, create them
        min_N = seq_df['Res_N'].min()
        max_N = seq_df['Res_N'].max()
        missing_residue_numbers = set(range(min_N, max_N + 1)).difference(
            seq_df['Res_N']
        )
        if len(missing_residue_numbers) > 0:
            tmp = pd.DataFrame(
                {
                    'Res_N': list(missing_residue_numbers),
                    'Res_type': 'X',
                    'Res_name': np.NaN,
                }
            )
            tmp['Res_name'] = tmp['Res_N'].astype(str) + tmp['Res_type']
            tmp['Res_name'] = tmp['Res_name'].str.rjust(5)
            tmp.index = tmp['Res_N']
            tmp.index.name = None
            # seq_df = seq_df.append(tmp).sort_index()
            seq_df = pd.concat([seq_df, tmp]).sort_index()

    # Add/delete residues from preds so it matches seq_df
    # Only keep predictions that are in seq_df
    tmp = len(preds.index)
    preds = preds[preds['Res_N'].isin(seq_df['Res_N'])]
    tmp2 = tmp - len(preds.index)
    if tmp2 > 0:
        logger.info(
            (
                'Predictions for %d residues were discarded '
                'because they were not present in the imported '
                'sequence file'
            )
            % tmp2
        )

    # Add predictions for any residue number that is in seq_df but not preds
    tmp = pd.DataFrame(
        data=seq_df[~seq_df['Res_N'].isin(preds['Res_N'])],
        columns=preds.columns,
    )
    # preds = preds.append(tmp)
    preds = pd.concat([preds, tmp])
    if len(tmp.index) > 0:
        logger.info(
            (
                '%d residues from the sequence were missing '
                'from the predictions'
            )
            % len(tmp.index)
        )

    # If Res_name is inconsistent between seq_df and preds, make a compromise
    preds.index = preds['Res_N']
    preds = preds.sort_index()
    seq_df.index = seq_df['Res_N']
    seq_df = seq_df.sort_index()
    mask = preds['Res_name'] != seq_df['Res_name']

    if any(mask):
        ambiguous_res_names = (
            seq_df.loc[mask, 'Res_name']
            + '('
            + preds.loc[mask, 'Res_type']
            + '?)'
        )
        preds.loc[mask, 'Res_name'] = ambiguous_res_names
        seq_df.loc[mask, 'Res_name'] = ambiguous_res_names
        # Note we don't update the Res_type in preds, because this is potentially
        # used for correcting the chemical shift
        logger.warning(
            (
                'There were inconsistencies between the provided sequence '
                'file and the predicted shifts. The following residues had '
                'inconsistent amino acid types: %s'
            )
            % ', '.join(ambiguous_res_names)
        )

    #        preds.index = preds["Res_name"]
    #        preds.index.name = None
    seq_df.index = seq_df['Res_name']
    seq_df.index.name = None

    #### Add the chemical shift info back in

    # Make columns for the i-1 predicted shifts of C, CA and CB
    preds_m1 = preds[
        list(
            {'C', 'CA', 'CB', 'Res_type', 'Res_name'}.intersection(
                preds.columns
            )
        )
    ].copy()
    preds_m1.index = preds_m1.index + 1
    preds_m1.columns = preds_m1.columns + '_m1'
    preds = pd.merge(
        preds, preds_m1, how='left', left_index=True, right_index=True
    )

    # Make column for the i+1 Res_name
    preds_p1 = preds[['Res_name']].copy()
    preds_p1.index = preds_p1.index - 1
    preds_p1.columns = ['Res_name_p1']
    preds = pd.merge(
        preds, preds_p1, how='left', left_index=True, right_index=True
    )

    # Set index to Res_name
    preds.index = preds['Res_name']
    preds.index.name = None

    # Restrict to only certain atom types
    atom_set = {'H', 'N', 'C', 'CA', 'CB', 'C_m1', 'CA_m1', 'CB_m1', 'HA'}
    preds = preds[
        [
            'Res_name',
            'Res_N',
            'Res_type',
            'Res_name_m1',
            'Res_name_p1',
            'Res_type_m1',
        ]
        + list(atom_set.intersection(preds.columns))
    ]

    logger.info(
        'Finished reading in %d predicted residues from %s'
        % (len(preds.index), filename)
    )

    return preds, seq_df
