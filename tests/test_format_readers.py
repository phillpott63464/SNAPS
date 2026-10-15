import logging
from urllib import request

import pandas as pd
import pytest
from bokeh.client.states import ErrorReason
from pandas.testing import assert_frame_equal
from pathlib import Path
import numpy as np

from modules.format_readers import (
    ShiftX2Reader,
    SpartaReader,
    NEFReader,
    NMRPipeReader,
    CCPNReader,
    SparkyReader,
    SnapsReader,
    XeasyReader,
    MarsReader,
)


@pytest.fixture
def shiftx2reader():
    return ShiftX2Reader()


@pytest.fixture
def spartareader():
    return SpartaReader()


@pytest.fixture
def nefreader():
    return NEFReader()


@pytest.fixture
def nmrpipereader():
    return NMRPipeReader()


@pytest.fixture
def ccpnreader():
    return CCPNReader()


@pytest.fixture
def sparkyreader():
    return SparkyReader()


@pytest.fixture
def snapsreader():
    return SnapsReader()


@pytest.fixture
def xeasyreader():
    return XeasyReader()


@pytest.fixture
def marsreader():
    return MarsReader()


@pytest.fixture
def shiftx2_file(tmp_path):
    """Create a sample ShiftX2 prediction file."""
    file_path = tmp_path / 'predictions.cs'

    data_file = pd.DataFrame(
        {
            'NUM': [1, 1, 2, 2],
            'RES': ['ALA', 'ALA', 'GLY', 'GLY'],
            'ATOMNAME': ['CA', 'CB', 'CA', 'N'],
            'SHIFT': [52.1, 19.3, 45.2, 109.5],
        }
    )

    expected_result = pd.DataFrame(
        {
            'Res_N': [1, 1, 2, 2],
            'Res_type': ['ALA', 'ALA', 'GLY', 'GLY'],
            'Atom_type': ['CA', 'CB', 'CA', 'N'],
            'Shift': [52.1, 19.3, 45.2, 109.5],
        }
    )

    data_file.to_csv(file_path, index=False)
    return (file_path, expected_result)


@pytest.fixture
def sparta_file(tmp_path):
    file_path = tmp_path / 'sparta.txt'

    data = """
REMARK SPARTA+ Protein Chemical Shift Prediction Table
REMARK  
REMARK  All chemical shifts are reported in ppm:
REMARK  
REMARK  SHIFT    is the predicted chemical shift.
REMARK  SS_SHIFT is the predicted secondary chemical shift.
REMARK  RC_SHIFT is the random coil chemical shift.
REMARK  HM_SHIFT is the ring current shift; a correction
REMARK           of 0.6 times this value is applied.
REMARK  EF_SHIFT is the adjustment for electric field effect,
REMARK           applied to proton shifts only.
REMARK  SIGMA    is the estimated prediction error.
REMARK  
REMARK Reference:
REMARK  Y. Shen and A. Bax: SPARTA+: a modest improvement in empirical NMR
REMARK  chemical shift prediction by means of an artificial neural network.
REMARK  J. Biomol. NMR 48, 13-22 (2010).

DATA SOURCE P3a_L273R_clement_model.pdb

DATA FIRST_RESID 28

DATA SEQUENCE IPNTDYIQLL SEIAKEQGFN ITYLDIDELS ANGQYQCRAE LSTSPITVCH 
DATA SEQUENCE GSGISCGNAQ SDAAHNALQY LKIIXERK

VARS   RESID RESNAME ATOMNAME SS_SHIFT SHIFT RC_SHIFT HM_SHIFT EF_SHIFT SIGMA 
FORMAT %4d %4s %4s %9.3f %9.3f %9.3f %9.3f %9.3f %9.3f

  28    I   HA     0.000     4.204     4.170     0.056     0.000     0.288
  28    I    C     0.000   176.144   176.117     0.045     0.000     1.063
  28    I   CA     0.000    61.340    61.300     0.068     0.000     1.566
  28    I   CB     0.000    38.028    38.000     0.046     0.000     1.214
    """

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'Res_N': [28, 28, 28, 28],
            'Res_type': ['I', 'I', 'I', 'I'],
            'Atom_type': ['HA', 'C', 'CA', 'CB'],
            'Shift': [4.204, 176.144, 61.340, 38.028],
        }
    )

    return (file_path, expected_result)


def _nef_file(tmp_path):
    file_path = tmp_path / 'nef.nef'

    data = """
    data_fasta

    save_nef_nmr_meta_data
       _nef_nmr_meta_data.sf_category      nef_nmr_meta_data
       _nef_nmr_meta_data.sf_framecode     nef_nmr_meta_data
       _nef_nmr_meta_data.format_name      nmr_exchange_format
       _nef_nmr_meta_data.format_version   1.1
       _nef_nmr_meta_data.program_name     NEFPipelines
       _nef_nmr_meta_data.script_name      fasta/import/sequence.py
       _nef_nmr_meta_data.program_version  0.1.82
       _nef_nmr_meta_data.creation_date    2024-08-06T08:40:13.386601
       _nef_nmr_meta_data.uuid             NEFPipelines-2024-08-06T08:40:13.386601-7370929880

       loop_
          _nef_run_history.run_number
          _nef_run_history.program_name
          _nef_run_history.program_version
          _nef_run_history.script_name


       stop_

    save_

    save_nef_molecular_system
       _nef_molecular_system.sf_category   nef_molecular_system
       _nef_molecular_system.sf_framecode  nef_molecular_system

       loop_
          _nef_sequence.index
          _nef_sequence.chain_code
          _nef_sequence.sequence_code
          _nef_sequence.residue_name
          _nef_sequence.linking
          _nef_sequence.residue_variant
          _nef_sequence.cis_peptide

         1    A   235   GLY   start    .   .    
         2    A   236   PRO   middle   .   .    
         3    A   237   ALA   middle   .   .  

       stop_

    save_

    save_nef_chemical_shift_list_shiftx2
       _nef_chemical_shift_list.sf_category   nef_chemical_shift_list
       _nef_chemical_shift_list.sf_framecode  nef_chemical_shift_list_shiftx2

       loop_
          _nef_chemical_shift.chain_code
          _nef_chemical_shift.sequence_code
          _nef_chemical_shift.residue_name
          _nef_chemical_shift.atom_name
          _nef_chemical_shift.value
          _nef_chemical_shift.value_uncertainty
          _nef_chemical_shift.element
          _nef_chemical_shift.isotope_number

         A   236   ILE   C    173.7894   .   .   .    
         A   236   ILE   CA   59.4824    .   .   .    
         A   236   ILE   CB   37.1218    .   .   .    
         A   236   ILE   H    8.0038     .   .   .     

       stop_

    save_

    save_nef_chemical_shift_list_default
       _nef_chemical_shift_list.sf_category   nef_chemical_shift_list
       _nef_chemical_shift_list.sf_framecode  nef_chemical_shift_list_default

       loop_
          _nef_chemical_shift.chain_code
          _nef_chemical_shift.sequence_code
          _nef_chemical_shift.residue_name
          _nef_chemical_shift.atom_name
          _nef_chemical_shift.value
          _nef_chemical_shift.value_uncertainty
          _nef_chemical_shift.element
          _nef_chemical_shift.isotope_number

         A   236   PRO   C      176.443   .   C   13    
         A   236   PRO   CA     62.822    .   C   13    
         A   236   PRO   CB     32.269    .   C   13    
         A   236   PRO   CD     49.668    .   C   13    

       stop_

    save_


        """

    with open(file_path, 'w') as f:
        f.write(data)

    return file_path


@pytest.fixture
def nef_file_pred(tmp_path):
    file_path = _nef_file(tmp_path)

    expected_result = pd.DataFrame(
        {
            'Res_N': [236, 236, 236, 236],
            'Res_type': ['P', 'P', 'P', 'P'],
            'Atom_type': ['C', 'CA', 'CB', 'CD'],
            'Shift': [176.443, 62.822, 32.269, 49.668],
        }
    )

    return (file_path, expected_result)


@pytest.fixture
def nef_file_obs(tmp_path):
    file_path = _nef_file(tmp_path)
    expected_result = pd.DataFrame(
        {
            'SS_name': ['236Pro', '236Pro', '236Pro', '236Pro'],
            'Atom_type': ['C', 'CA', 'CB', 'CD'],
            'Shift': [176.443, 62.822, 32.269, 49.668],
        }
    )

    return (file_path, expected_result)


@pytest.fixture
def nmrpipe_file(tmp_path):
    file_path = tmp_path / 'nmrpipe_file.tab'

    data = """

REMARK Chemical shifts for ubiquitin

DATA FIRST_RESID 1

DATA SEQUENCE MQIFVKTLTG KTITLEVEPS DTIENVKAKI QDKEGIPPDQ QRLIFAGKQL
DATA SEQUENCE EDGRTLSDYN IQKESTLHLV LRLRGG

VARS   RESID RESNAME ATOMNAME SHIFT
FORMAT %4d   %1s     %4s      %8.3f

   1 M   CB   33.270
   1 M   CA   54.450
   1 M    C  170.540
   1 M   HA    4.230
    """

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': ['1', '1', '1', '1'],
            'Atom_type': ['CB', 'CA', 'C', 'HA'],
            'Shift': [33.270, 54.450, 170.540, 4.230],
        }
    )

    return (file_path, expected_result)


@pytest.fixture
def ccpn_file(tmp_path):
    file_path = tmp_path / 'ccpn.txt'
    data = """
Number	#	Shift	SD	Assign Name	Residue	Isotope	Other Name	All Peaks	Shift List Peaks	Spin System #
1	243	19.37850	0.04473	Cb	310Ala	13C	None	4	4	74
2	244	52.85169	0.00706	Ca	310Ala	13C	None	4	4	74
3	245	64.66916	0.02857	Cb	265Ser	13C	None	6	6	68
4	246	57.99790	0.01886	Ca	265Ser	13C	None	5	5	68
    """

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': ['310Ala', '310Ala', '265Ser', '265Ser'],
            'Atom_type': ['CB', 'CA', 'CB', 'CA'],
            'Shift': [19.37850, 52.85169, 64.66916, 57.99790],
        }
    )

    return (file_path, expected_result)


@pytest.fixture
def sparky_file(tmp_path):
    file_path = tmp_path / 'sparky.txt'

    data = """
Group          Atom      Nuc      Shift      SDev     Assignments

 P236          C         13C      176.443     0.000       1
 P236          CA        13C       62.822     0.000       1
 P236          CB        13C       32.269     0.000       1

"""

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': ['P236', 'P236', 'P236'],
            'Atom_type': ['C', 'CA', 'CB'],
            'Shift': [176.443, 62.822, 32.269],
        }
    )

    return (file_path, expected_result)


@pytest.fixture
def snaps_file(tmp_path):
    file_path = tmp_path / 'snaps.txt'

    data = """
SS_name	Atom_type	Shift
15	 237AlaH	CA	52.372
633	 237AlaH	CA_m1	62.837
324	 237AlaH	CB	19.152
221	 237AlaH	CB_m1	32.296
"""

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': [
                ' 237AlaH',
                ' 237AlaH',
                ' 237AlaH',
                ' 237AlaH',
            ],  # Not sure if this whitespace is intentional
            'Atom_type': ['CA', 'CA_m1', 'CB', 'CB_m1'],
            'Shift': [52.372, 62.837, 19.152, 32.296],
        },
        index=[15, 633, 324, 221],
    )

    return (file_path, expected_result)


@pytest.fixture
def xeasy_file(tmp_path):
    file_path = tmp_path / 'xeasy.txt'

    data = """
   1 999.000 0.000 N     235
   2 999.000 0.000 CA    235
   3 999.000 0.000 HA1   235
   4 999.000 0.000 HA2   235
   5 999.000 0.000 QA    235
   6 999.000 0.000 C     235
   7 999.000 0.000 N     236
   8 999.000 0.000 CD    236
   9  62.837 0.000 CA    236
  10 999.000 0.000 HA    236
  11  32.296 0.000 CB    236
  12 999.000 0.000 HB2   236
  13 999.000 0.000 HB3   236
  14 999.000 0.000 QB    236
  15 999.000 0.000 CG    236
  16 999.000 0.000 HG2   236
  17 999.000 0.000 HG3   236
  18 999.000 0.000 QG    236
  19 999.000 0.000 HD2   236
  20 999.000 0.000 HD3   236
"""

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': ['236', '236'],
            'Atom_type': ['CA', 'CB'],
            'Shift': [62.837, 32.296],
        },
        index=[8, 10],
    )

    return (file_path, expected_result)


@pytest.fixture
def mars_file(tmp_path):
    file_path = tmp_path / 'mars.txt'

    data = """
        H        N        CA       CA-1     CB       CB-1     CO       CO-1      HA
PR_236  -        -        62.8219  -        32.269   -        176.443  -        4.43775
"""

    with open(file_path, 'w') as f:
        f.write(data)

    expected_result = pd.DataFrame(
        {
            'SS_name': ['PR_236'] * 9,
            'Atom_type': [
                'H',
                'HA',
                'N',
                'C',
                'CA',
                'CB',
                'C_m1',
                'CA_m1',
                'CB_m1',
            ],
            'Shift': [np.nan, 4.43775, np.nan, 176.443, 62.8219, 32.269]
            + [np.nan] * 3,
        }
    )

    return (file_path, expected_result)


@pytest.mark.parametrize(
    'reader, name',
    [
        ('shiftx2reader', 'shiftx2'),
        ('spartareader', 'sparta+'),
        ('nefreader', 'nef'),
        ('nmrpipereader', 'nmrpipe'),
        ('ccpnreader', 'ccpn'),
        ('sparkyreader', 'sparky'),
        ('snapsreader', 'snaps'),
        ('xeasyreader', 'xeasy'),
        ('marsreader', 'mars'),
    ],
)
def test_format_name(request, reader, name):
    reader = request.getfixturevalue(reader)
    assert reader.format_name == name


@pytest.mark.parametrize(
    'reader, formats',
    [
        ('shiftx2reader', ['.cs', '.txt']),
        ('spartareader', ['.txt']),
        ('nefreader', ['.nef']),
        ('nmrpipereader', ['.tab']),
        ('ccpnreader', ['.txt']),
        ('sparkyreader', ['.txt']),
        ('snapsreader', ['.txt', '.tab']),
        ('xeasyreader', ['.prot', '.txt']),
        ('marsreader', ['.txt', '.tab']),
    ],
)
def test_file_extensions(request, reader, formats):
    reader = request.getfixturevalue(reader)
    assert reader.file_extensions == formats


"""
This needs to be updated because can_read needs to be updated for everything - it should
go into the guts of the file to identify if it can actually read the file rather than
guessing from extensions. 

Subsequently, these tests then need to be ran with real files. 
"""


@pytest.mark.parametrize(
    'filename, expected',
    [
        ('test.cs', True),
        ('test.txt', True),
        ('test.CS', True),
        ('test.TXT', True),
        ('test.csv', False),
        ('test.json', False),
        ('test', False),
        ('test.nef', False),
    ],
)
def test_can_read_shiftx2(shiftx2reader, filename, expected):
    assert shiftx2reader.can_read(Path(filename)) is expected


@pytest.mark.parametrize(
    'reader, file',
    [('shiftx2reader', 'shiftx2_file'), ('spartareader', 'sparta_file')],
)
def test_read_observed_shifts_prediction_formats(request, reader, file):
    """ShiftX2 does not contain observed shifts."""
    reader = request.getfixturevalue(reader)
    file = request.getfixturevalue(file)
    filename, _ = file
    assert reader.read_observed_shifts(filename) is None


@pytest.mark.parametrize(
    'reader, file',
    [
        ('nefreader', 'nef_file_obs'),
        ('nmrpipereader', 'nmrpipe_file'),
        ('ccpnreader', 'ccpn_file'),
        ('sparkyreader', 'sparky_file'),
        ('snapsreader', 'snaps_file'),
        ('xeasyreader', 'xeasy_file'),
        ('marsreader', 'mars_file'),
    ],
)
def test_read_observed_shifts_positive(request, reader, file):
    reader = request.getfixturevalue(reader)
    file = request.getfixturevalue(file)
    filename, expected_result = file
    result = reader.read_observed_shifts(filename)

    print('Result')
    print(result)
    print('Expected')
    print(expected_result)

    assert_frame_equal(result, expected_result)

@pytest.mark.parametrize(
    'reader, file',
    [
        ('shiftx2reader', 'shiftx2_file'),
        ('spartareader', 'sparta_file'),
    ],
)
def test_read_observed_shifts_negative(request, reader, file):
    """None of these formats have observed shifts"""
    reader = request.getfixturevalue(reader)
    file = request.getfixturevalue(file)
    file_path, _ = file
    result = reader.read_observed_shifts(file_path)
    assert result == None

@pytest.mark.parametrize(
    'reader, file',
    [
        ('shiftx2reader', 'shiftx2_file'),
        ('spartareader', 'sparta_file'),
        ('nefreader', 'nef_file_pred'),
    ],
)
def test_read_predicted_shifts_positive(request, reader, file):
    """Test reading and renaming prediction columns."""
    file = request.getfixturevalue(file)
    reader = request.getfixturevalue(reader)
    file_path, expected_result = file
    result = reader.read_predicted_shifts(file_path)

    assert_frame_equal(result, expected_result)

@pytest.mark.parametrize(
    'reader, file',
    [
        ('nmrpipereader', 'nmrpipe_file'),
        ('ccpnreader', 'ccpn_file'),
        ('sparkyreader', 'sparky_file'),
        ('snapsreader', 'snaps_file'),
        ('xeasyreader', 'xeasy_file'),
        ('marsreader', 'mars_file'),
    ],
)
def test_read_predicted_shifts_negative(request, reader, file):
    """None of these formats have predicted shifts"""
    reader = request.getfixturevalue(reader)
    file = request.getfixturevalue(file)
    file_path, _ = file
    result = reader.read_predicted_shifts(file_path)
    assert result == None


def test_read_predicted_shifts_single_chain(shiftx2reader, tmp_path):
    """Test that a single CHAIN column is removed. Shiftx2 only concern."""
    file_path = tmp_path / 'single_chain.cs'

    pd.DataFrame(
        {
            'CHAIN': ['A', 'A'],
            'NUM': [1, 2],
            'RES': ['ALA', 'GLY'],
            'ATOMNAME': ['CA', 'CA'],
            'SHIFT': [52.1, 45.2],
        }
    ).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert 'CHAIN' not in result.columns
    assert list(result.columns) == ['Res_N', 'Res_type', 'Atom_type', 'Shift']
    assert len(result) == 2


def test_read_predicted_shifts_multiple_chains(
    shiftx2reader, tmp_path, capsys
):
    """
    Test multiple chains are merged and a warning is logged.
    Logging removed temporarily in favour for print.
    Shiftx2 only concern.
    """
    file_path = tmp_path / 'multiple_chains.cs'

    pd.DataFrame(
        {
            'CHAIN': ['A', 'B'],
            'NUM': [1, 1],
            'RES': ['ALA', 'GLY'],
            'ATOMNAME': ['CA', 'CA'],
            'SHIFT': [52.1, 45.2],
        }
    ).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    captured = capsys.readouterr()

    assert 'CHAIN' not in result.columns
    assert len(result) == 2
    assert (
        'Chain identifier dropped' in captured.out
        or 'Chain identifier dropped' in captured.err
    )


def test_read_predicted_shifts_extra_columns(shiftx2reader, tmp_path):
    """Test that unexpected columns are removed."""
    file_path = tmp_path / 'extra_columns.cs'

    pd.DataFrame(
        {
            'NUM': [1],
            'RES': ['ALA'],
            'ATOMNAME': ['CA'],
            'SHIFT': [52.1],
            'EXTRA': ['unused'],
        }
    ).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert list(result.columns) == ['Res_N', 'Res_type', 'Atom_type', 'Shift']
    assert 'EXTRA' not in result.columns


def test_read_predicted_shifts_missing_column(shiftx2reader, tmp_path):
    """Test missing columns are filled with NaN."""
    file_path = tmp_path / 'missing_column.cs'

    pd.DataFrame(
        {
            'NUM': [1],
            'RES': ['ALA'],
            'ATOMNAME': ['CA'],
        }
    ).to_csv(file_path, index=False)

    result = shiftx2reader.read_predicted_shifts(file_path)

    assert pd.isna(result.loc[0, 'Shift'])
    assert list(result.columns) == ['Res_N', 'Res_type', 'Atom_type', 'Shift']


@pytest.mark.parametrize(
    'reader',
    ['shiftx2reader', 'spartareader', 'nefreader'],
)
def test_read_predicted_shifts_nonexistent_file(request, reader, tmp_path):
    """Test that a missing file raises FileNotFoundError."""
    file_path = tmp_path / 'nonexistent'
    reader = request.getfixturevalue(reader)

    with pytest.raises(Exception):
        reader.read_predicted_shifts(file_path)


@pytest.mark.parametrize(
    'reader',
    ['shiftx2reader', 'spartareader', 'nefreader'],
)
def test_read_predicted_shifts_empty_file(request, reader, tmp_path):
    file_path = tmp_path / 'empty.cs'
    file_path.touch()
    reader = request.getfixturevalue(reader)

    with pytest.raises(Exception):
        reader.read_predicted_shifts(file_path)
