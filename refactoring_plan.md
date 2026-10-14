# SNAPS Refactoring Plan

Modernizing SNAPS architecture with separation of concerns, modular design, and pipeline-based processing.

## Table of Contents

1. [Goals & Principles](#goals--principles)
2. [Current Architecture](#current-architecture)
3. [Proposed Architecture](#proposed-architecture)
4. [Convention-Based CLI](#convention-based-cli)
5. [Data Flow Pipeline](#data-flow-pipeline)
6. [Component Interfaces](#component-interfaces)
7. [Configuration System](#configuration-system)
8. [Pipeline Example](#pipeline-example)
9. [Web Service Considerations](#web-service-considerations)
10. [Migration Path](#migration-path)

## Goals & Principles

### Core Principles

- **Separation of Concerns**: Each component has a single, well-defined responsibility with clear interfaces
- **Pipeline Architecture**: Data flows through discrete stages: import → process → analyze → output
- **Pluggable Formats**: Format readers as plugins; add new formats without touching core logic
- **Convention over Configuration**: Smart defaults minimize required CLI arguments
- **Modern CLI**: Typer-based interface with intuitive conventions
- **Web Service Ready**: Core pipeline works as library; CLI and web service are thin wrappers
- **Testable**: Pure functions, constructor injection, protocols, and structured data for easy testing
- **Self-Registration**: Components (strategies, readers) register themselves automatically via decorators

## Current Architecture

**Problems with current design:**

```
┌─────────────────────────────────────────────────┐
│            SNAPS.py (Monolithic)                │
│                                                 │
│  ┌───────────────────────────────────────────┐ │  ❌ Config scattered
│  │ Argument Parsing (argparse)               │ │  ❌ Many required args
│  └───────────────────────────────────────────┘ │  ❌ Monolithic importer
│              ↓                                  │  ❌ Mixed concerns
│  ┌───────────────────────────────────────────┐ │  ❌ Hard to test
│  │ Importer (all formats in one class)       │ │  ❌ Not web-ready
│  └───────────────────────────────────────────┘ │
│              ↓  ← tight coupling →             │
│  ┌───────────────────────────────────────────┐ │
│  │ Assigner (calc + output mixed)            │ │
│  └───────────────────────────────────────────┘ │
│              ↓                                  │
│  ┌───────────────────────────────────────────┐ │
│  │ Output Logic (NEF/SNAPS in main)          │ │
│  └───────────────────────────────────────────┘ │
│              ↓                                  │
│  ┌───────────────────────────────────────────┐ │
│  │ Plotting (embedded in flow)               │ │
│  └───────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Proposed Architecture

**Modular design with clear separation:**

```
┌──────────────┐     ┌─────────────┐       ┌──────────────┐       ┌────────────┐
│Format Readers│────→│  Importer   │────→  │  ShiftData   │────→  │ Strategies │
│  (plugins)   │uses │(orchestrate)│returns│  dataclass   │process│ (pattern)  │
│              │     │             │       │              │       │            │
│ • NEFReader  │     │ Auto-detect │       │ • obs_shifts │       │ • Iterative│
│ • ShiftX2... │     │   formats   │       │ • pred_shifts│       │ • Direct   │
│ • Sparky...  │     │             │       │ • sequence   │       └────────────┘
└──────────────┘     └─────────────┘       │ • restraints │     
                                           └──────────────┘             │ returns
                                                                        ↓
┌──────────────┐     ┌──────────────┐                            ┌────────────────┐
│   Plotters   │←─── │   Outputs    │←───────────────────────────│AssignmentResult│
│              │     │              │                            │   dataclass    │
│ • HSQCPlotter│     │ • NEFWriter  │      ┌──────────────┐      │                │
│ • StripPlot  │     │ • SNAPSWriter│←─────│  Analyzers   │←─────│ • assignments  │
└──────────────┘     │ • PeakList   │      │              │      │ • log_prob     │
                     │ • ShiftList  │      │ • Consistency│      └────────────────┘
                     └──────────────┘      │ • Quality    │
                                           └──────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    CLI (Typer) / Web Service                │
│                   thin wrappers around pipeline             │
└─────────────────────────────────────────────────────────────┘
```

## Convention-Based CLI

### File and Frame Naming Conventions

**Minimize required arguments through smart conventions:**

#### NEF Frame Name Conventions

the sequence is in `nef_molecular_system`, shifts are in `nef_chemical_shift_list_default` i-1 shifts are defined by 
`A.@93-1..CA` which in this example defines the chemical shift for the CA in the residue before the anonymous residue 
93. If there is more than one chain in the molecular system we currently give up unless a chain is specified in 
future we may iterate. predicted shifts unless specified on the command line are in `nef_chemical_shift_list_predicted`
For files without a moleculare system the molecular system is derived from the predicted shifts and we will
either assume all missing residues are PRO or require pro to be present with shifts defined as ****. In `nef` files 
prolines with missing shift have the value '.'
94. 
#### File Naming and Detection Conventions

1. **Single file mode:** If one file is given and it has the suffix `.nef`, assume it's NEF containing both observed and 
predicted chemcial shifts following the [nef frame conventions described below](NEF_frame_specification_syntax).
Otherwise its and error observed and predicted shift files must be specified.
2. 
   ```bash
   snaps data.nef
   # Looks for frames: observed, predicted (or conventions above)
   ```

2. **Two file mode:** First is observed, second is predicted and can be any format that supported
When the first frame is a `nef` file ':' can be a stand in for `nef_chemical_shift_list_predicted`
and `:my_predictions` takes the fist `nef` file and looks for the frame `nef_chemical_shift_list_my_predictions`

   ```bash
   snaps obs.nef pred.nef
   # obs.nef contains observed, pred.nef contains predicted
   snaps my_protein.nef:observed :predicted
   # my_protein.nef contains both observed and predicted in the frames called observed and predicted
   ```

3. **Format auto-detection from extension:**

detection from extensions can only be partially correct [for example NMRPipe and MARS use .tab files
but the mars.tab file is actually a sparky table file]. Most probably the best approach is to have file
identifiers that identify the first lines and also the ability to specify file formats via options.

   - `.nef` → NEF format
   - `.cs` → ShiftX2 format
   - `.list` → Sparky format
   - `.tab` → NMRPipe format
   - `.tab` → MARS format

> Note: the intention is to use NEF-Pipelines to injest the various file formats into internal nef 
> entries in PyNMRStar and then convert to pandas dataframes. NEF-Pipelines input formats will be 
> heavily tested. 

4. **Output conventions:**

outputs will be set by templates so that inputs define outputs. Output templates can be modified on 
the command line [use str.format not f strings for safety]
   - Default output: `{input_stem}_assigned.{input_ext}`
   - Default log: `{input_stem}_snaps.log`
   - Default plots: `{input_stem}_hsqc.html`, `{input_stem}_strips.html`

5. **NEF frame specification syntax:**
A frame name can be appended to the filename separated by a : e.g. `xyz.nef:my_shifts` -> the frame 
`nef_chemical_shift_list_my_shifts` in the file `xyz.nef`. A : on its own is the dfault frame for the
information resuired `nef_chemical_shift_frame_default` - observed shifts `nef_chemical_shift_frame_predicted`
for predicted shifts

   ```bash
   # Explicit frame names using colon separator
   snaps  data.nef:my_observed :my_predicted
   
   # Mix file and frame
   snaps  obs_file.nef:observed pred_file.nef:predicted
   
   # Default frames used if not specified
   snaps data.nef  # uses 'observed' and 'predicted' frames
   ```

### Minimal CLI Examples

```bash
# Absolute minimum - everything auto-detected
snaps data.nef
# → reads data.nef:observed and data.nef:predicted
# → writes data_assigned.nef
# → uses default config

# Two-file mode
snaps obs.nef pred.nef
# → reads obs.nef:observed and pred.nef:predicted
# → writes obs_assigned.nef

# Explicit frame names when needed
snaps data.nef:my_obs :my_pred

# Override just what you need
snaps data.nef --strategy direct --output results.nef

# With custom config
snaps data.nef -c my_config.yaml

# Full control when needed
snaps obs.nef:frame1 pred.nef:frame2 \
    --output results.nef \
    --config my_config.yaml \
    --strategy iterative \
    --plots  # Generate all default plots
```

### Input/Output Dataclasses

Before the CLI, define structured parameter containers:

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class InputSpec:
    """Input file specifications"""
    obs_file: Path
    pred_file: Path
    obs_format: Optional[str] = None
    pred_format: Optional[str] = None
    obs_frame: Optional[str] = None
    pred_frame: Optional[str] = None
    obs_chain: str = 'A'
    pred_chain: str = 'A'
    pred_offset: int = 0
    aa_restraints_file: Optional[Path] = None
    aa_restraints_frame: Optional[str] = None


@dataclass
class OutputSpec:
    """Output specifications"""
    output_file: Path
    output_format: str
    log_file: Optional[Path] = None
    hsqc_plot: Optional[Path] = None
    strip_plot: Optional[Path] = None
    analysis_level: str = 'standard'
```

### CLI Implementation with Conventions

```python
from pathlib import Path
from typing import Optional, Tuple, Dict, Type
import typer
from enum import Enum

class StrategyRegistry:
    """Registry for self-registering assignment strategies"""
    _strategies: Dict[str, Type[AssignmentStrategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""
        def decorator(strategy_class):
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[AssignmentStrategy]:
        """Get strategy class by name"""
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        return cls._strategies[name]
    
    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered strategy names"""
        return list(cls._strategies.keys())


# Strategies auto-build the enum
def _build_strategy_enum():
    """Build enum from registered strategies"""
    return Enum('Strategy', {name: name for name in StrategyRegistry.list_names()})


class OutputFormat(str, Enum):
    """Available output formats"""
    auto = "auto"  # Auto-detect from extension
    nef = "nef"
    snaps = "snaps"


class AnalysisLevel(str, Enum):
    """Analysis detail level for output"""
    minimal = "minimal"   # Just assignments
    standard = "standard" # + consistency metrics  
    full = "full"        # + all matrices and diagnostics

app = typer.Typer(
    help="SNAPS - Simple NMR Assignments from Predicted Shifts",
    add_completion=True
)


def parse_file_spec(spec: str) -> Tuple[Path, Optional[str], Optional[str]]:
    """
    Parse file specification with optional frame names.
    
    Format: path/to/file.nef[:obs_frame[:pred_frame]]
    
    Returns: (file_path, obs_frame_name, pred_frame_name)
    """
    parts = spec.split(':')
    file_path = Path(parts[0])
    obs_frame = parts[1] if len(parts) > 1 else None
    pred_frame = parts[2] if len(parts) > 2 else None
    return file_path, obs_frame, pred_frame


def auto_output_path(input_path: Path, suffix: str = "assigned") -> Path:
    """Generate default output path from input path"""
    return input_path.parent / f"{input_path.stem}_{suffix}{input_path.suffix}"


@app.command()
def assign(
    # Input files - minimal required args
    input_spec: str = typer.Argument(
        ...,
        help="Input file. For NEF: 'file.nef[:obs_frame[:pred_frame]]'. "
             "Single file mode uses conventional frame names."
    ),
    pred_spec: Optional[str] = typer.Argument(
        None,
        help="Predicted shifts file (optional, use for two-file mode). "
             "Format: 'file.ext[:frame_name]'"
    ),
    
    # Output - optional with smart default
    output: Optional[Path] = typer.Option(
        None, "-o", "--out",
        help="Output file (default: {input}_assigned.{ext})"
    ),
    
    output_format: OutputFormat = typer.Option(
        OutputFormat.auto, "--output-format",
        help="Output format (auto: detect from extension)"
    ),
    
    analysis_level: AnalysisLevel = typer.Option(
        AnalysisLevel.standard, "--analysis",
        help="Analysis detail level in output"
    ),
    
    # Info flag - show file info and exit
    info: bool = typer.Option(
        False, "--info",
        help="Show file info and exit (ignores all other options)"
    ),
    
    # Config - optional
    config: Optional[Path] = typer.Option(
        None, "-c", "--config",
        help="Config file (default: built-in defaults)"
    ),
    
    # Strategy - optional with smart default (built from registry)
    strategy: str = typer.Option(
        "iterative", "--strategy",
        help="Assignment strategy (use --info to list available)"
    ),
    
    # Common overrides
    seq_threshold: Optional[float] = typer.Option(
        None, "--seq-threshold",
        help="Sequential link threshold (overrides config)"
    ),
        
    no_iterate: bool = typer.Option(
        False, "--no-iterate",
        help="Disable iterative consistency checking"
    ),
    
    # Optional inputs
    aa_restraints: Optional[str] = typer.Option(
        None, "--aa-restraints",
        help="AA restraints file. Format: 'file.ext[:frame_name]'"
    ),
    
    # Chain and offset - rarely needed
    obs_chain: str = typer.Option('A', help="Observed shifts chain"),
    pred_chain: str = typer.Option('A', help="Predicted shifts chain"),
    pred_offset: int = typer.Option(0, help="Predicted shifts offset"),
    
    # Logging and plots
    log: Optional[Path] = typer.Option(
        None, "-l", "--log",
        help="Log file (default: {input}_snaps.log or stdout)"
    ),
    plots: bool = typer.Option(
        False, "--plots",
        help="Generate all default plots (HSQC and strips)"
    ),
    hsqc_plot: Optional[Path] = typer.Option(
        None, "--hsqc",
        help="HSQC plot output (default: {input}_hsqc.html if --plots)"
    ),
    strip_plot: Optional[Path] = typer.Option(
        None, "--strips",
        help="Strip plot output (default: {input}_strips.html if --plots)"
    ),
    
    # Verbosity (quiet by default)
    verbosity: int = typer.Option(
        0, "-v", "--verbose", count=True,
        help="Verbosity level: 0=quiet (errors only), 1=info (-v), 2=debug (-vv)"
    ),
):
    """
    Assign NMR chemical shifts using predicted shifts.
    
    EXAMPLES:
    
      # Minimal - single NEF file with conventional frames
      snaps assign data.nef
      
      # Two-file mode
      snaps assign obs.nef pred.nef
      
      # Explicit frame names
      snaps assign data.nef:my_observed:my_predicted
      
      # Mix explicit and conventional
      snaps assign obs.nef:frame1 pred.nef
      
      # With plots and custom config
      snaps assign data.nef -c config.yaml --plots
      
      # Override specific parameters
      snaps assign data.nef --strategy direct --seq-threshold 0.5
    
    CONVENTIONS:
    
      NEF frames auto-detected in order:
        Observed: 'observed', 'nef_chemical_shift_list_observed', first list
        Predicted: 'predicted', 'nef_chemical_shift_list_predicted', second list
      
      Output defaults: {input}_assigned.{ext}
      Log defaults: {input}_snaps.log (or stdout if none specified)
      Plot defaults: {input}_hsqc.html, {input}_strips.html
    """
    
    # Handle --info flag first (show info and exit)
    if info:
        show_file_info(input_spec, pred_spec)
        raise typer.Exit()
    
    # Parse input specifications
    input_path, obs_frame, pred_frame = parse_file_spec(input_spec)
    
    # Determine mode (single file or two file)
    if pred_spec is None:
        # Single file mode - both obs and pred in same file
        obs_file = pred_file = input_path
        obs_frame_name = obs_frame or "observed"  # Use convention
        pred_frame_name = pred_frame or "predicted"  # Use convention
    else:
        # Two file mode
        obs_file = input_path
        pred_path, pred_frame_override = parse_file_spec(pred_spec)[:2]
        pred_file = pred_path
        obs_frame_name = obs_frame or "observed"
        pred_frame_name = pred_frame_override or pred_frame or "predicted"
    
    # Auto-detect formats from extensions if not specified
    obs_format = _detect_format(obs_file)
    pred_format = _detect_format(pred_file)
    
    # Apply output conventions
    if output is None:
        output = auto_output_path(input_path, "assigned")
    
    # Auto-detect output format if needed
    if output_format == OutputFormat.auto:
        output_format = _detect_format(output)  # Detect from output extension
    else:
        output_format = output_format.value
    
    # Apply log conventions (only if verbose or specified)
    if log is None and verbosity > 0:
        log = auto_output_path(input_path, "snaps").with_suffix('.log')
    
    # Apply plot conventions
    if plots:
        if hsqc_plot is None:
            hsqc_plot = auto_output_path(input_path, "hsqc").with_suffix('.html')
        if strip_plot is None:
            strip_plot = auto_output_path(input_path, "strips").with_suffix('.html')
    
    # Parse AA restraints if provided
    aa_restraints_file = None
    aa_restraints_frame = None
    if aa_restraints:
        aa_restraints_file, aa_restraints_frame = parse_file_spec(aa_restraints)[:2]
    
    # Validate strategy exists
    if strategy not in StrategyRegistry.list_names():
        available = ', '.join(StrategyRegistry.list_names())
        typer.secho(f"✗ Unknown strategy: {strategy}", fg=typer.colors.RED, err=True)
        typer.echo(f"Available strategies: {available}")
        raise typer.Exit(code=1)
    
    # Build config with hierarchy
    snaps_config = build_config(
        config_file=config,
        cli_overrides={
            'strategy': strategy,
            'seq_link_threshold': seq_threshold,
            'iterate_until_consistent': not no_iterate,
        }
    )
    
    # Package parameters into dataclasses
    input_spec = InputSpec(
        obs_file=obs_file,
        pred_file=pred_file,
        obs_format=obs_format,
        pred_format=pred_format,
        obs_frame=obs_frame_name,
        pred_frame=pred_frame_name,
        obs_chain=obs_chain,
        pred_chain=pred_chain,
        pred_offset=pred_offset,
        aa_restraints_file=aa_restraints_file,
        aa_restraints_frame=aa_restraints_frame,
    )
    
    output_spec = OutputSpec(
        output_file=output,
        output_format=output_format,
        log_file=log,
        hsqc_plot=hsqc_plot,
        strip_plot=strip_plot,
        analysis_level=analysis_level,
    )
    
    # Show what will be done (if verbose)
    if verbosity > 0:
        typer.echo(f"Input: {obs_file} (frame: {obs_frame_name})")
        if pred_file != obs_file:
            typer.echo(f"Predicted: {pred_file} (frame: {pred_frame_name})")
        typer.echo(f"Output: {output} ({output_format})")
        typer.echo(f"Strategy: {strategy}")
    
    # Run the pipeline
    try:
        run_pipeline(
            input_spec=input_spec,
            output_spec=output_spec,
            config=snaps_config,
            verbosity=verbosity,
        )
        
        if verbosity >= 0:  # Always show success unless truly silent
            typer.secho(f"✓ Assignment complete: {output}", fg=typer.colors.GREEN)
            
    except Exception as e:
        typer.secho(f"✗ Error: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


def _detect_format(file_path: Path) -> str:
    """Auto-detect format from file extension"""
    ext_to_format = {
        '.nef': 'nef',
        '.cs': 'shiftx2',
        '.list': 'sparky',
        '.peaks': 'nmrpipe',
        '.prot': 'mars',
        '.tab': 'xeasy',
    }
    return ext_to_format.get(file_path.suffix.lower(), 'nef')


def show_file_info(input_spec: str, pred_spec: Optional[str] = None):
    """
    Show information about input files and exit.
    Also shows available strategies and formats.
    """
    # Parse file specs
    input_path, _, _ = parse_file_spec(input_spec)
    
    # Show file info
    if input_path.suffix.lower() == '.nef':
        inspector = NEFInspector()
        info = inspector.inspect(input_path)
        
        typer.echo(f"\nNEF File: {input_path}")
        typer.echo(f"\nShift Lists:")
        for frame in info['shift_lists']:
            typer.echo(f"  - {frame['name']} ({frame['shifts']} shifts)")
        
        typer.echo(f"\nSequence:")
        if info['sequence']:
            typer.echo(f"  - {info['sequence']['length']} residues")
            typer.echo(f"  - Chain: {info['sequence']['chain']}")
        
        if info['restraints']:
            typer.echo(f"\nRestraints:")
            for r in info['restraints']:
                typer.echo(f"  - {r}")
    else:
        typer.echo(f"\nFile: {input_path}")
        typer.echo(f"Format: {_detect_format(input_path)}")
    
    # Show pred file if specified
    if pred_spec:
        pred_path, _, _ = parse_file_spec(pred_spec)
        typer.echo(f"\nPredicted shifts: {pred_path}")
        typer.echo(f"Format: {_detect_format(pred_path)}")
    
    # Show available strategies
    typer.echo(f"\nAvailable Strategies:")
    for name in StrategyRegistry.list_names():
        typer.echo(f"  - {name}")
    
    # Show output formats
    typer.echo(f"\nOutput Formats:")
    for fmt in OutputFormat:
        typer.echo(f"  - {fmt.value}")


if __name__ == "__main__":
    app()
```

### NEF Frame Auto-Detection Implementation

```python
from typing import List, Dict, Optional
import pynmrstar


class NEFInspector:
    """Inspect NEF files to find conventional frame names"""
    
    OBSERVED_CONVENTIONS = [
        'observed',
        'nef_chemical_shift_list_observed',
        'nef_chemical_shift_list_1',
    ]
    
    PREDICTED_CONVENTIONS = [
        'predicted',
        'nef_chemical_shift_list_predicted',
        'nef_chemical_shift_list_2',
    ]
    
    def inspect(self, nef_file: Path) -> Dict:
        """Inspect NEF file and return available frames"""
        entry = pynmrstar.Entry.from_file(str(nef_file))
        
        shift_lists = []
        for sf in entry.get_saveframes_by_category('nef_chemical_shift_list'):
            shift_lists.append({
                'name': sf.name,
                'shifts': len(sf.loops[0].data) if sf.loops else 0,
            })
        
        sequence_info = None
        seq_sf = entry.get_saveframes_by_category('nef_molecular_system')
        if seq_sf:
            sequence_info = {
                'length': len(seq_sf[0].loops[0].data) if seq_sf[0].loops else 0,
                'chain': seq_sf[0].loops[0].data[0][0] if seq_sf[0].loops else 'A',
            }
        
        return {
            'shift_lists': shift_lists,
            'sequence': sequence_info,
            'restraints': [sf.name for sf in entry.get_saveframes_by_category('nef_nmr_restraints')],
        }
    
    def find_frame(self, nef_file: Path, frame_type: str) -> Optional[str]:
        """
        Find frame by type using conventions.
        
        Args:
            nef_file: Path to NEF file
            frame_type: 'observed' or 'predicted'
        
        Returns:
            Frame name or None if not found
        """
        entry = pynmrstar.Entry.from_file(str(nef_file))
        shift_lists = [sf.name for sf in entry.get_saveframes_by_category('nef_chemical_shift_list')]
        
        conventions = (
            self.OBSERVED_CONVENTIONS if frame_type == 'observed'
            else self.PREDICTED_CONVENTIONS
        )
        
        # Try each convention in order
        for convention in conventions:
            if convention in shift_lists:
                return convention
        
        # Fallback: use first or second shift list
        if shift_lists:
            if frame_type == 'observed' and len(shift_lists) >= 1:
                return shift_lists[0]
            elif frame_type == 'predicted' and len(shift_lists) >= 2:
                return shift_lists[1]
        
        return None
```

## Data Flow Pipeline

Pipeline stages with structured data flow:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  1. IMPORT  │───→│ 2. PROCESS  │───→│  3. ANALYZE │───→│  4. OUTPUT  │───→│  5. RETURN  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                                                                                      
Input:                Strategy:           Analyzers:         Text outputs:      For web:
• NEF files           • Iterative         • Consistency      • NEF format        • JSON result
• ShiftX2             • Direct            • Quality          • SNAPS format      • DataFrames
• Restraints          • Bayesian          • Confidence       
                                                             Plots:              For CLI:
Output:               Calculates:         Adds to result:    • HSQC              • Files written
ShiftData             • Log prob matrix   • Metadata         • Strip plots       • Exit code
→ DataFrames          • Assignments       • Statistics       
                                                             Logs:
                      Output:                                • Runtime info
                      AssignmentResult                       • Statistics
```

## Component Interfaces

### 1. Data Classes

Structured containers for data passed between pipeline stages:

```python
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from datetime import datetime
import pandas as pd


@dataclass
class ShiftData:
    """Container for all imported shift data"""
    obs_shifts: pd.DataFrame      # Observed shifts (i, i-1)
    pred_shifts: pd.DataFrame      # Predicted shifts (i, i-1)
    sequence: pd.DataFrame         # Protein sequence
    aa_restraints: Optional[pd.DataFrame] = None  # Amino acid type restraints
    unlabel_restraints: Optional[pd.DataFrame] = None  # Unlabelling info
    metadata: Dict[str, Any] = field(default_factory=dict)  # Sources, timestamps
    
    def __post_init__(self):
        """Add timestamp if not present"""
        if 'import_timestamp' not in self.metadata:
            self.metadata['import_timestamp'] = datetime.now().isoformat()


@dataclass
class AssignmentResult:
    """Results from assignment strategy"""
    assignments: pd.DataFrame      # Main assignment table
    log_prob_matrix: Optional[pd.DataFrame] = None
    mismatch_matrix: Optional[pd.DataFrame] = None
    confidence: Optional[pd.DataFrame] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON (web service)"""
        return {
            'assignments': self.assignments.to_dict(orient='records'),
            'metadata': self.metadata,
        }


@dataclass
class AnalysisResult:
    """Results from analysis"""
    consistency_metrics: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    enriched_assignments: pd.DataFrame
    
    def summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        return {
            'total_assignments': len(self.enriched_assignments),
            'consistency': self.consistency_metrics,
            'quality': self.quality_metrics,
        }
```

### 2. Format Reader Protocol

Protocol for pluggable format readers:

```python
from typing import Protocol, Optional
from pathlib import Path


class FormatReader(Protocol):
    """Protocol that all format readers must implement"""
    
    @property
    def format_name(self) -> str:
        """Return format identifier (e.g., 'nef', 'shiftx2')"""
        ...
    
    @property
    def file_extensions(self) -> list[str]:
        """Return supported file extensions (e.g., ['.nef'])"""
        ...
    
    def can_read(self, file_path: Path) -> bool:
        """Check if this reader can handle the file"""
        ...
    
    def read_observed_shifts(
        self, 
        file_path: Path,
        frame: Optional[str] = None,
        chain: str = 'A'
    ) -> pd.DataFrame:
        """Read observed chemical shifts"""
        ...
    
    def read_predicted_shifts(
        self,
        file_path: Path,
        frame: Optional[str] = None,
        chain: str = 'A',
        offset: int = 0
    ) -> pd.DataFrame:
        """Read predicted chemical shifts"""
        ...
    
    def read_sequence(
        self, 
        file_path: Path,
        frame: Optional[str] = None
    ) -> pd.DataFrame:
        """Read protein sequence if available in format"""
        ...
    
    def read_aa_restraints(
        self,
        file_path: Path,
        frame: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Read amino acid type restraints if available"""
        ...


# Example implementation
class NEFReader:
    """NEF format reader with frame name conventions"""
    
    OBSERVED_FRAMES = ['observed', 'nef_chemical_shift_list_observed']
    PREDICTED_FRAMES = ['predicted', 'nef_chemical_shift_list_predicted']
    
    @property
    def format_name(self) -> str:
        return "nef"
    
    @property
    def file_extensions(self) -> list[str]:
        return ['.nef']
    
    def can_read(self, file_path: Path) -> bool:
        return file_path.suffix.lower() == '.nef'
    
    def _find_frame(
        self, 
        entry: pynmrstar.Entry, 
        frame_name: Optional[str],
        conventions: list[str]
    ) -> str:
        """Find frame by name or convention"""
        if frame_name:
            return frame_name
        
        # Try conventions in order
        shift_lists = [sf.name for sf in entry.get_saveframes_by_category('nef_chemical_shift_list')]
        for convention in conventions:
            if convention in shift_lists:
                return convention
        
        # Fallback to first available
        if shift_lists:
            return shift_lists[0]
        
        raise ValueError(f"No chemical shift list found in {entry}")
    
    def read_observed_shifts(
        self, 
        file_path: Path,
        frame: Optional[str] = None,
        chain: str = 'A'
    ) -> pd.DataFrame:
        """Read observed shifts from NEF file"""
        entry = pynmrstar.Entry.from_file(str(file_path))
        frame_name = self._find_frame(entry, frame, self.OBSERVED_FRAMES)
        
        # Implementation using pynmrstar
        saveframe = entry.get_saveframe_by_name(frame_name)
        # ... convert to standard DataFrame format
        
        return df
    
    def read_predicted_shifts(
        self,
        file_path: Path,
        frame: Optional[str] = None,
        chain: str = 'A',
        offset: int = 0
    ) -> pd.DataFrame:
        """Read predicted shifts from NEF file"""
        entry = pynmrstar.Entry.from_file(str(file_path))
        frame_name = self._find_frame(entry, frame, self.PREDICTED_FRAMES)
        
        # Implementation
        saveframe = entry.get_saveframe_by_name(frame_name)
        # ... convert and apply offset
        
        return df


class ShiftX2Reader:
    """ShiftX2 format reader"""
    
    @property
    def format_name(self) -> str:
        return "shiftx2"
    
    @property
    def file_extensions(self) -> list[str]:
        return ['.cs', '.txt']
    
    def can_read(self, file_path: Path) -> bool:
        # Check extension and/or file content
        return file_path.suffix.lower() in ['.cs', '.txt']
    
    # ... implement other methods
```

### 3. Importer

Orchestrates format readers and returns structured data:

```python
from typing import List, Optional
from pathlib import Path


class SNAPSImporter:
    """Orchestrates format readers to import all required data"""
    
    def __init__(self, readers: Optional[List[FormatReader]] = None):
        self.readers = readers or self._default_readers()
        self._reader_by_format = {r.format_name: r for r in self.readers}
        self._reader_by_ext = {}
        for reader in self.readers:
            for ext in reader.file_extensions:
                self._reader_by_ext[ext] = reader
    
    def _default_readers(self) -> List[FormatReader]:
        """Load all available format readers"""
        return [
            NEFReader(),
            ShiftX2Reader(),
            SparkyReader(),
            MarsReader(),
            XeasyReader(),
            NMRPipeReader(),
        ]
    
    def _get_reader(
        self, 
        file_path: Path, 
        format_hint: Optional[str] = None
    ) -> FormatReader:
        """Find appropriate reader for file"""
        # Try format hint first
        if format_hint and format_hint in self._reader_by_format:
            return self._reader_by_format[format_hint]
        
        # Try extension mapping
        ext = file_path.suffix.lower()
        if ext in self._reader_by_ext:
            return self._reader_by_ext[ext]
        
        # Try each reader's can_read method
        for reader in self.readers:
            if reader.can_read(file_path):
                return reader
        
        raise ValueError(f"No reader found for {file_path}")
    
    def import_data(
        self,
        obs_file: Path,
        pred_file: Path,
        obs_format: Optional[str] = None,
        pred_format: Optional[str] = None,
        obs_frame: Optional[str] = None,
        pred_frame: Optional[str] = None,
        obs_chain: str = 'A',
        pred_chain: str = 'A',
        pred_offset: int = 0,
        aa_restraints_file: Optional[Path] = None,
        aa_restraints_frame: Optional[str] = None,
    ) -> ShiftData:
        """Import all data and return structured container"""
        
        # Get appropriate readers
        obs_reader = self._get_reader(obs_file, obs_format)
        pred_reader = self._get_reader(pred_file, pred_format)
        
        # Import observed shifts
        obs_shifts = obs_reader.read_observed_shifts(
            obs_file, 
            frame=obs_frame,
            chain=obs_chain
        )
        
        # Import predicted shifts
        pred_shifts = pred_reader.read_predicted_shifts(
            pred_file,
            frame=pred_frame,
            chain=pred_chain,
            offset=pred_offset
        )
        
        # Import sequence
        sequence = pred_reader.read_sequence(pred_file)
        
        # Import AA restraints if provided
        aa_restraints = None
        if aa_restraints_file:
            restraint_reader = self._get_reader(aa_restraints_file)
            aa_restraints = restraint_reader.read_aa_restraints(
                aa_restraints_file,
                frame=aa_restraints_frame
            )
        
        return ShiftData(
            obs_shifts=obs_shifts,
            pred_shifts=pred_shifts,
            sequence=sequence,
            aa_restraints=aa_restraints,
            metadata={
                'obs_file': str(obs_file),
                'pred_file': str(pred_file),
                'obs_format': obs_reader.format_name,
                'pred_format': pred_reader.format_name,
                'obs_frame': obs_frame,
                'pred_frame': pred_frame,
            }
        )
```

### 4. Assignment Strategy

Strategy pattern for different assignment algorithms:

```python
from abc import ABC, abstractmethod


class AssignmentStrategy(ABC):
    """Base class for assignment strategies"""
    
    name: str = ""  # Subclasses must set this
    
    def __init__(self, config: dict):
        self.config = config
    
    @abstractmethod
    def assign(self, data: ShiftData) -> AssignmentResult:
        """Perform assignment and return results"""
        pass


# Strategy registry for self-registration
class StrategyRegistry:
    """Registry for self-registering assignment strategies"""
    _strategies: Dict[str, Type[AssignmentStrategy]] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a strategy"""
        def decorator(strategy_class):
            strategy_class.name = name
            cls._strategies[name] = strategy_class
            return strategy_class
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Type[AssignmentStrategy]:
        """Get strategy class by name"""
        if name not in cls._strategies:
            available = ', '.join(cls._strategies.keys())
            raise ValueError(f"Unknown strategy: {name}. Available: {available}")
        return cls._strategies[name]
    
    @classmethod
    def list_names(cls) -> List[str]:
        """List all registered strategy names"""
        return sorted(cls._strategies.keys())
    
    @classmethod
    def create(cls, name: str, config: dict) -> AssignmentStrategy:
        """Factory method to create a strategy instance"""
        strategy_class = cls.get(name)
        return strategy_class(config)


@StrategyRegistry.register('iterative')
class IterativeStrategy(AssignmentStrategy):
    """Iterative consistency-based assignment"""
    
    def assign(self, data: ShiftData) -> AssignmentResult:
        # Calculate probability matrix
        log_prob = self._calc_log_prob_matrix(data)
        
        # Iterate until consistent
        assignments = self._iterate_until_consistent(
            data, 
            log_prob,
            threshold=self.config.get('seq_link_threshold', 0.45)
        )
        
        return AssignmentResult(
            assignments=assignments,
            log_prob_matrix=log_prob,
            metadata={
                'strategy': 'iterative',
                'config': self.config,
            }
        )
    
    def _calc_log_prob_matrix(self, data: ShiftData) -> pd.DataFrame:
        """Calculate log probability matrix"""
        # Implementation from current SNAPS_assigner
        ...
    
    def _iterate_until_consistent(
        self, 
        data: ShiftData, 
        log_prob: pd.DataFrame,
        threshold: float
    ) -> pd.DataFrame:
        """Iterate assignments until sequentially consistent"""
        # Implementation from current SNAPS_assigner
        ...


@StrategyRegistry.register('direct')
class DirectStrategy(AssignmentStrategy):
    """Direct assignment without iteration"""
    
    def assign(self, data: ShiftData) -> AssignmentResult:
        log_prob = self._calc_log_prob_matrix(data)
        assignments = self._assign_from_matrix(log_prob)
        
        return AssignmentResult(
            assignments=assignments,
            log_prob_matrix=log_prob,
            metadata={'strategy': 'direct'}
        )


# Example of adding a new strategy - just add the decorator!
@StrategyRegistry.register('bayesian')
class BayesianStrategy(AssignmentStrategy):
    """Bayesian probabilistic assignment (future)"""
    
    def assign(self, data: ShiftData) -> AssignmentResult:
        # Implementation
        ...


# Factory function (simplified - just delegates to registry)
def create_strategy(strategy_name: str, config: dict) -> AssignmentStrategy:
    """
    Factory function to create assignment strategies.
    Strategies are auto-registered via @StrategyRegistry.register decorator.
    """
    return StrategyRegistry.create(strategy_name, config)
```

(Additional components in same style: Analyzers, Plotters, OutputWriters - see HTML version for full code)

## Configuration System

### Priority Hierarchy

```
    ╱▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔╲
   ╱  CLI Arguments  ╲     ← Highest priority (--strategy direct)
  ╱__________________╲
  ▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔▔
         ↓ overrides
  ┌──────────────────┐
  │ CLI Config File  │     ← Medium priority (-c config.yaml)
  └──────────────────┘
         ↓ overrides
  ┌──────────────────┐
  │ Default Config   │     ← Lowest priority (built-in)
  └──────────────────┘
```

### Implementation

```python
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
import yaml


class SNAPSConfig(BaseModel):
    """Configuration model with validation"""
    
    # Atom settings
    atom_set: list[str] = Field(
        default=['H', 'N', 'HA', 'CA', 'CB', 'C'],
        description="Atom types to use in assignment"
    )
    atom_sd: dict[str, float] = Field(
        default={
            'H': 0.1726, 'N': 1.1169, 'HA': 0.1231,
            'CA': 0.4412, 'CB': 0.5163, 'C': 0.5256
        },
        description="Standard deviations for each atom type"
    )
    
    # Assignment settings
    seq_link_threshold: float = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="Carbon shift threshold for sequential links"
    )
    iterate_until_consistent: bool = Field(
        default=True,
        description="Iterate assignments until sequentially consistent"
    )
    use_ss_class_info: bool = Field(
        default=False,
        description="Use amino acid class restraints"
    )
    delta_correlation: bool = Field(
        default=True,
        description="Account for correlations in prediction errors"
    )
    
    # Strategy
    strategy: str = Field(
        default='iterative',
        description="Assignment strategy: iterative, direct, bayesian"
    )
    
    @classmethod
    def load_default(cls) -> 'SNAPSConfig':
        """Load default configuration"""
        # Built-in defaults
        return cls()
    
    @classmethod
    def load_from_file(cls, config_path: Path) -> 'SNAPSConfig':
        """Load configuration from YAML file"""
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def merge(self, overrides: dict) -> 'SNAPSConfig':
        """Merge with overrides, overrides take priority"""
        merged = self.dict()
        merged.update({k: v for k, v in overrides.items() if v is not None})
        return SNAPSConfig(**merged)


def build_config(
    config_file: Optional[Path] = None,
    cli_overrides: Optional[dict] = None
) -> SNAPSConfig:
    """
    Build configuration with proper priority.
    
    Priority: CLI overrides > config file > defaults
    """
    # Start with defaults
    config = SNAPSConfig.load_default()
    
    # Merge config file if provided
    if config_file and config_file.exists():
        file_config = SNAPSConfig.load_from_file(config_file)
        config = config.merge(file_config.dict())
    
    # Apply CLI overrides
    if cli_overrides:
        config = config.merge(cli_overrides)
    
    return config
```

## Pipeline Example

```python
def run_pipeline(
    input_spec: InputSpec,
    output_spec: OutputSpec,
    config: SNAPSConfig,
    verbosity: int = 0,
) -> AnalysisResult:
    """
    Main pipeline - can be called from CLI or web service.
    
    Args:
        input_spec: Input file specifications
        output_spec: Output file specifications
        config: SNAPS configuration
        verbosity: Logging level (0=quiet, 1=info, 2=debug)
    
    Returns:
        Analysis results with assignments and metrics
    """
    
    # Setup logging
    logger = setup_logger(output_spec.log_file, verbosity)
    logger.info("Starting SNAPS pipeline")
    
    # Stage 1: Import
    logger.info("Importing data...")
    importer = SNAPSImporter()
    data = importer.import_data(
        obs_file=input_spec.obs_file,
        pred_file=input_spec.pred_file,
        obs_format=input_spec.obs_format,
        pred_format=input_spec.pred_format,
        obs_frame=input_spec.obs_frame,
        pred_frame=input_spec.pred_frame,
        obs_chain=input_spec.obs_chain,
        pred_chain=input_spec.pred_chain,
        pred_offset=input_spec.pred_offset,
        aa_restraints_file=input_spec.aa_restraints_file,
        aa_restraints_frame=input_spec.aa_restraints_frame,
    )
    logger.info(f"  {len(data.obs_shifts)} observed spin systems")
    logger.info(f"  {len(data.pred_shifts)} predicted residues")
    
    # Stage 2: Assignment
    logger.info(f"Running {config.strategy} assignment...")
    strategy = create_strategy(config.strategy, config.dict())
    assignment_result = strategy.assign(data)
    logger.info("  Assignments complete")
    
    # Stage 3: Analysis
    logger.info("Analyzing results...")
    analyzers = [
        ConsistencyAnalyzer(threshold=config.seq_link_threshold),
        QualityAnalyzer(),
    ]
    analysis_pipeline = AnalysisPipeline(analyzers)
    analysis_result = analysis_pipeline.run(assignment_result, data)
    logger.info(f"  Quality: {analysis_result.quality_metrics}")
    
    # Stage 4: Output
    logger.info(f"Writing output ({output_spec.output_format})...")
    output_manager = OutputManager()
    output_manager.write(
        analysis_result,
        output_spec.output_file,
        format=output_spec.output_format,
        chain=input_spec.obs_chain,
        analysis_level=output_spec.analysis_level,
    )
    logger.info(f"  Written to {output_spec.output_file}")
    
    # Stage 5: Plots (optional)
    if output_spec.hsqc_plot or output_spec.strip_plot:
        logger.info("Creating plots...")
        plot_manager = PlotManager()
        
        if output_spec.hsqc_plot:
            plot_manager.create_plot('hsqc', analysis_result, data, output_spec.hsqc_plot)
            logger.info(f"  HSQC → {output_spec.hsqc_plot}")
        
        if output_spec.strip_plot:
            plot_manager.create_plot('strip', analysis_result, data, output_spec.strip_plot)
            logger.info(f"  Strips → {output_spec.strip_plot}")
    
    logger.info("Pipeline complete")
    return analysis_result
```

## Web Service Considerations

The refactored architecture is web-service ready out of the box:

```python
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
import tempfile


app = FastAPI(title="SNAPS Web Service")


@app.post("/assign")
async def assign_shifts(
    obs_file: UploadFile = File(...),
    pred_file: UploadFile = File(...),
    strategy: str = Form("iterative"),
    output_format: str = Form("nef"),
    analysis_level: str = Form("standard"),
):
    """Assign shifts via web API"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        
        # Save uploads
        obs_path = tmpdir_path / obs_file.filename
        pred_path = tmpdir_path / pred_file.filename
        output_path = tmpdir_path / "output.nef"
        
        async with aiofiles.open(obs_path, 'wb') as f:
            await f.write(await obs_file.read())
        async with aiofiles.open(pred_path, 'wb') as f:
            await f.write(await pred_file.read())
        
        # Build config
        config = SNAPSConfig(strategy=strategy)
        
        # Package parameters into dataclasses
        input_spec = InputSpec(
            obs_file=obs_path,
            pred_file=pred_path,
        )
        
        output_spec = OutputSpec(
            output_file=output_path,
            output_format=output_format,
            analysis_level=analysis_level,
        )
        
        # Run pipeline (same code as CLI!)
        result = run_pipeline(
            input_spec=input_spec,
            output_spec=output_spec,
            config=config,
            verbosity=0,  # Quiet for web service
        )
        
        # Return results
        return FileResponse(output_path, filename="assignments.nef")


@app.get("/info")
async def info():
    """API information"""
    return {
        "formats": ["nef", "shiftx2", "sparky", "mars"],
        "strategies": StrategyRegistry.list_names(),
        "analysis_levels": ["minimal", "standard", "full"],
    }


@app.get("/strategies")
async def list_strategies():
    """List available assignment strategies"""
    return {
        "strategies": [
            {"name": name, "class": cls.__name__, "doc": cls.__doc__}
            for name, cls in StrategyRegistry._strategies.items()
        ]
    }
```

## Migration Path

### Phase-by-Phase Implementation

1. **Phase 1 - Data structures (1-2 days)**
   - Create `ShiftData`, `AssignmentResult`, `AnalysisResult` dataclasses
   - Update tests to use new structures
   - No breaking changes to existing code

2. **Phase 2 - Format readers (3-5 days)**
   - Extract format reading to separate classes
   - Implement `FormatReader` protocol
   - Create NEF, ShiftX2, Sparky readers
   - Add auto-detection logic

3. **Phase 3 - Importer (2-3 days)**
   - Create `SNAPSImporter` orchestrating class
   - Implement frame name conventions
   - Add tests for convention detection

4. **Phase 4 - Strategy pattern (3-4 days)**
   - Extract current assignment logic to `IterativeStrategy`
   - Create strategy base class
   - Implement `DirectStrategy`
   - Add strategy factory

5. **Phase 5 - Analyzers (2-3 days)**
   - Extract consistency checking to `ConsistencyAnalyzer`
   - Create `QualityAnalyzer` for confidence levels
   - Implement `AnalysisPipeline`

6. **Phase 6 - Output (2-3 days)**
   - Create `NEFWriter`, `SNAPSWriter` classes
   - Implement `OutputManager`
   - Separate plotting to `PlotManager`

7. **Phase 7 - CLI (2-3 days)**
   - Implement Typer-based CLI
   - Add file/frame conventions
   - Create `snaps info` command
   - Add rich output formatting

8. **Phase 8 - Config (2-3 days)**
   - Implement Pydantic-based config
   - Add YAML config support
   - Implement priority hierarchy
   - Migration from old config format

9. **Phase 9 - Integration (2-3 days)**
   - Wire everything in `run_pipeline`
   - Comprehensive integration tests
   - Performance benchmarking
   - Documentation

10. **Phase 10 - Web service (1-2 days, optional)**
    - Add FastAPI wrapper
    - Implement file upload handling
    - Add API documentation
    - Deploy examples

**Total estimated time: 20-30 days**

Each phase maintains backward compatibility where possible and includes comprehensive tests.

---

## Summary of Key Improvements

This refactored plan addresses all identified TODOs and implements:

### 1. **Self-Registering Strategies**
- Strategies use `@StrategyRegistry.register('name')` decorator
- Auto-builds available strategy list
- Adding new strategies requires zero changes to factory or CLI
- Strategy names enumerated dynamically from registry

### 2. **Structured Parameter Passing**
- `InputSpec` and `OutputSpec` dataclasses package related parameters
- Pipeline signature simplified from 17+ parameters to 3
- Cleaner, more maintainable code
- Easy to extend without breaking changes

### 3. **Quiet by Default**
- Verbosity levels: 0=quiet (errors only), 1=info (`-v`), 2=debug (`-vv`)
- No noise unless requested
- Professional CLI behavior

### 4. **Auto-Detect Output Format**
- `OutputFormat.auto` detects from file extension
- Users rarely need to specify format explicitly
- Convention over configuration

### 5. **Info as Flag**
- `--info` shows file information and exits
- Ignores all other options when set
- Lists available strategies, formats, analysis levels
- Convenient for exploration

### 6. **Analysis Level Control**
- `--analysis minimal|standard|full` controls output detail
- Minimal: just assignments
- Standard: + consistency metrics (default)
- Full: + all matrices and diagnostics

### 7. **Constructor Injection (not full DI)**
- Simple dependency passing via constructors
- Protocol-based interfaces for flexibility
- No heavyweight DI framework needed
- Python's duck typing makes this natural

---

*This refactoring prioritizes separation of concerns, testability, and flexibility while maintaining SNAPS core functionality and adding modern conveniences like smart conventions, minimal CLI arguments, and quiet-by-default operation.*