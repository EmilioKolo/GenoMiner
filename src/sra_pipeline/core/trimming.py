#!/usr/bin/env python3
"""
Adapter trimming and quality filtering using Trimmomatic.
"""

import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional
import structlog

from ..utils import log_command


def run_trimming(
    sample_id: str,
    fastq_files: List[Path],
    output_dir: Path,
    logger: structlog.BoundLogger,
    adapter_dir: Optional[Path] = None,
    threads: int = 1,
) -> List[Path]:
    """
    Run Trimmomatic for adapter trimming and quality filtering.

    Args:
        sample_id: Sample identifier
        fastq_files: List of input FASTQ file paths (1 or 2 files)
        output_dir: Output directory for trimmed FASTQ files
        logger: Logger instance
        adapter_dir: Directory containing Trimmomatic adapter FASTAs
                     (TruSeq3-PE.fa for paired-end, TruSeq3-SE.fa for single-end)
        threads: Number of threads to use

    Returns:
        List of trimmed FASTQ file paths

    Raises:
        RuntimeError: If trimming fails or input file count is invalid
    """
    if len(fastq_files) not in (1, 2):
        raise ValueError(f"Expected 1 or 2 FASTQ files, got {len(fastq_files)}")

    # Determine which adapter file is needed based on SE vs PE.
    if len(fastq_files) == 1:
        adapter_name = "TruSeq3-SE.fa"
        mode = "SE"
    else:
        adapter_name = "TruSeq3-PE.fa"
        mode = "PE"

    # Resolve adapter directory: explicit config path first, then discovery.
    adapter_dir = _resolve_adapter_dir(adapter_dir, mode, logger)
    adapter_fasta = adapter_dir / adapter_name
    if not adapter_fasta.exists():
        raise FileNotFoundError(
            f"{adapter_name} not found in adapter directory {adapter_dir}. "
            f"Expected file: {adapter_fasta}. "
            f"Set TRIMMOMATIC_ADAPTER_DIR in config.ini or export it as an "
            f"environment variable."
        )

    logger.info(
        "Starting Trimmomatic trimming",
        sample_id=sample_id,
        fastq_files=[str(f) for f in fastq_files],
        adapter_fasta=str(adapter_fasta),
        threads=threads,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build output file names
    prefix = output_dir / f"{sample_id}_trimmed"
    if len(fastq_files) == 1:
        # Single-end
        output_paired = prefix.with_suffix(".fastq.gz")
        cmd = _build_trimmomatic_se_cmd(
            fastq_files[0], output_paired, adapter_fasta, threads
        )
        trimmed_files = [output_paired]
    else:
        # Paired-end
        output_forward_paired = prefix.with_name(f"{sample_id}_1_trimmed.fastq.gz")
        output_forward_unpaired = prefix.with_name(f"{sample_id}_1_unpaired.fastq.gz")
        output_reverse_paired = prefix.with_name(f"{sample_id}_2_trimmed.fastq.gz")
        output_reverse_unpaired = prefix.with_name(f"{sample_id}_2_unpaired.fastq.gz")
        cmd = _build_trimmomatic_pe_cmd(
            fastq_files[0],
            fastq_files[1],
            output_forward_paired,
            output_forward_unpaired,
            output_reverse_paired,
            output_reverse_unpaired,
            adapter_fasta,
            threads,
        )
        trimmed_files = [output_forward_paired, output_reverse_paired]

    log_command(logger, " ".join(cmd), sample_id=sample_id)

    try:
        result = subprocess.run(
            " ".join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            #timeout=10800,  # 3 hours timeout
        )
        logger.info(
            "Trimmomatic completed",
            sample_id=sample_id,
            stdout=result.stdout[:500] if result.stdout else "",
        )
        return trimmed_files
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Trimmomatic timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Trimmomatic failed for sample {sample_id}: {e.stderr}")


def _build_trimmomatic_se_cmd(
    input_fastq: Path,
    output_paired: Path,
    adapter_fasta: Path,
    threads: int,
) -> List[str]:
    """Build command line for single-end Trimmomatic."""
    return [
        "TrimmomaticSE",
        "-threads", str(threads),
        str(input_fastq),
        str(output_paired),
        f"ILLUMINACLIP:{adapter_fasta}:2:30:10",
        "SLIDINGWINDOW:4:20",
        "LEADING:3",
        "TRAILING:3",
        "MINLEN:36",
    ]


def _build_trimmomatic_pe_cmd(
    forward_input: Path,
    reverse_input: Path,
    forward_paired: Path,
    forward_unpaired: Path,
    reverse_paired: Path,
    reverse_unpaired: Path,
    adapter_fasta: Path,
    threads: int,
) -> List[str]:
    """Build command line for paired-end Trimmomatic."""
    return [
        "TrimmomaticPE",
        "-threads", str(threads),
        str(forward_input),
        str(reverse_input),
        str(forward_paired),
        str(forward_unpaired),
        str(reverse_paired),
        str(reverse_unpaired),
        f"ILLUMINACLIP:{adapter_fasta}:2:30:10",
        "SLIDINGWINDOW:4:20",
        "LEADING:3",
        "TRAILING:3",
        "MINLEN:36",
    ]

def _resolve_adapter_dir(
    adapter_dir: Optional[Path],
    mode: str,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Resolve the directory that contains Trimmomatic adapter FASTAs.

    Order of precedence:
      1. Explicit directory passed in (from config.ini via PipelineConfig).
      2. TRIMMOMATIC_ADAPTER_DIR environment variable.
      3. $TRIMMOMATIC_HOME/adapters.
      4. $CONDA_PREFIX/share/trimmomatic/adapters.
      5. Paths derived from the `trimmomatic` binary on PATH.
      6. Well-known system locations.
      7. Current working directory.

    A candidate directory is accepted only if it exists AND contains the
    adapter file matching the requested mode (TruSeq3-SE.fa for SE,
    TruSeq3-PE.fa for PE). The `default_name` argument is used only to
    probe directories when the mode-specific file is not present yet, so
    the caller can still get a useful error listing what was tried.

    Raises:
        FileNotFoundError: If no candidate directory contains the required
                           adapter file.
    """
    required_name = "TruSeq3-SE.fa" if mode == "SE" else "TruSeq3-PE.fa"
    candidates: List[Path] = []

    # 1. Explicit path from config
    if adapter_dir is not None:
        candidates.append(adapter_dir)

    # 2. Environment variable
    env_dir = os.environ.get("TRIMMOMATIC_ADAPTER_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    # 3. TRIMMOMATIC_HOME
    tm_home = os.environ.get("TRIMMOMATIC_HOME")
    if tm_home:
        candidates.append(Path(tm_home) / "adapters")

    # 4. Conda prefix
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        candidates.append(Path(conda_prefix) / "share" / "trimmomatic" / "adapters")

    # 5. Derive from the `trimmomatic` binary on PATH
    tm_which = shutil.which("trimmomatic")
    if tm_which:
        tm_path = Path(tm_which).resolve()
        candidates.append(tm_path.parent.parent / "share" / "trimmomatic" / "adapters")
        candidates.append(tm_path.parent / "adapters")
        for jar in tm_path.parent.glob("trimmomatic*.jar"):
            candidates.append(jar.parent / "adapters")

    # 6. Well-known system locations
    candidates.extend([
        Path("/usr/share/trimmomatic/adapters"),
        Path("/usr/local/share/trimmomatic/adapters"),
        Path("/opt/trimmomatic/adapters"),
    ])

    # 7. Current working directory
    candidates.append(Path.cwd())

    tried: List[str] = []
    for candidate in candidates:
        tried.append(str(candidate))
        if candidate.is_dir() and (candidate / required_name).exists():
            logger.info(
                "Resolved Trimmomatic adapter directory",
                adapter_dir=str(candidate),
                adapter_file=required_name,
                mode=mode,
            )
            return candidate

    tried_str = "\n  - ".join(tried)
    raise FileNotFoundError(
        f"Could not find {required_name} in any candidate adapter directory "
        f"(mode={mode}). Tried:\n  - {tried_str}\n\n"
        "Provide the adapter directory by any of these methods:\n"
        "  1. Set TRIMMOMATIC_ADAPTER_DIR in config.ini under [Paths]\n"
        "  2. Export the TRIMMOMATIC_ADAPTER_DIR environment variable\n"
        "  3. Pass adapter_dir explicitly to run_trimming()"
    )
