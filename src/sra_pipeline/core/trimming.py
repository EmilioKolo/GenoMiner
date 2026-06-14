#!/usr/bin/env python3
"""
Adapter trimming and quality filtering using Trimmomatic.
"""

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
    adapter_fasta: Optional[Path] = None,
    threads: int = 1,
) -> List[Path]:
    """
    Run Trimmomatic for adapter trimming and quality filtering.

    Args:
        sample_id: Sample identifier
        fastq_files: List of input FASTQ file paths (1 or 2 files)
        output_dir: Output directory for trimmed FASTQ files
        logger: Logger instance
        adapter_fasta: Path to adapter sequences FASTA (default: TruSeq3-PE.fa)
        threads: Number of threads to use

    Returns:
        List of trimmed FASTQ file paths

    Raises:
        RuntimeError: If trimming fails or input file count is invalid
    """
    if len(fastq_files) not in (1, 2):
        raise ValueError(f"Expected 1 or 2 FASTQ files, got {len(fastq_files)}")

    # Default adapter file (TruSeq3-PE) assumes standard location or PATH
    if adapter_fasta is None:
        # Try to locate TruSeq3-PE.fa in common Trimmomatic installation
        import shutil
        trimmomatic_path = shutil.which("trimmomatic")
        if trimmomatic_path:
            base_dir = Path(trimmomatic_path).parent.parent
            candidate = base_dir / "share" / "trimmomatic" / "adapters" / "TruSeq3-PE.fa"
            if candidate.exists():
                adapter_fasta = candidate
        if adapter_fasta is None or not adapter_fasta.exists():
            # Fallback: assume file is in current directory or provided
            adapter_fasta = Path("TruSeq3-PE.fa")
            if not adapter_fasta.exists():
                raise FileNotFoundError(
                    "TruSeq3-PE.fa not found. Please provide adapter_fasta path."
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
            timeout=3600,  # 1 hour timeout
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
        "trimmomatic",
        "SE",
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
        "trimmomatic",
        "PE",
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
