"""
Sequence alignment functionality using BWA.
"""

import subprocess
from pathlib import Path
from typing import List
import structlog

from ..utils import log_command, log_error


def run_alignment(
    sample_id: str,
    fastq_files: List[Path],
    reference_fasta: Path,
    output_dir: Path,
    threads: int,
    logger: structlog.BoundLogger
) -> Path:
    """
    Run BWA alignment of FASTQ files to reference genome.
    
    Args:
        sample_id: Sample identifier
        fastq_files: List of FASTQ file paths
        reference_fasta: Reference genome FASTA file
        output_dir: Output directory for BAM files
        threads: Number of threads to use
        logger: Logger instance
        
    Returns:
        Path to the sorted and indexed BAM file
        
    Raises:
        RuntimeError: If alignment fails
    """
    logger.info("Starting BWA alignment", 
                sample_id=sample_id, 
                fastq_files=[str(f) for f in fastq_files],
                reference=str(reference_fasta),
                threads=threads)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: BWA alignment to SAM
    sam_file = _run_bwa_mem(sample_id, fastq_files, reference_fasta, output_dir, threads, logger)
    
    # Step 2: Convert SAM to BAM
    bam_file = _convert_sam_to_bam(sample_id, sam_file, output_dir, logger)
    
    # Step 3: Sort BAM file
    sorted_bam_file = _sort_bam(sample_id, bam_file, output_dir, logger)
    
    # Step 4: Index BAM file
    indexed_bam_file = _index_bam(sample_id, sorted_bam_file, logger)
    
    # Clean up intermediate files
    _cleanup_intermediate_files([sam_file, bam_file], logger)
    
    logger.info("BWA alignment completed", 
                sample_id=sample_id, 
                output_bam=str(indexed_bam_file))
    
    return indexed_bam_file


def _run_bwa_mem(
    sample_id: str,
    fastq_files: List[Path],
    reference_fasta: Path,
    output_dir: Path,
    threads: int,
    logger: structlog.BoundLogger
) -> Path:
    """Run BWA mem alignment."""
    sam_file = output_dir / f"{sample_id}.sam"
    
    # Build BWA command
    cmd = [
        "bwa", "mem",
        "-M",  # Mark shorter split hits as secondary
        "-t", str(threads),
        str(reference_fasta)
    ]
    
    # Add FASTQ files
    cmd.extend([str(f) for f in fastq_files])
    
    # Redirect output to SAM file
    cmd.extend([">", str(sam_file)])
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            " ".join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            #timeout=9800  # 3 hour timeout
        )
        
        logger.info("BWA mem completed", 
                    sample_id=sample_id, 
                    sam_file=str(sam_file),
                    stdout_lines=len(result.stdout.split('\n')) if result.stdout else 0)
        
        return sam_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"BWA mem alignment timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"BWA mem alignment failed for sample {sample_id}: {e.stderr}")


def _convert_sam_to_bam(
    sample_id: str,
    sam_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """Convert SAM file to BAM format."""
    bam_file = output_dir / f"{sample_id}.bam"
    
    cmd = [
        "samtools", "view",
        "-bS", str(sam_file),
        "-o", str(bam_file)
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=7200  # 2 hour timeout
        )
        
        logger.info("SAM to BAM conversion completed", 
                    sample_id=sample_id, 
                    bam_file=str(bam_file))
        
        return bam_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"SAM to BAM conversion timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"SAM to BAM conversion failed for sample {sample_id}: {e.stderr}")


def _sort_bam(
    sample_id: str,
    bam_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """Sort BAM file by coordinates."""
    sorted_bam_file = output_dir / f"{sample_id}_sorted.bam"
    
    cmd = [
        "samtools", "sort",
        str(bam_file),
        "-o", str(sorted_bam_file)
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=7200  # 2 hour timeout
        )
        
        logger.info("BAM sorting completed", 
                    sample_id=sample_id, 
                    sorted_bam=str(sorted_bam_file))
        
        return sorted_bam_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"BAM sorting timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"BAM sorting failed for sample {sample_id}: {e.stderr}")


<<<<<<< Updated upstream
=======
def _mark_duplicates(
    sample_id: str,
    sorted_bam_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """
    Run Picard MarkDuplicates to mark duplicate reads.
    
    Args:
        sample_id: Sample identifier
        sorted_bam_file: Sorted BAM file input
        output_dir: Output directory for marked BAM and metrics file
        logger: Logger instance
        
    Returns:
        Path to the metrics file
    """
    marked_bam_file = output_dir / f"{sample_id}_sorted_marked.bam"
    metrics_file = output_dir / f"{sample_id}_dup_metrics.txt"
    
    cmd = [
        "picard", "MarkDuplicates",
        f"I={sorted_bam_file}",
        f"O={marked_bam_file}",
        f"M={metrics_file}",
        "REMOVE_DUPLICATES=false",
        "ASSUME_SORTED=true",
        "VALIDATION_STRINGENCY=SILENT"
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=7200  # 2 hour timeout
        )
        logger.info("Duplicate marking completed", 
                    sample_id=sample_id, 
                    marked_bam=str(marked_bam_file),
                    metrics_file=str(metrics_file))
        
        # Replace original sorted BAM with marked BAM for further steps
        sorted_bam_file.unlink()
        marked_bam_file.rename(sorted_bam_file)
        
        return metrics_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"MarkDuplicates timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"MarkDuplicates failed for sample {sample_id}: {e.stderr}")


>>>>>>> Stashed changes
def _index_bam(
    sample_id: str,
    bam_file: Path,
    logger: structlog.BoundLogger
) -> Path:
    """Index BAM file."""
    index_file = bam_file.with_suffix(".bam.bai")
    
    cmd = [
        "samtools", "index",
        str(bam_file)
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=7200  # 2 hour timeout
        )
        
        logger.info("BAM indexing completed", 
                    sample_id=sample_id, 
                    index_file=str(index_file))
        
        return bam_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"BAM indexing timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"BAM indexing failed for sample {sample_id}: {e.stderr}")


def _cleanup_intermediate_files(files: List[Path], logger: structlog.BoundLogger):
    """Clean up intermediate files."""
    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("Cleaned up intermediate file", file=str(file_path))
        except Exception as e:
            logger.warning("Failed to clean up intermediate file", 
                          file=str(file_path), error=str(e))


def validate_bam_file(bam_file: Path, logger: structlog.BoundLogger) -> bool:
    """Validate that BAM file is properly formatted and indexed."""
    try:
<<<<<<< Updated upstream
        # Check if BAM file exists and is not empty
        if not bam_file.exists():
            logger.error("BAM file does not exist", file=str(bam_file))
            return False
        
        if bam_file.stat().st_size == 0:
            logger.error("BAM file is empty", file=str(bam_file))
            return False
        
        # Check if index file exists
        index_file = bam_file.with_suffix(".bam.bai")
        if not index_file.exists():
            logger.error("BAM index file does not exist", file=str(index_file))
            return False
        
        # Validate BAM file format
        cmd = ["samtools", "quickcheck", str(bam_file)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error("BAM file validation failed", 
                        file=str(bam_file), stderr=result.stderr)
            return False
        
        # Get basic statistics
        cmd = ["samtools", "flagstat", str(bam_file)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        logger.info("BAM file validation passed", 
                    file=str(bam_file), 
                    flagstat=result.stdout.strip())
        
        return True
        
=======
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, #timeout=1200
                                )
        # Parse duplication rate from flagstat output
        for line in result.stdout.splitlines():
            if 'duplicates' in line:
                # Example line: "0 + 0 duplicates"
                parts = line.split()
                if len(parts) >= 1:
                    dup_reads = int(parts[0])
                    total_reads_line = [l for l in result.stdout.splitlines() if 'in total' in l]
                    if total_reads_line:
                        total_reads = int(total_reads_line[0].split()[0])
                        if total_reads > 0:
                            rates['duplication_rate_flagstat'] = (dup_reads / total_reads) * 100.0
        logger.info("Flagstat duplication rate parsed", rate=rates['duplication_rate_flagstat'])
>>>>>>> Stashed changes
    except Exception as e:
        logger.error("BAM file validation error", file=str(bam_file), error=str(e))
        return False 