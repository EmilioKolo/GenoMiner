"""
Variant calling functionality using bcftools.
"""

import subprocess
from pathlib import Path
import structlog

from ..utils import log_command


def run_variant_calling(
    sample_id: str,
    bam_file: Path,
    reference_fasta: Path,
    output_dir: Path,
    snpeff_dir: Path,
    genome_name: str,
    min_quality_score: int,
    min_coverage: int,
    logger: structlog.BoundLogger
) -> tuple[Path, Path]:
    """
    Run variant calling using bcftools.
    
    Args:
        sample_id: Sample identifier
        bam_file: Input BAM file
        reference_fasta: Reference genome FASTA file
        output_dir: Output directory for VCF files
        min_quality_score: Minimum quality score for variants
        min_coverage: Minimum coverage for variant calling
        logger: Logger instance
        
    Returns:
        Path to the final merged VCF file
        
    Raises:
        RuntimeError: If variant calling fails
    """
    logger.info("Starting variant calling", 
                sample_id=sample_id, 
                bam_file=str(bam_file),
                reference=str(reference_fasta),
                min_quality=min_quality_score,
                min_coverage=min_coverage)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create temporary directory for intermediate files
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    
    intermediate_files = []
    
    try:
        # Step 1: Generate pileup
        bcf_file = _run_mpileup(sample_id, bam_file, reference_fasta, tmp_dir, logger)
        intermediate_files.append(bcf_file)
        
        # Step 2: Call variants
        vcf_file = _call_variants(sample_id, bcf_file, tmp_dir, min_quality_score, logger)
        intermediate_files.append(vcf_file)
        
        # Step 3: Filter variants
        filtered_vcf_file = _filter_variants(sample_id, vcf_file, tmp_dir, min_quality_score, min_coverage, logger)
        intermediate_files.append(filtered_vcf_file)
        
        # Step 4: Compress and index VCF
        compressed_vcf_file = _compress_and_index_vcf(sample_id, filtered_vcf_file, tmp_dir, logger)
        intermediate_files.append(compressed_vcf_file)
        
        # Step 5: Move filtered VCF and its index to output directory and rename
        final_vcf_file = output_dir / f"{sample_id}_variants.vcf.gz"
        final_vcf_file.unlink(missing_ok=True)  # Remove if exists
        compressed_vcf_file.rename(final_vcf_file)

        compressed_vcf_file_index = tmp_dir / f"{sample_id}_compressed.vcf.gz.tbi"
        final_vcf_file_index = output_dir / f"{sample_id}_variants.vcf.gz.tbi"
        final_vcf_file_index.unlink(missing_ok=True)  # Remove if exists
        compressed_vcf_file_index.rename(final_vcf_file_index)
        
        # Step 6: Analysis with snpEff
        snpeff_vcf = _snpeff_analysis(sample_id, final_vcf_file, tmp_dir, snpeff_dir, genome_name, logger)
        intermediate_files.append(snpeff_vcf)

        # Step 7: Compress and index snpEff vcf
        compressed_snpeff_file = _compress_and_index_vcf(sample_id, snpeff_vcf, output_dir, logger)

        # Step 8: Move snpeff VCF and its index to output directory and rename
        final_snpeff = output_dir / f"{sample_id}_snpeff.vcf.gz"
        final_snpeff.unlink(missing_ok=True)  # Remove if exists
        compressed_snpeff_file.rename(final_snpeff)

        compressed_snpeff_file_index = output_dir / f"{sample_id}_compressed.vcf.gz.tbi"
        final_snpeff_index = output_dir / f"{sample_id}_snpeff.vcf.gz.tbi"
        final_snpeff_index.unlink(missing_ok=True)  # Remove if exists
        compressed_snpeff_file_index.rename(final_snpeff_index)

        # Step 9: Clean up intermediate files
        _cleanup_intermediate_files(intermediate_files, logger)
        
        logger.info("Variant calling completed", 
                    sample_id=sample_id, 
                    output_vcf=str(final_snpeff))
        
        return final_vcf_file, final_snpeff
        
    except Exception as e:
        # Clean up intermediate files on error
        _cleanup_intermediate_files(intermediate_files, logger)
        raise e


def _run_mpileup(
    sample_id: str,
    bam_file: Path,
    reference_fasta: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """Run bcftools mpileup to generate pileup."""
    try:
        bcf_file = _run_mpileup_by_chr(
            sample_id=sample_id,
            bam_file=bam_file,
            reference_fasta=reference_fasta,
            output_dir=output_dir,
            logger=logger
        )
        
        logger.info("Mpileup completed", 
                    sample_id=sample_id, 
                    bcf_file=str(bcf_file))
        
        return bcf_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Mpileup timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Mpileup failed for sample {sample_id}: {e.stderr}")
    except Exception as e:
        raise RuntimeError(f"Mpileup encountered an error for sample {sample_id}: {str(e)}")


def _run_mpileup_by_chr(
    sample_id: str,
    bam_file: Path,
    reference_fasta: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    
    logger.info("Running mpileup by chromosome", sample_id=sample_id)

    # Make sure output directory exists
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of chromosomes from the reference
    chrom_sizes_file = Path(str(reference_fasta) + ".fai")

    # Only index if the .fai does NOT already exist
    if not chrom_sizes_file.exists():
        cmd_idx = ["samtools", "faidx", str(reference_fasta)]
        subprocess.run(cmd_idx, check=True)

    if not chrom_sizes_file.exists():
        raise RuntimeError(
            "FASTA index (.fai) not found and could not be created."
        )

    chromosomes = []
    with chrom_sizes_file.open() as f:
        for line in f:
            chrom = line.split("\t")[0]
            chromosomes.append(chrom)

    if not chromosomes:
        raise RuntimeError("No chromosomes found in FASTA index.")

    # Run mpileup per chromosome
    per_chrom_bcf = []
    for chrom in chromosomes:
        chrom_bcf = output_dir / f"{sample_id}.{chrom}.bcf"

        cmd = [
            "bcftools", "mpileup",
            "-f", str(reference_fasta),
            "-r", chrom,
            str(bam_file),
            "-o", str(chrom_bcf),
            "-d", "200",
            "--threads", "1"
        ]

        logger.info(f"Running mpileup on {chrom}...", 
                    sample_id=sample_id, chromosome=chrom)
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.stderr:
            logger.debug(f'Mpileup stderr: {result.stderr.strip()}')
        
        per_chrom_bcf.append(str(chrom_bcf))

    # Combine all partial BCFs
    final_bcf = output_dir / f"{sample_id}.bcf"

    logger.info("Concatenating chromosome BCF files...", 
                sample_id=sample_id)
    cmd_concat = [
        "bcftools", "concat", "-o", str(final_bcf), "-O", "b"
    ] + per_chrom_bcf
    result = subprocess.run(
        cmd_concat, 
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.stderr:
        logger.debug(f'Mpileup stderr: {result.stderr.strip()}')

    logger.info(f"Final BCF generated at: {final_bcf}. " + \
                "Cleaning up temporary chromosome BCF files...", 
                sample_id=sample_id)

    for tmp_bcf in per_chrom_bcf:
        try:
            Path(tmp_bcf).unlink()
        except Exception as e:
            logger.warning(f"Could not delete {tmp_bcf}: {e}", 
                           sample_id=sample_id)
    
    logger.info("Cleanup finished.", sample_id=sample_id)

    return final_bcf


def _call_variants(
    sample_id: str,
    bcf_file: Path,
    output_dir: Path,
    min_quality_score: int,
    logger: structlog.BoundLogger
) -> Path:
    """Call variants using bcftools call."""
    vcf_file = output_dir / f"{sample_id}.vcf"
    
    cmd = [
        "bcftools", "call",
        "-mv",  # Multi-allelic variant caller
        "-o", str(vcf_file),
        str(bcf_file)
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=3600  # 1 hour timeout
        )
        
        logger.info("Variant calling completed", 
                    sample_id=sample_id, 
                    vcf_file=str(vcf_file))
        
        return vcf_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Variant calling timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Variant calling failed for sample {sample_id}: {e.stderr}")


def _filter_variants(
    sample_id: str,
    vcf_file: Path,
    output_dir: Path,
    min_quality_score: int,
    min_coverage: int,
    logger: structlog.BoundLogger
) -> Path:
    """Filter variants based on quality criteria."""
    filtered_vcf_file = output_dir / f"{sample_id}_filtered.vcf"
    
    # Create filter expression
    filter_expr = f"QUAL >= {min_quality_score} && INFO/DP >= {min_coverage}"
    
    cmd = [
        "bcftools", "filter",
        "-i", filter_expr,
        "-o", str(filtered_vcf_file),
        str(vcf_file)
    ]
    
    log_command(logger, " ".join(cmd), sample_id=sample_id, filter_expr=filter_expr)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1800  # 30 minute timeout
        )
        
        logger.info("Variant filtering completed", 
                    sample_id=sample_id, 
                    filtered_vcf=str(filtered_vcf_file))
        
        return filtered_vcf_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Variant filtering timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Variant filtering failed for sample {sample_id}: {e.stderr}")


def _compress_and_index_vcf(
    sample_id: str,
    vcf_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """Compress and index VCF file."""
    compressed_vcf_file = output_dir / f"{sample_id}_compressed.vcf.gz"
    
    # Compress VCF
    cmd = ["bgzip", "-c", str(vcf_file), ">", str(compressed_vcf_file)]
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            " ".join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=900  # 15 minute timeout
        )
        
        # Index VCF
        cmd = ["tabix", "-p", "vcf", str(compressed_vcf_file)]
        log_command(logger, " ".join(cmd), sample_id=sample_id)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=900  # 15 minute timeout
        )
        
        logger.info("VCF compression and indexing completed", 
                    sample_id=sample_id, 
                    compressed_vcf=str(compressed_vcf_file))
        
        return compressed_vcf_file
        
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"VCF compression/indexing timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"VCF compression/indexing failed for sample {sample_id}: {e.stderr}")


def _snpeff_analysis(
    sample_id: str,
    vcf_file: Path,
    output_dir: Path,
    snpeff_dir: Path,
    genome_name: str,
    logger: structlog.BoundLogger
) -> Path:
    """Filters variants using snpEff."""
    snpeff_file = output_dir / f"{sample_id}_snpeff.vcf"

    # Run snpEff on the input vcf file
    cmd = ['java', '-Xmx8g', '-jar', 
           f'{str(snpeff_dir)}/snpEff/snpEff.jar', str(genome_name),
           str(vcf_file), '>', str(snpeff_file)]

    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        result = subprocess.run(
            " ".join(cmd),
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            #timeout=3600  # 1 hour timeout
        )

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"snpEff analysis timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"snpEff analysis failed for sample {sample_id}: {e.stderr}")
    
    return snpeff_file


def _cleanup_intermediate_files(files: list, logger: structlog.BoundLogger):
    """Clean up intermediate files."""
    for file_path in files:
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info("Cleaned up intermediate file", file=str(file_path))
        except Exception as e:
            logger.warning("Failed to clean up intermediate file", 
                          file=str(file_path), error=str(e))


def merge_vcf_files(
    vcf_files: list,
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Path:
    """
    Merge multiple VCF files into a single VCF file.
    
    Args:
        vcf_files: List of VCF file paths to merge
        output_dir: Output directory for merged VCF
        logger: Logger instance
        
    Returns:
        Path to the merged VCF file
        
    Raises:
        RuntimeError: If merging fails
    """
    if not vcf_files:
        raise ValueError("No VCF files provided for merging")
    
    if len(vcf_files) == 1:
        # If only one VCF file, just copy it to output directory
        source_vcf = Path(vcf_files[0])
        merged_vcf = output_dir / f"merged_variants.vcf.gz"
        merged_vcf.unlink(missing_ok=True)
        source_vcf.rename(merged_vcf)
        logger.info("Single VCF file copied", source=str(source_vcf), target=str(merged_vcf))
        return merged_vcf
    
    logger.info("Starting VCF merging", 
                vcf_count=len(vcf_files),
                vcf_files=[str(f) for f in vcf_files])
    
    # Create file list for bcftools merge
    vcf_list_file = output_dir / "vcf_files.txt"
    with open(vcf_list_file, 'w') as f:
        for vcf_file in vcf_files:
            f.write(f"{vcf_file}\n")
    
    # Merge VCF files
    merged_vcf = output_dir / "merged_variants.vcf.gz"
    
    cmd = [
        "bcftools", "merge",
        "--file-list", str(vcf_list_file),
        "-o", str(merged_vcf),
        "-O", "z"  # Output compressed VCF
    ]
    
    log_command(logger, " ".join(cmd))
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            #timeout=3600  # 1 hour timeout for merging
        )
        
        # Index the merged VCF
        cmd = ["tabix", "-p", "vcf", str(merged_vcf)]
        log_command(logger, " ".join(cmd))
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=900  # 15 minute timeout
        )
        
        # Clean up file list
        vcf_list_file.unlink(missing_ok=True)
        
        logger.info("VCF merging completed", 
                    merged_vcf=str(merged_vcf),
                    variant_count=_count_variants(merged_vcf, logger))
        
        return merged_vcf
        
    except subprocess.TimeoutExpired:
        raise RuntimeError("VCF merging timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"VCF merging failed: {e.stderr}")
    finally:
        # Clean up file list
        vcf_list_file.unlink(missing_ok=True)


def _count_variants(vcf_file: Path, logger: structlog.BoundLogger) -> int:
    """Count the number of variants in a VCF file."""
    try:
        cmd = ["bcftools", "view", "-H", str(vcf_file)]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=1200 # 20 minute timeout
        )
        return len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
    except Exception as e:
        logger.warning("Failed to count variants", file=str(vcf_file), error=str(e))
        return 0 