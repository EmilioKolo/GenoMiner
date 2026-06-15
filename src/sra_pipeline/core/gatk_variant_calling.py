#!/usr/bin/env python3
"""
GATK-based variant calling with BQSR, HaplotypeCaller GVCF, joint genotyping, 
Mutect2, VQSR/hard-filtering, ANNOVAR annotation.
"""

import subprocess
from pathlib import Path
from typing import List, Optional
import structlog

from ..utils import log_command


def run_bqsr(
    sample_id: str,
    bam_file: Path,
    reference_fasta: Path,
    known_sites_vcf: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Run GATK BaseRecalibrator and ApplyBQSR.
    
    Args:
        sample_id: Sample identifier
        bam_file: Input BAM file (sorted, duplicates marked)
        reference_fasta: Reference genome FASTA
        known_sites_vcf: Known sites VCF (e.g., dbSNP + 1000G)
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Path to recalibrated BAM file
    """
    logger.info("Starting GATK BQSR", sample_id=sample_id, bam_file=str(bam_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    recal_table = output_dir / f"{sample_id}_recal_data.table"
    recalibrated_bam = output_dir / f"{sample_id}_recalibrated.bam"
    
    # Step 1: BaseRecalibrator
    cmd1 = [
        "gatk", "BaseRecalibrator",
        "-I", str(bam_file),
        "-R", str(reference_fasta),
        "--known-sites", str(known_sites_vcf),
        "-O", str(recal_table)
    ]
    log_command(logger, " ".join(cmd1), sample_id=sample_id)
    try:
        subprocess.run(cmd1, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"BaseRecalibrator timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"BaseRecalibrator failed: {e.stderr}")
    
    # Step 2: ApplyBQSR
    cmd2 = [
        "gatk", "ApplyBQSR",
        "-I", str(bam_file),
        "-R", str(reference_fasta),
        "--bqsr-recal-file", str(recal_table),
        "-O", str(recalibrated_bam)
    ]
    log_command(logger, " ".join(cmd2), sample_id=sample_id)
    try:
        subprocess.run(cmd2, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ApplyBQSR timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ApplyBQSR failed: {e.stderr}")
    
    logger.info("BQSR completed", sample_id=sample_id, recalibrated_bam=str(recalibrated_bam))
    return recalibrated_bam


def run_haplotypecaller_gvcf(
    sample_id: str,
    bam_file: Path,
    reference_fasta: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Run GATK HaplotypeCaller in GVCF mode.
    
    Returns:
        Path to the GVCF file
    """
    logger.info("Starting HaplotypeCaller GVCF", sample_id=sample_id, bam_file=str(bam_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    gvcf_file = output_dir / f"{sample_id}.g.vcf.gz"
    
    cmd = [
        "gatk", "HaplotypeCaller",
        "-R", str(reference_fasta),
        "-I", str(bam_file),
        "-O", str(gvcf_file),
        "-ERC", "GVCF"
    ]
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)
        logger.info("HaplotypeCaller completed", sample_id=sample_id, gvcf=str(gvcf_file))
        return gvcf_file
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"HaplotypeCaller timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"HaplotypeCaller failed: {e.stderr}")


def run_joint_genotyping(
    gvcf_files: List[Path],
    reference_fasta: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Run GATK GenotypeGVCFs on multiple GVCF files.
    
    Args:
        gvcf_files: List of GVCF file paths
        reference_fasta: Reference genome FASTA
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Path to joint VCF file
    """
    logger.info("Starting joint genotyping", gvcf_count=len(gvcf_files))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_vcf = output_dir / "joint_genotypes.vcf.gz"
    
    # Build command with -V arguments
    cmd = ["gatk", "GenotypeGVCFs", "-R", str(reference_fasta)]
    for gvcf in gvcf_files:
        cmd.extend(["-V", str(gvcf)])
    cmd.extend(["-O", str(combined_vcf)])
    
    log_command(logger, " ".join(cmd))
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)
        logger.info("Joint genotyping completed", output_vcf=str(combined_vcf))
        return combined_vcf
    except subprocess.TimeoutExpired:
        raise RuntimeError("Joint genotyping timed out")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Joint genotyping failed: {e.stderr}")


def run_mutect2(
    sample_id: str,
    tumor_bam: Path,
    reference_fasta: Path,
    panel_of_normals: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Run GATK Mutect2 in tumor-only mode using a panel of normals (PON).
    
    Returns:
        Path to somatic VCF file
    """
    logger.info("Starting Mutect2", sample_id=sample_id, tumor_bam=str(tumor_bam))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    somatic_vcf = output_dir / f"{sample_id}_somatic.vcf.gz"
    
    cmd = [
        "gatk", "Mutect2",
        "-R", str(reference_fasta),
        "-I", str(tumor_bam),
        "--panel-of-normals", str(panel_of_normals),
        "-O", str(somatic_vcf)
    ]
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=7200)
        logger.info("Mutect2 completed", sample_id=sample_id, somatic_vcf=str(somatic_vcf))
        return somatic_vcf
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Mutect2 timed out for sample: {sample_id}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Mutect2 failed: {e.stderr}")


def apply_variant_filtering(
    sample_id: str,
    vcf_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
    use_vqsr: bool = False,
    vqsr_training_vcf: Optional[Path] = None,
    vqsr_resource: str = "hapmap,omni,1000G,knownsites",
) -> Path:
    """
    Apply hard-filtering or VQSR to a VCF file.
    
    Hard-filter thresholds (from paper):
        QD < 2.0, FS > 60.0, MQ < 40.0, MQRankSum < -12.5, ReadPosRankSum < -8.0
    
    Args:
        sample_id: Sample identifier
        vcf_file: Input VCF file
        output_dir: Output directory
        logger: Logger instance
        use_vqsr: If True, use VQSR instead of hard-filtering
        vqsr_training_vcf: Optional VCF with known variants for VQSR training
        vqsr_resource: Comma-separated list of training resources
        
    Returns:
        Path to filtered VCF file
    """
    logger.info("Starting variant filtering", sample_id=sample_id, vcf_file=str(vcf_file), use_vqsr=use_vqsr)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if use_vqsr:
        # VQSR requires a training VCF
        if vqsr_training_vcf is None or not vqsr_training_vcf.exists():
            raise ValueError("VQSR requires a training VCF file (--vqsr-training-vcf)")
        
        filtered_vcf = output_dir / f"{sample_id}_filtered_vqsr.vcf.gz"
        recal_file = output_dir / f"{sample_id}_recal.file"
        tranches_file = output_dir / f"{sample_id}_tranches.file"
        
        # Step 1: VariantRecalibrator
        cmd_recal = [
            "gatk", "VariantRecalibrator",
            "-R", str(vcf_file.parent.parent / "reference.fasta"),
            "-V", str(vcf_file),
            "--resource:known", str(vqsr_training_vcf),
            "-an", "QD", "-an", "MQ", "-an", "MQRankSum", "-an", "ReadPosRankSum", "-an", "FS",
            "-mode", "SNP",
            "-O", str(recal_file),
            "--tranches-file", str(tranches_file)
        ]
        # Add resources
        for res in vqsr_resource.split(","):
            cmd_recal.extend(["--resource", res])
        
        log_command(logger, " ".join(cmd_recal), sample_id=sample_id)
        try:
            subprocess.run(cmd_recal, capture_output=True, text=True, check=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"VariantRecalibrator failed: {e.stderr}")
        
        # Step 2: ApplyVQSR
        cmd_apply = [
            "gatk", "ApplyVQSR",
            "-R", str(vcf_file.parent.parent / "reference.fasta"),
            "-V", str(vcf_file),
            "-O", str(filtered_vcf),
            "--recal-file", str(recal_file),
            "--tranches-file", str(tranches_file),
            "--mode", "SNP"
        ]
        log_command(logger, " ".join(cmd_apply), sample_id=sample_id)
        try:
            subprocess.run(cmd_apply, capture_output=True, text=True, check=True, timeout=3600)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ApplyVQSR failed: {e.stderr}")
    else:
        # Hard filtering
        filtered_vcf = output_dir / f"{sample_id}_filtered_hard.vcf.gz"
        filter_expression = (
            "QD < 2.0 || FS > 60.0 || MQ < 40.0 || "
            "MQRankSum < -12.5 || ReadPosRankSum < -8.0"
        )
        cmd = [
            "gatk", "VariantFiltration",
            "-R", str(vcf_file.parent.parent / "reference.fasta"),
            "-V", str(vcf_file),
            "-O", str(filtered_vcf),
            "--filter-expression", filter_expression,
            "--filter-name", "GATK_hard_filter"
        ]
        log_command(logger, " ".join(cmd), sample_id=sample_id)
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=3600)
            logger.info("Hard filtering completed", sample_id=sample_id, filtered_vcf=str(filtered_vcf))
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"VariantFiltration failed: {e.stderr}")
    
    return filtered_vcf


def run_annovar(
    sample_id: str,
    vcf_file: Path,
    annovar_db: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
) -> Path:
    """
    Annotate VCF with ANNOVAR using RefSeq/GENCODE and population frequencies.
    
    Args:
        sample_id: Sample identifier
        vcf_file: Input VCF file
        annovar_db: ANNOVAR database directory (contains refGene, etc.)
        output_dir: Output directory
        logger: Logger instance
        
    Returns:
        Path to annotated VCF file (with ANN field)
    """
    logger.info("Starting ANNOVAR annotation", sample_id=sample_id, vcf_file=str(vcf_file))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert VCF to ANNOVAR input format (avinput)
    avinput = output_dir / f"{sample_id}.avinput"
    cmd_convert = ["convert2annovar.pl", "-format", "vcf4", str(vcf_file), "-outfile", str(avinput)]
    log_command(logger, " ".join(cmd_convert), sample_id=sample_id)
    try:
        subprocess.run(cmd_convert, capture_output=True, text=True, check=True, timeout=900)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"convert2annovar.pl failed: {e.stderr}")
    
    # Run table_annovar.pl with gene annotation and population frequencies
    annotated = output_dir / f"{sample_id}_annotated"
    cmd_annotate = [
        "table_annovar.pl", str(avinput), str(annovar_db),
        "-buildver", "hg38",  # or detect from reference
        "-out", str(annotated),
        "-remove",
        "-protocol", "refGene,gnomad30_genome,1000g2015aug",
        "-operation", "g,f,f",
        "-nastring", ".",
        "-vcfinput"
    ]
    log_command(logger, " ".join(cmd_annotate), sample_id=sample_id)
    try:
        subprocess.run(cmd_annotate, capture_output=True, text=True, check=True, timeout=3600)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"table_annovar.pl failed: {e.stderr}")
    
    annotated_vcf = output_dir / f"{sample_id}_annotated.hg38_multianno.vcf"
    logger.info("ANNOVAR annotation completed", sample_id=sample_id, annotated_vcf=str(annotated_vcf))
    return annotated_vcf


def filter_somatic_variants(
    sample_id: str,
    vcf_file: Path,
    output_dir: Path,
    logger: structlog.BoundLogger,
    min_vaf: float = 0.01,
    min_depth: int = 10,
    min_alt_reads: int = 3,
) -> Path:
    """
    Filter somatic variants by VAF, depth, and alternate allele count.
    Uses bcftools filter on FORMAT fields.
    
    Returns:
        Path to filtered VCF file
    """
    logger.info("Filtering somatic variants", sample_id=sample_id, vcf_file=str(vcf_file),
                min_vaf=min_vaf, min_depth=min_depth, min_alt_reads=min_alt_reads)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filtered_vcf = output_dir / f"{sample_id}_somatic_filtered.vcf.gz"
    
    # Build filter expression for bcftools
    filter_expr = f"FORMAT/DP >= {min_depth} && FORMAT/AD[1] >= {min_alt_reads}"
    
    cmd = [
        "bcftools", "filter",
        "-i", filter_expr,
        "-o", str(filtered_vcf),
        "-O", "z",
        str(vcf_file)
    ]
    log_command(logger, " ".join(cmd), sample_id=sample_id)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=1800)
        # Index the result
        subprocess.run(["tabix", "-p", "vcf", str(filtered_vcf)], check=True, timeout=300)
        logger.info("Somatic filtering completed", sample_id=sample_id, filtered_vcf=str(filtered_vcf))
        return filtered_vcf
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Somatic filtering failed: {e.stderr}")