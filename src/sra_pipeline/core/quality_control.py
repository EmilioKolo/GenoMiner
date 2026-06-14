"""
Quality control functionality for genomic data.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
from ..utils import log_command
import gzip
import numpy as np
import pysam
import re
import structlog
import subprocess
import zipfile


def run_quality_control(
    fastq_files: List[Path],
    bam_file: Path|None,
    output_dir: Path,
    logger: structlog.BoundLogger,
    post_trimming_fastq: Optional[List[Path]] = None,
) -> Dict[str, Any]:
    """
    Run quality control on FASTQ files.
    
    Args:
        fastq_files: List of FASTQ file paths
        output_dir: Output directory for QC results
        logger: Logger instance
        
    Returns:
        Dictionary containing quality control metrics
    """
    logger.info("Starting quality control", 
                fastq_files=[str(f) for f in fastq_files],
                output_dir=str(output_dir))
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    qc_results = {}
    
    # Initial FastQC on raw FASTQ
    qc_results.update(_run_fastqc(fastq_files, output_dir / "fastqc_raw", logger))
    
    # If post-trimming FASTQ files are provided, run FastQC on them
    if post_trimming_fastq:
        post_trim_dir = output_dir / "fastqc_trimmed"
        post_trim_dir.mkdir(parents=True, exist_ok=True)
        trimmed_qc = _run_fastqc(post_trimming_fastq, post_trim_dir, logger)
        # Store trimmed metrics under a separate key
        qc_results["post_trimming"] = trimmed_qc
        logger.info("Post-trimming FastQC completed", trimmed_files=[str(f) for f in post_trimming_fastq])
    
    # Calculate basic statistics
    if bam_file is not None:
        qc_results.update(_calculate_bam_stats(bam_file, logger))
    
    # Calculate quality scores
    qc_results.update(_calculate_quality_scores(qc_results, logger))
    
    logger.info("Quality control completed", 
                metrics=list(qc_results.keys()))
    
    return qc_results


def _run_fastqc(
    fastq_files: List[Path],
    output_dir: Path,
    logger: structlog.BoundLogger
) -> Dict[str, Any]:
    """Run FastQC on FASTQ files."""
    # Check for fastq file count
    if not fastq_files or len(fastq_files) > 2:
        logger.error("Invalid number of FASTQ files provided. Max 2 files allowed.",
                     fastq_files=[str(f) for f in fastq_files])
        return {}
    # Create the output directory if it does not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running FastQC analysis", 
                fastq_files=[str(f) for f in fastq_files])
    
    # Initialize the list of quality metrics
    fastqc_results: list[dict[str, Any]] = []
    # Go through FASTQ files
    for fastq_file in fastq_files:
        curr_fastq_results: dict[str, Any] = {}
        if not fastq_file.exists():
            logger.error("FASTQ file not found.", file=fastq_file)
            continue
        logger.info(f"Running FastQC for {str(fastq_file)}",
                    fastq_files=[str(f) for f in fastq_files])
        # Run FastQC
        try:
            cmd = ["fastqc", str(fastq_file), "-o", str(output_dir)]
            log_command(logger, " ".join(cmd),
                        fastq_files=[str(f) for f in fastq_files])
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"FastQC completed for {str(fastq_file)}",
                        fastq_files=[str(f) for f in fastq_files])
        except FileNotFoundError:
            logger.error("FastQC command not found. Ensure FastQC is installed and in your PATH.",
                         fastq_files=[str(f) for f in fastq_files])
            continue
        except subprocess.CalledProcessError as e:
            logger.error(f"FastQC exited with an error: {e}",
                         fastq_files=[str(f) for f in fastq_files])
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred during FastQC execution: {e}",
                         fastq_files=[str(f) for f in fastq_files])
            continue
        # Parse FastQC report
        base_name = str(fastq_file.stem).split('.fastq')[0]
        report_path: Path = output_dir / f"{base_name}_fastqc.zip"

        if not report_path.exists():
            logger.error("FastQC report not found.", 
                         fastq_files=[str(f) for f in fastq_files])
            continue

        logger.info(f"Parsing FastQC report: {str(report_path)}",
                    fastq_files=[str(f) for f in fastq_files])
        
        try:
            # Extract quality metrics
            curr_fastq_results = _extract_fastqc_metrics(report_path,
                                                         logger)
        except Exception as e:
            logger.error(f"Error parsing FastQC report: {e}",
                         fastq_files=[str(f) for f in fastq_files])
            continue

        logger.info(f"Processing fastq files for quality metrics.",
                    fastq_files=[str(f) for f in fastq_files])
        
        try:
            # Add basic fastq quality metrics
            curr_fastq_results.update(_process_fastq_basic(fastq_file,
                                                           logger))
        except Exception as e:
            logger.error(f"Error obtaining basic quality metrics: {e}",
                         fastq_files=[str(f) for f in fastq_files])
            continue
        fastqc_results.append(curr_fastq_results.copy())
    # Define values for the output
    output_dict = _define_output_dict(fastqc_results, logger)
    return output_dict


def _calculate_bam_stats(
    bam_file: Path,
    logger: structlog.BoundLogger
) -> Dict[str, Any]:
    """Calculate basic statistics from a BAM file."""
    logger.info("Calculating basic statistics", 
                bam_file=bam_file)
    return get_bam_statistics(bam_file, logger)


def _calculate_quality_scores(
    qc_results: Dict[str, Any],
    logger: structlog.BoundLogger
) -> Dict[str, Any]:
    """Calculate overall quality scores."""
    logger.info("Calculating quality scores")
    
    # Define thresholds
    # Overall cutoff for final score (fraction of maximum)
    OVERALL_CUTOFF = 0.6
    # Expected GC percentage
    EXPECTED_GC = 51.0 # 51 for human exome, 41 for human genome
    # Expected quality before raising a warning
    EXPECTED_QUAL = 30
    # Minimum accepted quality before raising an error
    MIN_QUAL = 20
    # Minimum expected read length before raising a warning
    EXPECTED_LEN = 90
    # Minimum percentage of non-duplicated nucleotides before raising a warning
    EXPECTED_DUP = 50
    # Expected BAM file coverage before raising a warning
    EXPECTED_COVER = 50
    # Minimum BAM file coverage before raising an error
    MIN_COVER = 10

    # Initialize warning and error lists
    l_warn = []
    l_error = []

    # Define base fastq variables
    gc_content = float(qc_results['gc_content'])
    dup_rate = qc_results['duplication_rate']
    # Define fastqc variables
    qual_score_dict = qc_results['quality_scores']
    l_per_base_qual = qual_score_dict['per_base_quality']
    l_per_seq_qual = qual_score_dict['per_sequence_quality']
    l_per_base_gc = qual_score_dict['per_base_gc_content']
    l_per_seq_gc = qual_score_dict['per_sequence_gc_content']
    l_per_base_n = qual_score_dict['per_base_n_content']
    l_seq_len = qual_score_dict['sequence_length_distribution']
    l_dup_seq = qual_score_dict['duplicate_sequences']
    l_dup_seq_uniq = qual_score_dict['duplicate_sequences_unique']
    l_num_over = qual_score_dict['number_of_overrepresented_sequences']
    l_rate_over = qual_score_dict['percent_of_overrepresented_sequences']
    l_adap_cont = qual_score_dict['adapter_content']
    # Define bam variables
    mapped_reads = qc_results['mapped_reads']
    mapping_rate = qc_results['mapping_rate']
    mean_cover = qc_results['mean_coverage']
    stdev_cover = qc_results['coverage_std']

    # Perform quality checks
    try:
        gc_warn, gc_error, gc_dict, gc_score = _gc_eval(
            gc_content, l_per_base_gc, l_per_seq_gc, EXPECTED_GC
        )
    except Exception as e:
        logger.warning(f'GC evaluation failed with error: {e}', 
                       gc_content=gc_content, l_per_base_gc=l_per_base_gc,
                       l_per_seq_gc=l_per_seq_gc)
        gc_warn = ['GC evaluation failed']
        gc_error = []
        gc_dict = {'per_base_gc_content':0, 'per_seq_gc_content':0}
        gc_score = 100
    try:
        qual_warn, qual_error, qual_dict, qual_score = _qual_eval(
            l_per_base_qual, l_per_seq_qual, EXPECTED_QUAL, MIN_QUAL
        )
    except Exception as e:
        logger.warning(f'Quality evaluation failed with error: {e}',
                       l_per_base_qual=l_per_base_qual,
                       l_per_seq_qual=l_per_seq_qual)
        qual_warn = ['Quality evaluation failed']
        qual_error = []
        qual_dict = {'per_base_qual':0, 'per_seq_qual':0}
        qual_score = 100
    try:
        dup_warn, dup_error, dup_dict, dup_score = _dup_eval(
            dup_rate, l_dup_seq, l_dup_seq_uniq, EXPECTED_DUP
        )
    except Exception as e:
        logger.warning(f'Duplication evaluation failed with error: {e}',
                       dup_rate=dup_rate, l_dup_seq=l_dup_seq,
                       l_dup_seq_uniq=l_dup_seq_uniq)
        dup_warn = ['Duplication evaluation failed']
        dup_error = []
        dup_dict = {'dup_seq':0, 'dup_seq_uniq':0}
        dup_score = 100
    try:
        n_warn, n_error, n_dict, n_score = _n_eval(
            l_per_base_n
        )
    except Exception as e:
        logger.warning(f'N evaluation failed with error: {e}',
                       l_per_base_n=l_per_base_n)
        n_warn = ['N evaluation failed']
        n_error = []
        n_dict = {'per_base_n':0}
        n_score = 100
    try:
        len_warn, len_error, len_dict, len_score = _len_eval(
            l_seq_len, EXPECTED_LEN
        )
    except Exception as e:
        logger.warning(f'Read length evaluation failed with error: {e}',
                       l_seq_len=l_seq_len)
        len_warn = ['Read length evaluation failed']
        len_error = []
        len_dict = {'seq_len':0}
        len_score = 100
    try:
        rep_warn, rep_error, rep_dict, rep_score = _rep_eval(
            l_num_over, l_rate_over, l_adap_cont
        )
    except Exception as e:
        logger.warning(f'Repeated nucleotide evaluation failed with error: {e}',
                       l_num_over=l_num_over, l_rate_over=l_rate_over,
                       l_adap_cont=l_adap_cont)
        rep_warn = ['Repeated nucleotide evaluation failed']
        rep_error = []
        rep_dict = {'num_overrep':0, 'rate_overrep':0, 'adap_cont':0}
        rep_score = 100
    try:
        bam_warn, bam_error, bam_score = _bam_eval(
            mapped_reads, mapping_rate, mean_cover, stdev_cover, 
            EXPECTED_COVER, MIN_COVER
        )
    except Exception as e:
        logger.warning(f'BAM evaluation failed with error: {e}',
                       mapped_reads=mapped_reads, mapping_rate=mapping_rate,
                       mean_cover=mean_cover, stdev_cover=stdev_cover)
        bam_warn = ['BAM evaluation failed']
        bam_error = []
        bam_score = 100
    # Join warning lists
    l_warn = l_warn + gc_warn + qual_warn + dup_warn + \
        n_warn + len_warn + rep_warn + bam_warn
    l_error = l_error + gc_error + qual_error + dup_error + \
        n_error + len_error + rep_error + bam_error
    # Calculate overall quality
    overall_quality = (gc_score + qual_score + dup_score + \
        n_score + len_score + rep_score + bam_score) / 700
    passes_quality = overall_quality > OVERALL_CUTOFF
    if not passes_quality:
        l_error.append(f'Overall quality is lower than {OVERALL_CUTOFF}')
    # Populate the output dict
    out_dict = {
        "overall_quality_score": overall_quality,
        "passes_quality_thresholds": passes_quality,
        "quality_warnings": l_warn,
        "quality_failures": l_error,
        "quality_scores": {
            "per_base_quality": qual_dict['per_base_qual'],
            "per_sequence_quality": qual_dict['per_seq_qual'],
            "per_base_gc_content": gc_dict['per_base_gc_content'],
            "per_sequence_gc_content": gc_dict['per_seq_gc_content'],
            "per_base_n_content": n_dict['per_base_n'],
            "sequence_length_distribution": len_dict['seq_len'],
            "duplicate_sequences": dup_dict['dup_seq'],
            "duplicate_sequences_unique": dup_dict['dup_seq_uniq'],
            "number_of_overrepresented_sequences": rep_dict['num_overrep'],
            "percent_of_overrepresented_sequences": rep_dict['rate_overrep'],
            "adapter_content": rep_dict['adap_cont']
        }
    }
    return out_dict


def _extract_fastqc_metrics(
    report_path: Path,
    logger: structlog.BoundLogger
) -> dict[str, Any]:
    """Extract quality metrics from the FastQC report."""
    # Initialize metrics dictionary
    file_metrics = {
        "gc_content": None,
        "duplication_rate": None,
        "quality_scores": {
            "per_base_quality": [],
            "per_sequence_quality": [],
            "per_base_gc_content": [],
            "per_sequence_gc_content": [],
            "per_base_n_content": [],
            "sequence_length_distribution": [],
            "duplicate_sequences": [],
            "number_of_overrepresented_sequences": [],
            "adapter_content": []
        }
    }
    # Open report file
    try:
        with zipfile.ZipFile(report_path, 'r') as zf:
            # Find the fastqc_data.txt file within the zip file
            data_file_name = None
            for name in zf.namelist():
                if name.endswith('fastqc_data.txt'):
                    data_file_name = name
                    break
            # Check that the file was found
            if not data_file_name:
                logger.error(
                    "fastqc_data.txt not found in FastQC zip archive.",
                    report_path=report_path
                )
                return file_metrics
            # Open the specific fastqc_data.txt file
            with zf.open(data_file_name) as f:
                content = f.read().decode('utf-8')
                lines = content.splitlines()
            # Initialize module variables
            current_module = None
            module_data_lines = []
            # Go through fastqc_data.txt text lines
            for line in lines:
                if line.startswith('>>'):
                    # Process previous module before starting a new one
                    if current_module and module_data_lines:
                        file_metrics = _process_module_data(
                            current_module,
                            module_data_lines,
                            file_metrics,
                            logger
                        )
                    # Extract new module name
                    module_name_match = re.match(r'>>([^\t]+)', line)
                    if module_name_match:
                        current_module = module_name_match.group(1).strip()
                        current_module = current_module.replace(' ', '_')
                    else:
                        current_module = None # Should not happen
                        logger.warning(
                            f'Current module does not match: {line}',
                            report_path=report_path
                        )
                    # Reset for new module
                    module_data_lines = []
                elif line.strip() == ">>END_MODULE":
                    # End of a module, process its data
                    if current_module and module_data_lines:
                        file_metrics = _process_module_data(
                            current_module,
                            module_data_lines,
                            file_metrics,
                            logger
                        )
                    # Reset module state
                    current_module = None
                    module_data_lines = []
                elif current_module and not line.startswith('#') and \
                     not line.strip() == "":
                    # Collect data lines belonging to the current module
                    # Ignores comments and empty lines
                    module_data_lines.append(line.strip())
            
            # Process any remaining data from the last module in the file
            if current_module and module_data_lines:
                file_metrics = _process_module_data(
                    current_module,
                    module_data_lines,
                    file_metrics,
                    logger
                )
    except zipfile.BadZipFile:
        logger.error("Invalid FastQC zip file.", report_path=report_path)
        return file_metrics
    except FileNotFoundError:
        logger.error("FastQC zip file not found.", report_path=report_path)
        return file_metrics
    except Exception as e:
        logger.error(f"An error occurred during parsing FastQC zip report: {e}",
                     report_path=report_path)
        return file_metrics

    return file_metrics


def _process_module_data(
        module_name: str,
        data_lines: List[str],
        metrics: Dict[str, Any],
        logger: structlog.BoundLogger
    ) -> Dict[str, Any]:
    """Function to process data lines for a specific FastQC module.
    It populates the metrics dictionary with the parsed data."""
    # Initialize the output dictionary
    dict_out = metrics.copy()
    if module_name == "Basic_Statistics":
        # Extract GC Content from Basic Statistics
        for line in data_lines:
            if line.startswith("%GC"):
                try:
                    # GC Content is the second tab-separated value
                    dict_out["gc_content"] = \
                        float(line.split('\t')[1].strip())
                except (ValueError, IndexError):
                    logger.warning(
                        "Could not parse GC Content from Basic_Statistics.",
                        line=line
                    )
    elif module_name == "Per_base_sequence_quality":
        # Data format: Base\tMean\tMedian\tLower Quartile\tUpper Quartile\t10th percentile\t90th percentile
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 7:
                try:
                    dict_out["quality_scores"]["per_base_quality"].append({
                        "base": parts[0],
                        "mean": float(parts[1]),
                        "median": float(parts[2]),
                        "lower_quartile": float(parts[3]),
                        "upper_quartile": float(parts[4]),
                        "10th_percentile": float(parts[5]),
                        "90th_percentile": float(parts[6]),
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Per_base_sequence_quality line.",
                        line=line
                    )
    elif module_name == "Per_sequence_quality_scores":
        # Data format: Quality\tCount
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 2:
                try:
                    dict_out["quality_scores"]["per_sequence_quality"].append({
                        "quality_score": int(parts[0]),
                        "count": float(parts[1])
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Per_sequence_quality_scores line.",
                        parts=parts
                    )
    elif module_name == "Per_base_sequence_content":
        # Data format: Base\tG_percent\tA_percent\tT_percent\tC_percent
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 5:
                try:
                    # Define percentage values
                    g_p = float(parts[1])
                    a_p = float(parts[2])
                    t_p = float(parts[3])
                    c_p = float(parts[4])
                    total = g_p+a_p+t_p+c_p
                    dict_out["quality_scores"]["per_base_gc_content"].append({
                        "base": parts[0],
                        "gc_percent": (g_p+c_p)/total*100.0
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Per_base_GC_content line.",
                        line=line
                    )
    elif module_name == "Per_sequence_GC_content":
        # Data format: GC_content\tCount
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 2:
                try:
                    dict_out["quality_scores"]["per_sequence_gc_content"].append({
                        "gc_content": int(parts[0]),
                        "count": float(parts[1])
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Per_sequence_GC_content line.",
                        parts=parts
                    )
    elif module_name == "Per_base_N_content":
        # Data format: Base\tN_percent
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 2:
                try:
                    dict_out["quality_scores"]["per_base_n_content"].append({
                        "base": parts[0],
                        "n_percent": float(parts[1])
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Per_base_N_content line.",
                        line=line
                    )
    elif module_name == "Sequence_Length_Distribution":
        # Data format: Length\tCount
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 2:
                try:
                    dict_out["quality_scores"]["sequence_length_distribution"].append({
                        "length_range": parts[0],
                        "count": float(parts[1])
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Sequence_Length_Distribution line.",
                        parts=parts
                    )
    elif module_name == "Sequence_Duplication_Levels":
        # Start a total duplicated percentage variable
        total_duplicated_percentage = 0.0
        for line in data_lines:
            # Data for duplicate sequences table: Duplication_level\tPercentage_of_reads\tPercentage_of_unique
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    duplication_level_str = parts[0].strip()
                    percentage_of_reads = float(parts[1])
                    percentage_of_unique = float(parts[2]) if len(parts) > 2 else None
                    
                    dict_out["quality_scores"]["duplicate_sequences"].append({
                        "duplication_level": duplication_level_str,
                        "percentage_of_reads": percentage_of_reads,
                        "percentage_of_unique": percentage_of_unique
                    })

                    # Calculate total_duplicated_percentage: Sum percentages for levels > 1
                    if duplication_level_str == '1':
                        # Reads with duplication level 1 are unique, not duplicated
                        pass
                    elif '-' in duplication_level_str:
                        # For ranges like '2-3', '4-5', etc., these are duplicated
                        total_duplicated_percentage += percentage_of_reads
                    elif duplication_level_str.startswith('>'):
                        # For '>10', also duplicated
                        total_duplicated_percentage += percentage_of_reads
                    else:
                        # For single number levels > 1 (e.g., '2', '3'), also duplicated
                        try:
                            if int(duplication_level_str) > 1:
                                total_duplicated_percentage += percentage_of_reads
                        except ValueError:
                            logger.warning(
                                f"Could not parse duplication level as int for calculation: {duplication_level_str}",
                                parts=parts
                            )
                except ValueError:
                    logger.warning(
                        "Could not parse duplicate_sequences line.",
                        line=line
                    )
        # Assign the calculated total duplicated percentage
        dict_out["duplication_rate"] = total_duplicated_percentage
    elif module_name == "Overrepresented_sequences":
        # Data format: Sequence\tCount\tPercentage\tPossible Source
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) == 4:
                try:
                    dict_out["quality_scores"]["number_of_overrepresented_sequences"].append({
                        "sequence": parts[0],
                        "count": int(parts[1]),
                        "percentage": float(parts[2]),
                        "possible_source": parts[3]
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Overrepresented_sequences line.",
                        line=line
                    )
    elif module_name == "Adapter_Content":
        # Data format: Position\tAdapter_Percentage
        for line in data_lines:
            parts = line.split('\t')
            if len(parts) >= 2:
                try:
                    dict_out["quality_scores"]["adapter_content"].append({
                        "position": parts[0],
                        "adapter_percentage": sum([float(x) for x in parts[1:]])
                    })
                except ValueError:
                    logger.warning(
                        "Could not parse Adapter_Content line.",
                        line=line
                    )
    else:
        logger.info(f'Module named {module_name} not processed',
                    module_name=module_name)
    return dict_out


def _process_fastq_basic(
    fastq_file: Path,
    logger: structlog.BoundLogger
) -> dict[str, int|float]:
    """Processes a FASTQ file to calculate basic metrics."""
    # Initialize dictionary to return
    dict_out: dict[str, int|float] = {}
    # Initialize values
    total_reads = 0
    total_bases = 0
    total_read_lengths = 0
    total_quality_scores = 0
    # Open the fastq file
    try:
        with gzip.open(fastq_file, 'rt') as f:
            for i, line in enumerate(f):
                # Sequence line
                if (i % 4) == 1:
                    sequence = line.strip()
                    read_length = len(sequence)
                    total_bases += read_length
                    total_read_lengths += read_length
                    total_reads += 1
                # Quality line
                elif (i % 4) == 3:
                    quality_string = line.strip()
                    # Convert ASCII quality scores to Phred scores
                    for char in quality_string:
                        total_quality_scores += ord(char) - 33
                if i==0 or ((i+1) % (4000 * 1000)) == 0:
                    logger.debug(f'### Processed sequences: {int(i/4)+1}',
                                 file=fastq_file)
    except FileNotFoundError:
        logger.error(f"Error: The file {fastq_file} was not found.",
                     file=fastq_file)
        return {}
    # Calculate averages
    if total_reads > 0:
        avg_read_length = total_read_lengths / total_reads
    else:
        avg_read_length = 0.0
    if total_reads > 0 and avg_read_length > 0:
        avg_quality_per_read = \
            (total_quality_scores / total_reads) / avg_read_length
    else:
        avg_quality_per_read = 0.0
    if total_bases > 0:
        avg_quality_per_base = total_quality_scores / total_bases
    else:
        avg_quality_per_base = 0.0
    # Load averages to dict_out
    dict_out['total_reads'] = total_reads
    dict_out['total_bases'] = total_bases
    dict_out['avg_read_length'] = avg_read_length
    dict_out['avg_quality_per_read'] = avg_quality_per_read
    dict_out['avg_quality_per_base'] = avg_quality_per_base
    return dict_out


def _define_output_dict(
    fastqc_results: list[Dict[str, Any]],
    logger: structlog.BoundLogger
) -> Dict[str, Any]:
    """Defines the output_dict from the fastqc_results list of 
    dictionaries."""
    # Define values for the output
    total_reads = sum_values(fastqc_results, 'total_reads', logger)
    total_bases = sum_values(fastqc_results, 'total_bases', logger)
    gc_cont = avg_values(fastqc_results, 'gc_content',
                         'total_bases', total_bases, logger)
    dup_rate = avg_values(fastqc_results, 'duplication_rate',
                          'total_bases', total_bases, logger)
    avg_read_length = float(total_bases) / float(total_reads)
    avg_qual_per_read = avg_values(fastqc_results, 'avg_quality_per_read',
                                   'total_reads', total_reads, logger)
    avg_qual_per_base = avg_values(fastqc_results, 'avg_quality_per_base',
                                   'total_bases', total_bases, logger)
    per_base_quality = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='per_base_quality',
        key_val='mean',
        key_id='base',
        avg_key='total_bases',
        total=total_bases,
        logger=logger
    )
    per_seq_quality = sum_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='per_sequence_quality',
        key_val='count',
        key_id='quality_score',
        logger=logger
    )
    per_base_gc_content = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='per_base_gc_content',
        key_val='gc_percent',
        key_id='base',
        avg_key='total_bases',
        total=total_bases,
        logger=logger
    )
    per_seq_gc_content = sum_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='per_sequence_gc_content',
        key_val='count',
        key_id='gc_content',
        logger=logger
    )
    per_base_n_content = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='per_base_n_content',
        key_val='n_percent',
        key_id='base',
        avg_key='total_bases',
        total=total_bases,
        logger=logger
    )
    seq_len_distribution = sum_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='sequence_length_distribution',
        key_val='count',
        key_id='length_range',
        logger=logger
    )
    overrep_seq = sum_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='number_of_overrepresented_sequences',
        key_val='count',
        key_id='sequence',
        logger=logger
    )
    overrep_seq_percent = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='number_of_overrepresented_sequences',
        key_val='percentage',
        key_id='sequence',
        avg_key='total_bases',
        total=total_bases,
        logger=logger
    )
    adap_content = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='adapter_content',
        key_val='adapter_percentage',
        key_id='position',
        avg_key='total_bases',
        total=total_bases,
        logger=logger
    )
    dup_seq_val = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='duplicate_sequences',
        key_val='percentage_of_reads',
        key_id='duplication_level',
        avg_key='total_reads',
        total=total_reads,
        logger=logger
    )
    dup_seq_unique = avg_list_val(
        fastqc_results,
        key1='quality_scores',
        key2='duplicate_sequences',
        key_val='percentage_of_unique',
        key_id='duplication_level',
        avg_key='total_reads',
        total=total_reads,
        logger=logger
    )
    output_dict = {
        "gc_content": gc_cont,
        "duplication_rate": dup_rate,
        "total_reads": total_reads,
        "total_bases": total_bases,
        "avg_read_length": avg_read_length,
        "avg_quality_per_read": avg_qual_per_read,
        "avg_quality_per_base": avg_qual_per_base,
        "quality_scores": {
            "per_base_quality": per_base_quality,
            "per_sequence_quality": per_seq_quality,
            "per_base_gc_content": per_base_gc_content,
            "per_sequence_gc_content": per_seq_gc_content,
            "per_base_n_content": per_base_n_content,
            "sequence_length_distribution": seq_len_distribution,
            "duplicate_sequences": dup_seq_val,
            "duplicate_sequences_unique": dup_seq_unique,
            "number_of_overrepresented_sequences": overrep_seq,
            "percent_of_overrepresented_sequences": overrep_seq_percent,
            "adapter_content": adap_content
        }
    }
    return output_dict


def get_bam_statistics(
    bam_file_path: Path,
    logger: structlog.BoundLogger
) -> dict[str, int|float]:
    """Extracts statistics from a BAM file including mapped reads, mapping
    percentage, mean coverage, and coverage standard deviation."""
    try:
        # Open the BAM file with pysam
        bam_file = pysam.AlignmentFile(str(bam_file_path), "rb")
        # Calculate mapped reads and mapping percentage
        total_reads: int = bam_file.mapped + bam_file.unmapped
        mapped_reads: int = bam_file.mapped
        if total_reads > 0:
            mapping_percentage = \
                (float(mapped_reads) / float(total_reads)) * 100.0
        else:
            mapping_percentage = 0
        # Calculate coverage statistics without processing the entire file
        sum_coverage = 0
        n_values = 0
        # Calculate the mean first
        for pileupcolumn in bam_file.pileup():
            sum_coverage += pileupcolumn.nsegments
            n_values += 1
        if n_values > 0:
            mean_coverage = sum_coverage / n_values
        else:
            mean_coverage = 0
        # Close the file
        bam_file.close()
        # Calculate standard deviation
        if n_values > 0:
            # Reopen the file
            bam_file = pysam.AlignmentFile(str(bam_file_path), "rb")
            sum_of_squared_diff = 0
            
            for pileupcolumn in bam_file.pileup():
                coverage = pileupcolumn.nsegments
                sum_of_squared_diff += (coverage - mean_coverage) ** 2
            # Calculate the variance and standard deviation
            variance = sum_of_squared_diff / n_values
            coverage_stdev = np.sqrt(variance)
            # Close the file
            bam_file.close()
        else:
            coverage_stdev = 0
        # Return results into a dictionary
        dict_out = {
            "mapped_reads": mapped_reads,
            "mapping_rate": mapping_percentage,
            "mean_coverage": mean_coverage,
            "coverage_std": coverage_stdev
        }
    except FileNotFoundError:
        logger.error(f"The file {bam_file_path} was not found. Ensure both the .bam and .bai files are present.",
                     bam_file=bam_file_path)
        dict_out = {}
    except pysam.utils.SamtoolsError as e:
        logger.error(f"Pysam error: {e}", bam_file=bam_file_path)
        dict_out = {}
    except Exception as e:
        logger.error(f"Unhandled error: {e}", bam_file=bam_file_path)
        dict_out = {}
    return dict_out


def avg_list_val(
    l_dict_val:list[dict],
    key1:str,
    key2:str,
    key_val:str,
    key_id:str,
    avg_key:str,
    total:int|float,
    logger: structlog.BoundLogger
) -> list[dict[str, int|float|str]]:
    """Averages values from a list of dictionaries inside a list of 
    dictionaries."""
    # Initialise return value
    l_out = []
    try:
        # Initialise an intermediary dictionary
        dict_hist = {}
        # Go through the list of dictionaries
        for dict_val in l_dict_val:
            weight: float = float(dict_val[avg_key]) / total
            # Get the list of dictionaries to apply the sum to
            l_dict_inner = dict_val[key1][key2]
            for hist_dict_val in l_dict_inner:
                hist_key = hist_dict_val[key_id]
                if hist_key in dict_hist.keys():
                    dict_hist[hist_key] += hist_dict_val[key_val] * weight
                else:
                    dict_hist[hist_key] = hist_dict_val[key_val] * weight
        # Rearrange dict_hist to be a list of dictionaries
        for key, val in dict_hist.items():
            l_out.append({key_id:key, key_val:val})
    except Exception as e:
        logger.warning(f'Error in avg_list_val: {e}', key1=key1, 
                       key2=key2, key_val=key_val, key_id=key_id)
    return l_out


def sum_list_val(
    l_dict_val:list[dict],
    key1:str,
    key2:str,
    key_val:str,
    key_id:str,
    logger: structlog.BoundLogger
) -> list[dict[str, int|float|str]]:
    """Sums values from a list of dictionaries inside a list of 
    dictionaries."""
    # Initialise return value
    l_out = []
    try:
        # Initialise an intermediary dictionary
        dict_hist = {}
        # Go through the list of dictionaries
        for dict_val in l_dict_val:
            # Get the list of dictionaries to apply the sum to
            l_dict_inner = dict_val[key1][key2]
            for hist_dict_val in l_dict_inner:
                hist_key = hist_dict_val[key_id]
                if hist_key in dict_hist.keys():
                    dict_hist[hist_key] += hist_dict_val[key_val]
                else:
                    dict_hist[hist_key] = hist_dict_val[key_val]
        # Rearrange dict_hist to be a list of dictionaries
        for key, val in dict_hist.items():
            l_out.append({key_id:key, key_val:val})
    except Exception as e:
        logger.warning(f'Error in sum_list_val: {e}', key1=key1, 
                       key2=key2, key_val=key_val, key_id=key_id)
    return l_out


def avg_values(
    l_dict_val:list[dict],
    key:str,
    avg_key:str,
    total:int|float,
    logger: structlog.BoundLogger
) -> int|float:
    """Averages the values from key inside a list of dictionaries.
    Uses avg_key as the value to divide by total for a weighted average."""
    # Initialise return value
    out_val = 0
    try:
        for dict_val in l_dict_val:
            weight: float = float(dict_val[avg_key]) / total
            out_val += dict_val[key] * weight
    except Exception as e:
        logger.warning(f'Error in avg_values: {e}', key=key)
    return out_val


def sum_values(
    l_dict_val:list[dict],
    key:str,
    logger: structlog.BoundLogger
) -> int|float:
    """Sums the values from key inside a list of dictionaries."""
    # Initialise return value
    out_val = 0
    try:
        for dict_val in l_dict_val:
            out_val += dict_val[key]
    except Exception as e:
        logger.warning(f'Error in sum_values: {e}', key=key)
    return out_val


def validate_quality_metrics(
    qc_results: Dict[str, Any],
    logger: structlog.BoundLogger,
    min_mapping_rate: float = 80.0,
    min_coverage: float = 10.0,
    max_duplication_rate: float = 20.0,
) -> bool:
    """
    Validate quality metrics against thresholds.
    
    Args:
        qc_results: Quality control results
        min_mapping_rate: Minimum mapping rate percentage
        min_coverage: Minimum coverage
        max_duplication_rate: Maximum duplication rate percentage
        logger: Logger instance
        
    Returns:
        True if all quality metrics pass thresholds
    """
    logger.info("Validating quality metrics", 
                min_mapping_rate=min_mapping_rate,
                min_coverage=min_coverage,
                max_duplication_rate=max_duplication_rate)
    
    # Check mapping rate
    mapping_rate = qc_results.get("mapping_rate", 0.0)
    if mapping_rate < min_mapping_rate:
        logger.warning("Mapping rate below threshold", 
                      mapping_rate=mapping_rate, 
                      threshold=min_mapping_rate)
        return False
    
    # Check coverage
    mean_coverage = qc_results.get("mean_coverage", 0.0)
    if mean_coverage < min_coverage:
        logger.warning("Coverage below threshold", 
                      mean_coverage=mean_coverage, 
                      threshold=min_coverage)
        return False
    
    # Check duplication rate
    duplication_rate = qc_results.get("duplication_rate", 100.0)
    if duplication_rate > max_duplication_rate:
        logger.warning("Duplication rate above threshold", 
                      duplication_rate=duplication_rate, 
                      threshold=max_duplication_rate)
        return False
    
    logger.info("Quality metrics validation passed")
    return True


def _gc_eval(
    gc_content: float|int,
    l_per_base_gc: List,
    l_per_seq_gc: List,
    expected_gc: float|int
) -> tuple[List, List, Dict, float|int]:
    """Evaluates GC percent values.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []
    error_val = 0
    # Check the difference between gc_content and expected_gc
    gc_dif = abs(gc_content - expected_gc)
    # Define warning or error
    if gc_dif > 10:
        l_error.append(f'GC content error: {gc_content}')
        error_val += 25
    elif gc_dif > 5:
        l_warn.append(f'GC content warning: {gc_content}')
        error_val += 10
    # Check per base gc content
    cont_ok = 0
    for i in range(len(l_per_base_gc)):
        curr_dict = l_per_base_gc[i]
        curr_base = curr_dict['base']
        curr_perc = curr_dict['gc_percent']
        perc_dif = abs(curr_perc - expected_gc)
        if perc_dif > 20:
            l_error.append(f'Per base GC content error in base {curr_base}: {curr_perc}')
        elif perc_dif > 10:
            l_warn.append(f'Per base GC content warning in base {curr_base}: {curr_perc}')
        else:
            cont_ok += 1
    fraction_ok = cont_ok / float(cont_ok + len(l_error) + len(l_warn))
    # Check per sequence gc content
    count_max = 0
    max_count_base = -1
    for i in range(len(l_per_seq_gc)):
        curr_dict = l_per_seq_gc[i]
        curr_gc_str = curr_dict['gc_content']
        # Check for range string
        if '-' in str(curr_gc_str):
            curr_gc = sum(str_to_range(curr_gc_str))/2.0
        else:
            curr_gc = int(curr_gc_str)
        curr_count = curr_dict['count']
        # Update count_max
        if curr_count > count_max:
            count_max = float(curr_count)
            max_count_base = int(curr_gc)
    # Check base with max count
    gc_dif_max = abs(max_count_base - expected_gc)
    # Define warning or error
    if gc_dif_max > 5:
        l_error.append(f'GC content with most counts error: {max_count_base}')
        error_val += 25
    elif gc_dif_max > 2:
        l_warn.append(f'GC content with most counts warning: {max_count_base}')
        error_val += 10
    # Define return dictionary
    ret_dict = {
        'per_base_gc_content': fraction_ok * 100,
        'per_seq_gc_content': max_count_base
    }
    # Define score value
    score_val = 50 * fraction_ok + 50 - error_val
    return l_warn, l_error, ret_dict, score_val


def _qual_eval(
    l_per_base_qual: List,
    l_per_seq_qual: List,
    expected_qual: int,
    min_qual: int
) -> tuple[List, List, Dict, float|int]:
    """Evaluates quality values.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []
    # Check quality per base position
    cont_ok = 0
    for i in range(len(l_per_base_qual)):
        curr_dict = l_per_base_qual[i]
        curr_base = curr_dict['base']
        curr_qual = curr_dict['mean']
        if curr_qual < min_qual:
            l_error.append(f'Quality error in base {curr_base}: {curr_qual}')
        elif curr_qual < expected_qual:
            l_warn.append(f'Quality warning in base {curr_base}: {curr_qual}')
        else:
            cont_ok += 1
    fract_ok = cont_ok / float(cont_ok + len(l_error) + len(l_warn))
    # Check counts of quality values per read
    count_total = 0
    count_low = 0
    for i in range(len(l_per_seq_qual)):
        curr_dict = l_per_seq_qual[i]
        curr_qual = curr_dict['quality_score']
        curr_count = curr_dict['count']
        if curr_qual < min_qual:
            count_low += curr_count
        count_total += curr_count
    fraction_low = float(count_low) / count_total
    # Check if the fraction of low quality reads is too high
    if fraction_low > 0.5:
        l_error.append(f'{fraction_low*100}% of reads have quality lower than: {min_qual}')
    elif fraction_low > 0.25:
        l_warn.append(f'{fraction_low*100}% of reads have quality lower than: {min_qual}')
    # Define return dictionary
    ret_dict = {
        'per_base_qual': fract_ok * 100,
        'per_seq_qual': (1 - fraction_low) * 100
    }
    # Define score value
    score_val = 100 * fract_ok * (1 - fraction_low)
    return l_warn, l_error, ret_dict, score_val


def _dup_eval(
    dup_rate: int|float,
    l_dup_seq: List,
    l_dup_seq_uniq: List,
    expected_dup: float
) -> tuple[List, List, Dict, float|int]:
    """Evaluates duplication values.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []
    # Go through l_dup_seq
    dup_lvl_1 = 0
    for i in range(len(l_dup_seq)):
        curr_dict = l_dup_seq[i]
        curr_dup_lvl = curr_dict['duplication_level']
        curr_percent = curr_dict['percentage_of_reads']
        # Record duplication_level = 1
        if str(curr_dup_lvl)=='1':
            dup_lvl_1 = curr_percent
    # Go through l_dup_seq_uniq
    dup_lvl_1_uniq = 0
    for i in range(len(l_dup_seq_uniq)):
        curr_dict = l_dup_seq_uniq[i]
        curr_dup_lvl = curr_dict['duplication_level']
        curr_percent = curr_dict['percentage_of_unique']
        # Record duplication_level = 1
        if str(curr_dup_lvl)=='1':
            dup_lvl_1_uniq = curr_percent
    ### DEBUG: Check duplication variable values
    dif_dup_rate = 100 - dup_lvl_1 - dup_rate
    if dif_dup_rate > 0.5:
        print(f'WARNING: Duplication rate', dup_rate, 
              'not consistent with dup_lvl_1 values:', 100 - dup_lvl_1)
    ###
    # Check that duplication level 1 values are high
    if dup_lvl_1 < expected_dup:
        l_warn.append(f'Duplication rate very high: {100-dup_lvl_1}')
    if dup_lvl_1_uniq < expected_dup:
        l_warn.append(f'Unique sequence duplication rate very high: {100-dup_lvl_1_uniq}')
    # Define return dictionary
    ret_dict = {
        'dup_seq': 100-dup_lvl_1,
        'dup_seq_uniq': 100-dup_lvl_1_uniq
    }
    # Define score value
    score_val = 100 * dup_lvl_1
    return l_warn, l_error, ret_dict, score_val


def _n_eval(
    l_per_base_n: List
) -> tuple[List, List, Dict, float|int]:
    """Evaluates the number of N sequences.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []
    # Go through l_per_base_n
    total_n_percent = 0.0
    for i in range(len(l_per_base_n)):
        curr_dict = l_per_base_n[i]
        curr_base = curr_dict['base']
        curr_percent = curr_dict['n_percent']
        # Add percent to total
        total_n_percent += float(curr_percent)
    # Check if total_n_percent is too high
    if total_n_percent > 1.0:
        l_error.append(f'N percent too high: {total_n_percent}%')
    elif total_n_percent > 0.1:
        l_warn.append(f'N percent too high: {total_n_percent}%')
    # Define return dictionary
    ret_dict = {
        'per_base_n': total_n_percent
    }
    # Define score value
    score_val = 100 * (1 - (total_n_percent/100))
    return l_warn, l_error, ret_dict, score_val


def _len_eval(
    l_seq_len: List,
    expected_len: int
) -> tuple[List, List, Dict, float|int]:
    """Evaluates read length values.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []

    # Go through l_seq_len
    total_count = 0.0
    total_len = 0
    short_count = 0
    n_len = len(l_seq_len)
    for i in range(n_len):
        curr_dict = l_seq_len[i]
        curr_count = float(curr_dict['count'])
        curr_str = curr_dict['length_range']
        # Get length from length range string
        if '-' in curr_str:
            curr_len_range = str_to_range(curr_str)
            curr_len = (curr_len_range[0] + curr_len_range[1])/2
        else:
            curr_len = float(curr_str)
        # Check if curr_len is shorter than expected_len
        if curr_len < expected_len:
            short_count += curr_count
        # Add current count to total
        total_count += curr_count
        # Add total length count to total
        total_len += curr_count*curr_len
    # Calculate short read fraction
    short_count_fraction = short_count / total_count
    short_count_percent = short_count_fraction * 100
    # Define average seq length
    avg_seq_len = total_len/total_count
    # Check for warnings/errors
    if avg_seq_len < expected_len:
        l_warn.append(f'Average sequence length below expected: {avg_seq_len}')
    if short_count_percent > 50:
        l_warn.append(f'More than half of sequences below expected: {short_count_percent}%')
    # Define return dictionary
    ret_dict = {
        'seq_len': avg_seq_len
    }
    # Define score value
    score_val = 100 - short_count_percent
    return l_warn, l_error, ret_dict, score_val


def _rep_eval(
    l_num_over: List,
    l_rate_over: List,
    l_adap_cont: List
) -> tuple[List, List, Dict, float|int]:
    """Evaluates the number of repeats and adapters.
    Returns a list of warnings, a list of errors, a dict of quality
    values and a score with a max of 100."""
    l_warn = []
    l_error = []
    # Go through l_num_over
    overrep_count = 0.0
    for i in range(len(l_num_over)):
        curr_dict = l_num_over[i]
        curr_seq = curr_dict['sequence']
        curr_count = curr_dict['count']
        # Add overrepresented count to total
        overrep_count += curr_count
    # Go through l_rate_over
    total_overrep = 0.0
    for i in range(len(l_rate_over)):
        curr_dict = l_rate_over[i]
        curr_seq = curr_dict['sequence']
        curr_percent = curr_dict['percentage']
        # Add overrepresented percent to total
        total_overrep += curr_percent
        # Check if curr_percent is too high
        if curr_percent > 1.0:
            l_warn.append(f'Following sequence found on {curr_percent}% of reads: {curr_seq}')
    # Go through l_adap_cont
    adap_total_percent = 0.0
    for i in range(len(l_adap_cont)):
        curr_dict = l_adap_cont[i]
        curr_pos = curr_dict['position']
        curr_percent = curr_dict['adapter_percentage']
        # Add adapter percent to total
        adap_total_percent += curr_percent
    # Check for warnings and errors
    if total_overrep > 5.0:
        l_error.append(f'Overrepresented sequences in {total_overrep}% of sequences.')
    elif total_overrep > 1.0:
        l_warn.append(f'Overrepresented sequences in {total_overrep}% of sequences.')
    if adap_total_percent > 5.0:
        l_error.append(f'Adapter content is too high: {total_overrep}%')
    elif adap_total_percent > 1.0:
        l_warn.append(f'Adapter content is slightly high: {total_overrep}%')
    # Define return dictionary
    ret_dict = {
        'num_overrep': overrep_count,
        'rate_overrep': total_overrep,
        'adap_cont': adap_total_percent
    }
    # Define score value
    score_val = 100 - adap_total_percent - total_overrep
    return l_warn, l_error, ret_dict, score_val


def _bam_eval(
    mapped_reads: int,
    mapping_rate: float,
    mean_cover: float,
    stdev_cover: float,
    expected_cover: int|float,
    min_cover: int|float
) -> tuple[List, List, float|int]:
    """Evaluates the quality of the bam file.
    Returns a list of warnings, a list of errors and a score 
    with a max of 100."""
    l_warn = []
    l_error = []
    error_val = 0
    # Check mapping percentage
    if mapping_rate < 10:
        l_warn.append(f'Mapping percentage very low: {mapping_rate}%')
        error_val += 20
    # Check mean coverage
    if mean_cover < min_cover:
        l_error.append(f'Mean coverage too low: {mean_cover}')
        error_val += 60
    elif mean_cover < expected_cover:
        l_warn.append(f'Mean coverage is low: {mean_cover}')
        error_val += 20
    # Check standard deviation
    if (mean_cover - stdev_cover) < min_cover:
        l_warn.append(f'Standard deviation is very high ({stdev_cover}) considering mean coverage ({mean_cover})')
        error_val += 20
    # Define score value
    score_val = 100 - error_val
    return l_warn, l_error, score_val


def str_to_range(range_str: str) -> List[int|float]:
    """Transforms a string with a range into a tuple with 
    an initial and end value.
    Splits by middle dash ('-')."""
    # Initialize the output tuple
    out_l:List[int|float] = [-1, -1]
    l_split = range_str.split('-')
    out_l[0] = float(l_split[0])
    out_l[1] = float(l_split[-1])
    return out_l