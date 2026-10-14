#!/bin/bash

# Run a batch of FastQ files through the SRA to Features Pipeline

FASTQ_FOLDER=/mnt/d/cdi/chukwu/Features_2026_09_17/fastq/
#FASTQ_FOLDER=./data/
THREADS=6

OUTPUT_DIR=./results/grace_oct_05_2026/

# Get all FastQ files in the specified folder
FASTQ_FILES=("$FASTQ_FOLDER"*.fastq.gz)

# Group FastQ files by sample (assuming paired-end files start with SampleN-)
declare -A SAMPLES

for FILE in "${FASTQ_FILES[@]}"; do
    BASENAME=$(basename "$FILE")
    SAMPLE_NAME="${BASENAME%%_R*}" # Extract sample name before _R1 or _R2
    SAMPLES[$SAMPLE_NAME]="${SAMPLES[$SAMPLE_NAME]:+${SAMPLES[$SAMPLE_NAME]} }$FILE"
done

echo "Found ${#SAMPLES[@]} samples to process."

for SAMPLE in "${!SAMPLES[@]}"; do
    echo "Sample: $SAMPLE"
    for FILE in ${SAMPLES[$SAMPLE]}; do
        echo "  File: $FILE"
    done
done

# Create a file to store the run scripts
mkdir -p $OUTPUT_DIR
rm -f "$OUTPUT_DIR/run_scripts.sh"
echo "#!/bin/bash" > "$OUTPUT_DIR/run_scripts.sh"

# Process each sample
for SAMPLE in "${!SAMPLES[@]}"; do
    # Expand into array (unquoted!) so whitespace splits into elements
    FILES=(${SAMPLES[$SAMPLE]})

    if [ "${#FILES[@]}" -eq 2 ]; then
        echo "Processing paired-end sample: $SAMPLE"
        fastq1="${FILES[0]}"
        fastq2="${FILES[1]}"
        # echo script to a file
        echo "sra-pipeline run --fastq \"$fastq1\" --fastq \"$fastq2\" --output-dir \"$OUTPUT_DIR/$SAMPLE\" --log-level DEBUG --threads $THREADS --config config.ini --log-file \"$OUTPUT_DIR/${SAMPLE}_pipeline.log\"" >> "$OUTPUT_DIR/run_scripts.sh"
        sra-pipeline run \
            --fastq "$fastq1" \
            --fastq "$fastq2" \
            --output-dir "$OUTPUT_DIR/$SAMPLE" \
            --log-level DEBUG --threads $THREADS --config config.ini \
            --log-file "$OUTPUT_DIR/${SAMPLE}_pipeline.log"

    elif [ "${#FILES[@]}" -eq 1 ]; then
        echo "Processing single-end sample: $SAMPLE"
        fastq1="${FILES[0]}"

        # echo script to a file
        echo "sra-pipeline run --fastq \"$fastq1\" --output-dir \"$OUTPUT_DIR/$SAMPLE\" --log-level DEBUG --threads $THREADS --config config.ini --log-file \"$OUTPUT_DIR/${SAMPLE}_pipeline.log\"" >> "$OUTPUT_DIR/run_scripts.sh"

        sra-pipeline run \
            --fastq "$fastq1" \
            --output-dir "$OUTPUT_DIR/$SAMPLE" \
            --log-level DEBUG --threads $THREADS --config config.ini \
            --log-file "$OUTPUT_DIR/${SAMPLE}_pipeline.log"

    else
        echo "Warning: Sample $SAMPLE does not have the expected number of FastQ files."
    fi
done
