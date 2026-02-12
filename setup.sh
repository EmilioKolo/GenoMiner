#!/bin/bash

# Exit on error and undefined variables
set -euo pipefail

# Define the config file
CONFIG_FILE="config.ini"
# Get relevant variables
BASE_DIR=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Paths" "BASE_DIR")
if [ -z "$BASE_DIR" ]; then
    echo "Error: BASE_DIR not found in config.ini"
    exit 1
fi
SNPEFF_DIR=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Paths" "SNPEFF_DIR")
if [ -z "$SNPEFF_DIR" ]; then
    echo "Error: SNPEFF_DIR not found in config.ini"
    exit 1
fi
GENOME_NAME=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Parameters" "GENOME_NAME")
if [ -z "$GENOME_NAME" ]; then
    echo "Error: GENOME_NAME not found in config.ini"
    exit 1
fi
THREADS=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Parameters" "THREADS")
if [ -z "$THREADS" ]; then
    echo "Error: THREADS not found in config.ini"
    exit 1
fi
FASTA_URL=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Links" "FASTA_URL")
if [ -z "$FASTA_URL" ]; then
    echo "Error: FASTA_URL not found in config.ini"
    exit 1
fi
GFF_URL=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Links" "GFF_URL")
if [ -z "$GFF_URL" ]; then
    echo "Error: GFF_URL not found in config.ini"
    exit 1
fi
SNPEFF_URL=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Links" "SNPEFF_URL")
if [ -z "$SNPEFF_URL" ]; then
    echo "Error: SNPEFF_URL not found in config.ini"
    exit 1
fi
REFERENCE_FASTA=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Paths" "REFERENCE_FASTA")
if [ -z "$REFERENCE_FASTA" ]; then
    echo "Error: REFERENCE_FASTA not found in config.ini"
    exit 1
fi
REFERENCE_GFF=$(python3 scripts/get_config_value.py "$CONFIG_FILE" "Paths" "REFERENCE_GFF")
if [ -z "$REFERENCE_GFF" ]; then
    echo "Error: REFERENCE_GFF not found in config.ini"
    exit 1
fi

# Make sure that BASE_DIR exists
mkdir -p "$BASE_DIR"

# Create data, install, logs, bin and tmp directories
DATA_DIR="$BASE_DIR/data"
INSTALL_DIR="$BASE_DIR/install"
BIN_DIR="$BASE_DIR/bin"
mkdir -p "$DATA_DIR"
mkdir -p "$INSTALL_DIR"
mkdir -p "$BIN_DIR"
# Define snpEff variables
snpeff_jar="$SNPEFF_DIR/snpEff/snpEff.jar"
# Add the bin directory to PATH
export PATH="$BIN_DIR:$PATH"

# Function to download with either curl or wget (whichever exists)
download() {
    if command -v wget >/dev/null; then
        wget "$1" -O "$2"
    elif command -v curl >/dev/null; then
        curl -L "$1" -o "$2"
    else
        echo "ERROR: Need either curl or wget to download files."
        exit 1
    fi
}
# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

log "Starting reference genome download."
# Download reference genome

log "Setting up reference FASTA..."

python3 ./scripts/install/setup_reference.py "$FASTA_URL" "$REFERENCE_FASTA"

log "Setting up reference GFF..."

python3 ./scripts/install/setup_reference.py "$GFF_URL" "$REFERENCE_GFF"


log "Starting snpEff download."

# Install snpEff
mkdir -p "$SNPEFF_DIR"
download "$SNPEFF_URL" "$SNPEFF_DIR/snpEff.zip"
unzip -o "$SNPEFF_DIR/snpEff.zip" -d "$SNPEFF_DIR"
rm -f "$SNPEFF_DIR/snpEff.zip"
log "Finished snpEff download. Starting snpEff custom genome setup."
# Create custom genome for snpeff
mkdir -p "$SNPEFF_DIR/snpEff/data/$GENOME_NAME"
cp "$REFERENCE_FASTA" "$SNPEFF_DIR/snpEff/data/$GENOME_NAME/sequences.fa"
cp "$REFERENCE_GFF" "$SNPEFF_DIR/snpEff/data/$GENOME_NAME/genes.gff"
echo "${GENOME_NAME}.genome : Custom genome" >> "$SNPEFF_DIR/snpEff/snpEff.config"
java -Xmx4g -jar "$snpeff_jar" build -gff3 -v "$GENOME_NAME"

log "Finished snpEff custom genome setup. Starting BWA indexing."
# Index reference genome with bwa
bwa index "$REFERENCE_FASTA"

### Temporarily disabled steps. Will be re-enabled in future versions.
log "Finished BWA indexing. Starting cnvpytor data download."
# Download cnvpytor data
cnvpytor -download

log "Finished cnvpytor data download."
