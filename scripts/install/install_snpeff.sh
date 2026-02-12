#!/bin/bash

# This script installs snpEff, a genetic variant annotation and effect prediction tool.

set -euo pipefail

# Get the directory into which install snpEff from command line argument
SNPEFF_DIR=${1:-"/usr/local/snpEff"}
# Get the snpEff version to install from command line argument
SNPEFF_URL=${2:-"https://sourceforge.net/projects/snpeff/files/snpEff_v5_0_core.zip/download"}
# Get the custom genome name and variables from command line argument
SNPEFF_GENOME_NAME=${3:-"custom_genome"}
REF_FASTA=${4:-"./reference.fasta"}
REF_GFF=${5:-"./annotations.gff"}
# Define snpEff jar file path
SNPEFF_JAR="$SNPEFF_DIR/snpEff/snpEff.jar"

echo "Installing snpEff to $SNPEFF_DIR..."

# Create installation directory if it doesn't exist
mkdir -p "$SNPEFF_DIR"
# Download snpEff
wget -O /tmp/snpeff.zip "$SNPEFF_URL"
# Unzip the downloaded file
unzip -o /tmp/snpeff.zip -d "$SNPEFF_DIR"
# Clean up
rm /tmp/snpeff.zip

echo "snpEff installation completed. Creating custom genome database..."

# Create custom genome for snpeff
mkdir -p "$SNPEFF_DIR/snpEff/data/$SNPEFF_GENOME_NAME"
cp "$REF_FASTA" "$SNPEFF_DIR/snpEff/data/$SNPEFF_GENOME_NAME/sequences.fa"
cp "$REF_GFF" "$SNPEFF_DIR/snpEff/data/$SNPEFF_GENOME_NAME/genes.gff"
echo "${SNPEFF_GENOME_NAME}.genome : Custom genome" >> "$SNPEFF_DIR/snpEff/snpEff.config"
java -Xmx4g -jar "$SNPEFF_JAR" build -gff3 -v "$SNPEFF_GENOME_NAME"

echo "Finished snpEff custom genome setup."
