#!/usr/bin/env python3

import os
import sys
import shutil
import urllib.request
from pathlib import Path

def setup_reference(fasta_url, reference_fasta):
    """
    Download or copy reference FASTA file based on the input URL/path
    """
    # Clean the input by removing any surrounding quotes
    fasta_url = fasta_url.strip('"\'')
    # Check if it's a URL
    if fasta_url.startswith(('http://', 'https://')):
        print(f"Downloading reference FASTA from {fasta_url}...")
        try:
            if fasta_url.endswith('.gz'):
                # Download compressed file
                temp_file = reference_fasta + '.gz'
                urllib.request.urlretrieve(fasta_url, temp_file)
                print("Decompressing reference FASTA...")
                
                # Decompress using Python
                import gzip
                with gzip.open(temp_file, 'rb') as f_in:
                    with open(reference_fasta, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(temp_file)
            else:
                # Download uncompressed file
                urllib.request.urlretrieve(fasta_url, reference_fasta)
            
            print("Finished reference genome download.")
            return True
            
        except Exception as e:
            print(f"Error downloading from URL: {e}")
            return False
    
    else:
        # Handle local file path
        fasta_path = Path(fasta_url)
        
        # Check if file exists using multiple methods
        if fasta_path.exists():
            print(f"Copying reference genome from local file: {fasta_url}")
            try:
                shutil.copy2(fasta_url, reference_fasta)
                print("Finished copying reference genome from local file.")
                return True
            except Exception as e:
                print(f"Error copying file: {e}")
                return False
        else:
            # Try to resolve the path and provide better error message
            absolute_path = fasta_path.absolute()
            print(f"Error: File '{fasta_url}' does not exist or is not accessible.")
            print(f"Absolute path attempted: {absolute_path}")
            print(f"Current working directory: {os.getcwd()}")
            print(f"File exists: {os.path.exists(fasta_url)}")
            print(f"File readable: {os.access(fasta_url, os.R_OK) if os.path.exists(fasta_url) else 'N/A'}")
            return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python setup_reference.py <FASTA_URL_OR_PATH> <REFERENCE_FASTA_OUTPUT>")
        sys.exit(1)
    
    fasta_url = sys.argv[1]
    reference_fasta = sys.argv[2]
    
    success = setup_reference(fasta_url, reference_fasta)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
