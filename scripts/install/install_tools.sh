#!/bin/bash

# SRA to Features Pipeline - Bioinformatics Tools Installer
# This script helps install the required bioinformatics tools

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check tool version
check_tool_version() {
    local tool=$1
    local min_version=$2
    local version_cmd=$3
    
    if command_exists "$tool"; then
        local version=$($version_cmd 2>&1 | head -n 1)
        print_success "$tool is installed: $version"
        return 0
    else
        print_warning "$tool is not installed or not in PATH"
        return 1
    fi
}

# Function to install tools based on OS
install_tools() {
    local os_type=$1
    
    case $os_type in
        "ubuntu"|"debian")
            install_tools_apt
            ;;
        "centos"|"rhel"|"fedora")
            install_tools_yum
            ;;
        "macos")
            install_tools_macos
            ;;
        *)
            print_error "Unsupported operating system: $os_type"
            exit 1
            ;;
    esac
}

# Install tools using apt (Ubuntu/Debian)
install_tools_apt() {
    print_status "Installing tools using apt..."
    
    # Update package list
    sudo apt-get update
    
    # Install basic tools
    sudo apt-get install -y \
        build-essential \
        wget \
        curl \
        git \
        unzip \
        default-jre \
        python3-pip
    
    # Install bioinformatics tools
    sudo apt-get install -y \
        bwa \
        samtools \
        bcftools \
        bedtools \
        fastqc \
        sra-toolkit \
        tabix \
        htslib
    
    print_success "Tools installed via apt"
}

# Install tools using yum (CentOS/RHEL/Fedora)
install_tools_yum() {
    print_status "Installing tools using yum..."
    
    # Install EPEL repository (for CentOS/RHEL)
    if [[ "$os_type" == "centos" || "$os_type" == "rhel" ]]; then
        sudo yum install -y epel-release
    fi
    
    # Install basic tools
    sudo yum install -y \
        gcc \
        gcc-c++ \
        make \
        wget \
        curl \
        git \
        unzip \
        java-1.8.0-openjdk \
        python3-pip
    
    # Install bioinformatics tools
    sudo yum install -y \
        bwa \
        samtools \
        bcftools \
        bedtools \
        fastqc \
        sra-toolkit \
        htslib
    
    print_success "Tools installed via yum"
}

# Install tools on macOS
install_tools_macos() {
    print_status "Installing tools on macOS..."
    
    # Check if Homebrew is installed
    if ! command_exists brew; then
        print_error "Homebrew is not installed. Please install it first:"
        echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        exit 1
    fi
    
    # Install basic tools
    brew install \
        wget \
        curl \
        git \
        unzip
    
    # Install bioinformatics tools
    brew install \
        bwa \
        samtools \
        bcftools \
        bedtools \
        fastqc \
        sratoolkit \
        htslib
    
    print_success "Tools installed via Homebrew"
}

# Install CNVpytor
install_cnvpytor() {
    print_status "Installing CNVpytor..."
    
    if command_exists pip3; then
        pip3 install cnvpytor==1.3.1
        print_success "CNVpytor installed via pip"
    else
        print_warning "pip3 not found, skipping CNVpytor installation"
    fi
}

# Main function
main() {
    echo "=========================================="
    echo "SRA to Features Pipeline - Tool Installer"
    echo "=========================================="
    echo
    
    # Detect operating system
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            os_type="ubuntu"
        elif command_exists yum; then
            os_type="centos"
        else
            print_error "Unsupported Linux distribution"
            exit 1
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        os_type="macos"
    else
        print_error "Unsupported operating system: $OSTYPE"
        exit 1
    fi
    
    print_status "Detected OS: $os_type"
    
    # Check existing tools
    echo
    print_status "Checking existing tools..."
    
    local missing_tools=()
    
    check_tool_version "bwa" "0.7.17" "bwa 2>&1" || missing_tools+=("bwa")
    check_tool_version "samtools" "1.10" "samtools --version" || missing_tools+=("samtools")
    check_tool_version "bcftools" "1.10" "bcftools --version" || missing_tools+=("bcftools")
    check_tool_version "bedtools" "2.29" "bedtools --version" || missing_tools+=("bedtools")
    check_tool_version "fastqc" "0.11" "fastqc --version" || missing_tools+=("fastqc")
    check_tool_version "fastq-dump" "2.10" "fastq-dump --version" || missing_tools+=("sra-toolkit")
    check_tool_version "tabix" "1.10" "tabix 2>&1" || missing_tools+=("htslib")
    check_tool_version "java" "1.8" "java -version" || missing_tools+=("java")
    
    # Install missing tools
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        echo
        print_status "Missing tools: ${missing_tools[*]}"
        echo
        read -p "Do you want to install missing tools? (y/N): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            install_tools "$os_type"
            install_cnvpytor
            
            echo
            print_status "Re-checking tools after installation..."
            check_tool_version "bwa" "0.7.17" "bwa 2>&1"
            check_tool_version "samtools" "1.10" "samtools --version"
            check_tool_version "bcftools" "1.10" "bcftools --version"
            check_tool_version "bedtools" "2.29" "bedtools --version"
            check_tool_version "fastqc" "0.11" "fastqc --version"
            check_tool_version "fastq-dump" "2.10" "fastq-dump --version"
            check_tool_version "tabix" "1.10" "tabix 2>&1"
            check_tool_version "java" "1.8" "java -version"
        else
            print_warning "Skipping tool installation"
        fi
    else
        print_success "All required tools are already installed!"
    fi
    
    echo
    print_status "Installation complete!"
    print_status "Next steps:"
    echo "  1. Install the Python package: pip install sra-to-features-pipeline"
    echo "  2. Download reference genomes and annotation files"
    echo "  3. Configure the pipeline using the config.ini file"
    echo "  4. Run: sra-pipeline validate"
    echo
}

# Run main function
main "$@" 