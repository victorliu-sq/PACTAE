# BambooSMP

BambooSMP is a high-performance parallel processing framework designed to efficiently solve the Stable Marriage
Problem (SMP). Named after the resilient and fast-growing bamboo plant, BambooSMP aims to deliver robust performance
across various challenging scenarios.

## Installation Guide

### Hardware Requirements

    At least 1 NVIDIA GPU.

### Software Requirements

Please ensure your system meets the following minimum software versions (greater than or equal to these values):

    bash
    wget or curl
    perf 
    ncu 
    CUDA: 12.6
    GCC: 11.4.0
    CMake: 3.22.1
    Python: 3.10

Note: perf and ncu must be installed and accessible in your system's PATH. If they are missing, consult your system
administrator or package manager.

---

### Setup & Execution

To run BambooSMP, simply execute the following from the project root directory:

    ./runme.sh

The script will automatically handle dependencies downloads (requires wget or curl), installation, compilation, and
execution of workloads. The resulting `figure3`, `figure5`, `table1`, `figure7`, `figure8`, and `figure9` will be stored
in the `data/figures` directory.