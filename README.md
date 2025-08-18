# PACT2025 Artifact Evaluation Guide

## Hardware Requirements

- At least one NVIDIA GPU, with each Streaming Multiprocessor (SM) supporting a block size of at least **1024 threads**.
- A multi-core CPU with at least **32 threads**.
- At least **24 GB** of both host memory and GPU memory.
- At least **120 GB** of free disk space for storing generated synthetic data.

## Software Requirements

Please ensure your system meets the following minimum software versions (greater than or equal to these values):

- bash
- wget or curl
- perf
- ncu
- CUDA: 12.6
- GCC: 11.4.0
- CMake: 3.22.1
- Python: 3.10

**Note:** `perf` and `ncu` must be installed and accessible in your system's `PATH`. If they are missing, consult your
system administrator or package manager.

---

## Setup & Execution

To run BambooSMP, simply execute the following from the project root directory:

```bash
./runme.sh
```

The script will automatically handle dependencies downloads (requires wget or curl), installation, compilation, and
execution of workloads. The resulting `figure3`, `figure5`, `table1`, `figure7`, `figure8`, and `figure9` will be stored
in the `data/figures` directory. The entire experiment may take approximately 4–6 hours to complete, depending on the hardware configuration.
