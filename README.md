# BambooSMP

BambooSMP is a high-performance parallel processing framework designed to efficiently solve the Stable Marriage
Problem (SMP). Named after the resilient and fast-growing bamboo plant, BambooSMP aims to deliver robust performance in
various challenging scenarios.

# Installation Guide

## Hardware Requirements

Ensure your system meets the following hardware requirement:

- At least 1 Nvidia GPUs

## Software Requirements

Ensure your system has the following software installed:

- g++ version 11.2.0
- cmake version 3.25.2
- nvcc version 12.3.52
- Linux operating system

## Setup

Follow these steps to set up the Bamboo-SMP project on your system:

1. **Clone the Repository**:

   Download the repository using the following command:

   ```bash
   git clone https://github.com/victorliu-sq/bamboo-smp.git
   ```


2. **Build the Project**:

   Navigate to the project directory, create a build directory, and compile the project:

   ```bash
   mkdir build
   cd build
   cmake ..
   ```


3. **Run the Program**:

   Execute the program:

   ```bash
   ./src/bamboosmp
   ```

When prompted, enter the path to the test files. Example test files include:

   ```
   ../data/perfect_case.txt
   ../data/solo_case.txt
   ../data/congested_case.txt
   ../data/random_case.txt
   ```

4. **Run the Test Program**:

Update the values of `WORKLOAD_TYPE` and `WORKLOAD_SIZE` in `test/bamboosmpTest.cu`, rebuild the entire project, and
then run the test program located in the build directory:

   ```bash
   ./bamboosmpTest
   ```
