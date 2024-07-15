# Pointcloud_processing
Voxelization, feature extraction, draw colorful mesh and write it to ply and las file.
# Point Cloud Processing with Open3D and CUDA

This repository contains code for point cloud processing using Open3D, CuPy, and CUDA. Follow the steps below to set up your environment on Windows with WSL and necessary dependencies.

## Prerequisites

- Windows 10 or later
- Windows Subsystem for Linux (WSL)
- Python version should not exceed 3.11

## Installation Guide

### Step 1: Install WSL

1. Open PowerShell as Administrator and run the following command to install WSL:
    ```powershell
    wsl --install
    ```

2. Restart your computer if prompted.

3. Set up your preferred Linux distribution (e.g., Ubuntu) from the Microsoft Store.

### Step 2: Set Up Your Environment

1. Open WSL (Ubuntu) and update the package list:
    ```bash
    sudo apt update
    ```

2. Install required packages:
    ```bash
    sudo apt install build-essential python3 python3-pip
    ```

3. Ensure Python version is not greater than 3.11. You can install a specific version of Python as follows:
    ```bash
    sudo apt install python3.9 python3.9-venv python3.9-dev
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1
    ```

4. Install CUDA 11.0:
    - Follow the official CUDA installation guide: [NVIDIA CUDA Installation Guide for WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)

### Step 3: Install Python Libraries

1. Install `Open3D`, `CuPy`, `NumPy`, and `PyTorch`:
    ```bash
    pip3 install open3d==0.18.0 cupy-cuda110==12.3.0 numpy==1.25.2 torch==1.7.1
    ```

### Step 4: Verify Installation

To verify your installation, run the following script which checks the versions of installed libraries:

```python
import sys
import os
import subprocess

def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode('utf-8')
        return output
    except FileNotFoundError:
        return "CUDA is not installed"

def get_library_version(lib_name):
    try:
        lib = __import__(lib_name)
        return lib.__version__
    except ImportError:
        return f"{lib_name} is not installed"

def main():
    print("System Information:")
    print(f"Operating System: {os.uname().sysname} {os.uname().release}")
    print(f"Python Version: {sys.version}")

    print("\nCUDA Version:")
    print(get_cuda_version())

    print("\nLibrary Versions:")
    libraries = ["open3d", "cupy", "numpy", "torch"]
    for lib in libraries:
        print(f"{lib}: {get_library_version(lib)}")

if __name__ == "__main__":
    main()

