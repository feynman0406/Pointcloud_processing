# Pointcloud_processing
Voxelization, feature extraction, draw colorful mesh and write it to ply and las file.
# Point Cloud Processing with Open3D and CUDA

This repository contains code for point cloud processing using Open3D, CuPy, and CUDA. Follow the steps below to set up your environment on Windows with WSL and necessary dependencies.

## Prerequisites

- Windows 10 or later
- Windows Subsystem for Linux (WSL)

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

3. Install CUDA 11.0:
    ```bash
    sudo apt install nvidia-cuda-toolkit
    ```

### Step 3: Install Python Libraries

1. Install `Open3D`, `CuPy`, `NumPy`, and `PyTorch`:
    ```bash
    pip3 install open3d cupy-cuda110 numpy torch
    ```

### Step 4: Import Required Libraries in Your Code

Your Python script should include the following imports:
```python
import open3d as o3d
import open3d.core as o3c
import numpy as np
import cupy as cp
import time
import torch
