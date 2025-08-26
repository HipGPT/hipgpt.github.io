---
layout: base
title: Getting Started
permalink: /getting-started/
---

# Getting Started

This guide will walk you through setting up your environment, cloning the project repository, and building the executables required for training and generation.

## 1\. Prerequisites

Before you begin, ensure your system has the following software installed:

  * An **AMD GPU** compatible with the ROCm toolkit.
  * The **ROCm Toolkit** (version 5.0 or newer) installed.
  * **CMake** (version 3.21 or newer).
  * A C++ compiler, such as `g++` or `clang++`.
  * `git` and `wget` for downloading the repository and the dataset.

## 2\. Clone the Repository

To get a local copy of the project, use `git` to clone the repository:

```bash
git git@github.com:aarnetalman/HipGPT.git
cd HipGPT
```

This will create a `HipGPT` directory containing all the project files.

## 3\. Download the Dataset

The project includes a convenient script to download the Tiny Shakespeare dataset, which is an excellent starting point for training a language model.

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

This command will create a `data/` directory and place a `data.txt` file inside it. If the file already exists, the script will skip the download.

## 4\. Build the Project

The project uses CMake to handle the entire build process, including fetching dependencies like the nlohmann/json library.

First, create a separate `build` directory to keep the main project folder clean.

```bash
mkdir build
cd build
```

Next, run CMake to configure the project. This step finds the necessary tools and prepares the build files. It's crucial to specify the HIP C++ compiler (`hipcc`) to ensure the GPU kernels are built correctly.

```bash
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
```

Finally, compile the project using `make` to build the executables.

```bash
make
```

After the build completes, you will find the `train_gpt` and `generate` executables inside the `build` directory. If you encounter an error, ensure that `build/train_gpt` is present before proceeding.