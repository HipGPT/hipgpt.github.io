---
layout: base
title: Getting Started
permalink: /getting-started/
---

# Getting Started

Follow these steps to download the dataset, build the executables, and start using the model.

## 1\. Prerequisites

  * An **AMD GPU** compatible with the ROCm toolkit.
  * The **ROCm Toolkit** (version 5.0 or newer) installed.
  * **CMake** (version 3.21 or newer).
  * A C++ compiler (like `g++` or `clang++`).
  * `git` and `wget` for downloading dependencies and data.

-----

## 2\. Download the Dataset

The project includes a convenient script to download the Tiny Shakespeare dataset, which is a great starting point for training.

```bash
chmod +x scripts/download_data.sh
./scripts/download_data.sh
```

This command will create a `data/` directory and place a `data.txt` file inside it.

-----

## 3\. Build the Project

The project uses CMake to handle the entire build process, including fetching dependencies.

Create a build directory: It's best practice to build the project in a separate directory to keep the main folder clean.

```bash
mkdir build
cd build
```

Run CMake to configure the project: This step finds the necessary tools and prepares the build files.

```bash
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/hipcc
```

Compile the project: This command builds the executables.

```bash
make
```

After the build completes, you will find the `train_gpt` and `generate` executables inside the build directory.