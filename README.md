# Quantum Convolution for Structure-Based Virtual Screening

This repository demonstrates a quantum convolutional approach for structure-based virtual screening using protein-ligand complexes. It utilizes Pennylane and PyTorch for quantum circuit simulation and model training.

## Installation

Before running the code, please install the required Python packages:

`pip install torch numpy pennylane`

## Folder Descriptions

- **1_database/**
  - Converts PDBbind_v2020 data into 2⁹ and 2¹² pixel quantum image formats.
  - Input files include `ligand.sdf` and `protein.pdb`.
  - All data files are gzip-compressed (`.gz`).
  - Only one PDB ID is provided here as an example due to storage constraints.

- **2_train/**
  - Scripts for training the quantum convolutional model.

- **3_check_par/**
  - Loads trained parameters and verifies model output using Pennylane.
  - Compares results against the PyTorch-based output.

- **4_noise/**
  - Applies noise in Pennylane to test trained model robustness under quantum noise conditions.

- **LICENSE**
  - MIT License.
