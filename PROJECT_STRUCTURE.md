# MXene Project Structure

This document provides a comprehensive overview of the project structure and files.

## Project Overview

This project focuses on MXene materials design and analysis using machine learning and DFT calculations. It includes tools for structure generation, DFT calculations, and visualization of results.

## Directory Structure

```
mxene_project/
├── data/
│   ├── processed/           # Processed data files for ML
│   │   ├── compositions.csv
│   │   ├── composition_features.npy
│   │   ├── deltaG_H.npy
│   │   ├── deltaG_scaler.pkl
│   │   ├── element_mapping.csv
│   │   ├── element_prop_features.npy
│   │   ├── feature_info.json
│   │   ├── latent_features.npy
│   │   ├── property_features.npy
│   │   └── property_scaler.pkl
│   └── pseudos/            # Pseudopotential files for DFT
│       └── *.psml          # PSML format pseudopotentials for each element
├── results/
│   └── dft/
│       └── stability/      # DFT stability test results
│           ├── *.traj     # Trajectory files from optimization
│           ├── *.log      # Log files with energy and forces
│           └── visualizations/  # Generated plots and animations
├── src/
│   ├── dft/
│   │   └── stability/     # DFT calculation scripts
│   │       ├── ase_siesta_opt.py  # Structure optimization with SIESTA
│   │       └── mxene_generator.py  # MXene structure generation
│   ├── ml/
│   │   ├── data_processing.py     # Data preprocessing
│   │   ├── feature_extraction.py  # Feature engineering
│   │   └── train_model.py        # ML model training
│   └── visualization/
│       └── dft_visualizer.py     # Visualization tools for DFT results
└── requirements.txt        # Python package dependencies

```

## Key Components

### 1. Data Processing (`data/`)

#### Processed Data (`data/processed/`)
- `compositions.csv`: List of MXene compositions
- `*_features.npy`: Various feature matrices for ML
- `*_scaler.pkl`: Scikit-learn scalers for normalization
- `element_mapping.csv`: Mapping of elements to indices
- `feature_info.json`: Feature descriptions and metadata

#### Pseudopotentials (`data/pseudos/`)
- Contains PSML format pseudopotentials for each element
- Used by SIESTA for DFT calculations

### 2. Source Code (`src/`)

#### DFT Calculations (`src/dft/`)
- `ase_siesta_opt.py`: Handles structure optimization using ASE and SIESTA
  - Implements optimization workflow
  - Logs energy, forces, and stress
  - Saves trajectories and results

- `mxene_generator.py`: Generates MXene structures
  - Creates base structures with different compositions
  - Handles different terminations (O, F, OH)
  - Sets up initial configurations

#### Machine Learning (`src/ml/`)
- `data_processing.py`: Data preprocessing pipeline
- `feature_extraction.py`: Feature engineering for ML
- `train_model.py`: ML model training and evaluation

#### Visualization (`src/visualization/`)
- `dft_visualizer.py`: Comprehensive visualization tools
  - Structure visualization
  - Optimization trajectory animations
  - Energy and force plots
  - Termination comparison plots

### 3. Results (`results/`)

#### DFT Results (`results/dft/stability/`)
- Trajectory files (*.traj): Contains atomic positions and cell parameters
- Log files (*.log): Energy, force, and stress data
- Visualizations: Generated plots and animations

### 4. Dependencies

Required Python packages (see `requirements.txt`):
- ASE (Atomic Simulation Environment)
- NumPy
- Matplotlib
- Scikit-learn
- PyTorch (for ML models)

## Usage

1. **Structure Generation**:
   ```python
   from src.dft.stability.mxene_generator import MXeneStructureGenerator
   generator = MXeneStructureGenerator()
   structure = generator.create_structure("Ti2CTx", "O")
   ```

2. **DFT Optimization**:
   ```python
   from src.dft.stability.ase_siesta_opt import optimize_structure
   optimize_structure(structure, "Ti2CTx_O")
   ```

3. **Visualization**:
   ```python
   from src.visualization.dft_visualizer import DFTVisualizer
   visualizer = DFTVisualizer()
   visualizer.view_structure("Ti2CTx", "O")
   visualizer.view_optimization("Ti2CTx", "O")
   ```

## File Naming Conventions

- Structure files: `{formula}_{termination}_opt.traj`
- Log files: `{formula}_{termination}_opt.log`
- Visualization files: 
  - `{formula}_{termination}_structure.png`
  - `{formula}_{termination}_optimization.mp4`
  - `{formula}_{termination}_optimization.png`
  - `{formula}_termination_comparison.png`
