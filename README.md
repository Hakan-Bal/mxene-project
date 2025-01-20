# MXene Materials Design with ML and DFT

This project combines machine learning and density functional theory (DFT) calculations to design and analyze MXene materials. It includes tools for structure generation, DFT-based stability analysis, and comprehensive visualization of results.

## Features

- **Structure Generation**: Create MXene structures with different compositions and terminations (O, F, OH)
- **DFT Calculations**: Perform stability tests using SIESTA through ASE interface
- **Machine Learning**: Train models to predict material properties
- **Visualization**: Comprehensive tools for viewing structures and analyzing results

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/mxene-project.git
cd mxene-project
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install SIESTA (required for DFT calculations):
   - Follow installation instructions at [SIESTA website](https://siesta-project.org/download/)
   - Make sure SIESTA executable is in your PATH

## Project Structure

See [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) for a detailed overview of the project organization.

## Usage

### 1. Generate MXene Structure

```python
from src.dft.stability.mxene_generator import MXeneStructureGenerator

# Initialize generator
generator = MXeneStructureGenerator()

# Create structure with oxygen termination
structure = generator.create_structure("Ti2CTx", "O")
```

### 2. Run DFT Optimization

```python
from src.dft.stability.ase_siesta_opt import optimize_structure

# Optimize structure
optimize_structure(structure, "Ti2CTx_O")
```

### 3. Visualize Results

```python
from src.visualization.dft_visualizer import DFTVisualizer

# Initialize visualizer
visualizer = DFTVisualizer()

# View structure and optimization trajectory
visualizer.view_structure("Ti2CTx", "O")
visualizer.view_optimization("Ti2CTx", "O")

# Plot optimization progress
visualizer.plot_optimization_progress("Ti2CTx", "O")
```

## Results

The project generates several types of outputs:

1. **Structure Files**: Optimized atomic structures in `.traj` format
2. **Energy Logs**: Energy, force, and stress evolution during optimization
3. **Visualizations**:
   - Structure images
   - Optimization animations
   - Energy and force plots
   - Termination comparison plots

All results are saved in the `results/dft/stability/` directory.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

Your Name - your.email@example.com

Project Link: https://github.com/yourusername/mxene-project
