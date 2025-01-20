#!/usr/bin/env python3
import os
import numpy as np
from ase import Atoms
from ase.io import write
from ase.calculators.siesta import Siesta
from ase.optimize import BFGS
from ase.units import Ry
from ase.visualize import view

class MXeneOptimizer:
    def __init__(self, base_dir):
        """Initialize optimizer with paths"""
        self.base_dir = base_dir
        self.pseudo_dir = os.path.join(base_dir, 'src', 'dft', 'stability', 'pseudos')
        self.results_dir = os.path.join(base_dir, 'results', 'dft', 'stability')
        
        # Create results directory if it doesn't exist
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)

    def create_mxene_structure(self, formula, termination='O'):
        """Create MXene structure based on chemical formula"""
        # Parse formula (example: Ti2CTx, Mn2NTx)
        if formula.startswith('Ti2'):
            metal = 'Ti'
            x_element = formula[3]  # C or N
        elif formula.startswith('Mn2'):
            metal = 'Mn'
            x_element = formula[3]  # C or N
        else:
            raise ValueError(f"Unsupported metal in formula: {formula}")
        
        # Define elements list based on termination
        if termination == 'O':
            elements = [metal]*2 + [x_element] + ['O']*2
        elif termination == 'F':
            elements = [metal]*2 + [x_element] + ['F']*2
        elif termination == 'OH':
            elements = [metal]*2 + [x_element] + ['O']*2 + ['H']*2
        else:
            raise ValueError(f"Unsupported termination: {termination}")
        
        # Define lattice parameters
        a = 3.2  # Å (in-plane)
        c = 25.0  # Å (including vacuum)
        
        # Create hexagonal cell
        cell = np.array([[a, 0, 0],
                        [-a/2, a*np.sqrt(3)/2, 0],
                        [0, 0, c]])
        
        # Define atomic positions (fractional coordinates)
        positions = [
            [1/3, 2/3, 0.4],   # M1
            [2/3, 1/3, 0.45],  # M2
            [0.0, 0.0, 0.5],   # X
        ]
        
        # Add termination atoms
        if termination in ['O', 'F']:
            positions.extend([
                [1/3, 2/3, 0.55],  # T1
                [2/3, 1/3, 0.35],  # T2
            ])
        elif termination == 'OH':
            positions.extend([
                [1/3, 2/3, 0.55],  # O1
                [2/3, 1/3, 0.35],  # O2
                [1/3, 2/3, 0.60],  # H1
                [2/3, 1/3, 0.30],  # H2
            ])
        
        # Create ASE Atoms object
        atoms = Atoms(elements,
                     scaled_positions=positions,
                     cell=cell,
                     pbc=[True, True, True])
        
        return atoms

    def setup_siesta_calculator(self, formula):
        """Setup SIESTA calculator with appropriate parameters"""
        # Define SIESTA parameters
        return Siesta(
            label=os.path.join(self.results_dir, formula),
            mesh_cutoff=150 * Ry,
            energy_shift=0.01,
            basis_set='DZP',
            xc='GGA',
            kpts=[6, 6, 1],  # Dense k-point grid in plane, single k-point in z
            spin='polarized',
            pseudo_path=self.pseudo_dir,
            pseudo_qualifier='.psml',
            atomic_coord_format='xyz',
            fdf_arguments={
                'XC.functional': 'GGA',
                'MaxSCFIterations': 200,
                'DM.Tolerance': '1.0E-4',
                'SolutionMethod': 'diagon',
                'ElectronicTemperature': '300 K',
                'DM.MixingWeight': 0.1,
                'DM.NumberPulay': 5,
                'DM.UseSaveDM': True,
                'DM.InitialSpinPol': 0.2,
                'XML.Write': True,
                'WriteEigenvalues': True,
                'WriteDenchar': True,
                'SaveRho': True,
                'SaveDeltaRho': True,
                'SaveElectrostaticPotential': True,
                'PAO.BasisSize': 'DZP',
                'PAO.EnergyShift': str(0.01) + ' Ry',
                'PAO.SplitNorm': 0.15,
                'PAO.SoftDefault': True,
                'SCF.DM.Tolerance': '1.0E-4',
                'SCF.EDM.Tolerance': '1.0E-4 eV',
                'SCF.H.Tolerance': '1.0E-4 eV',
                'MD.TypeOfRun': 'CG',
                'MD.NumCGsteps': 100,
                'MD.MaxForceTol': '0.04 eV/Ang',
                'MD.MaxStressTol': '1.0 GPa',
                'MD.VariableCell': True,
                'MD.ConstantVolume': False
            }
        )

    def run_optimization(self, formula, termination='O'):
        """Run SIESTA optimization for given MXene formula"""
        print(f"\nStarting optimization for {formula} with {termination} termination")
        
        try:
            # Create structure
            atoms = self.create_mxene_structure(formula, termination)
            
            # Setup calculator
            calc = self.setup_siesta_calculator(formula)
            atoms.calc = calc
            
            # Setup optimizer with trajectory and logfile
            traj_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.traj')
            log_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.log')
            
            # Create a custom logfile
            with open(log_file, 'w') as f:
                f.write(f"Optimization log for {formula} with {termination} termination\n")
                f.write("Step  Energy(eV)  Max-Force(eV/A)  Max-Stress(GPa)\n")
                f.write("-" * 50 + "\n")
            
            # Setup optimizer
            optimizer = BFGS(atoms, trajectory=traj_file, logfile=None)
            
            # Custom optimization loop
            fmax_history = []
            energy_history = []
            stress_history = []
            
            def log_step():
                energy = atoms.get_potential_energy()
                forces = atoms.get_forces()
                stress = atoms.get_stress()
                max_force = np.max(np.abs(forces))
                max_stress = np.max(np.abs(stress))
                
                with open(log_file, 'a') as f:
                    f.write(f"{len(fmax_history):3d}  {energy:10.3f}  {max_force:10.3f}  {max_stress:10.3f}\n")
                
                fmax_history.append(max_force)
                energy_history.append(energy)
                stress_history.append(max_stress)
            
            # Run optimization with custom logging
            print(f"Starting geometry optimization")
            while True:
                forces = atoms.get_forces()
                fmax = np.max(np.abs(forces))
                
                # Log current step
                log_step()
                
                # Check convergence
                if fmax < 0.04:  # eV/Å
                    break
                    
                # Perform one optimization step
                optimizer.step()
                
                # Break if too many steps
                if len(fmax_history) >= 100:
                    print("Maximum steps reached")
                    break
            
            # Get final results
            energy = atoms.get_potential_energy()
            forces = atoms.get_forces()
            stress = atoms.get_stress()
            
            # Save final summary to log
            with open(log_file, 'a') as f:
                f.write("\nFinal Results:\n")
                f.write(f"Total Energy: {energy:.3f} eV\n")
                f.write(f"Maximum Force: {np.max(np.abs(forces)):.3f} eV/Å\n")
                f.write(f"Average Force: {np.mean(np.abs(forces)):.3f} eV/Å\n")
                f.write(f"Maximum Stress: {np.max(np.abs(stress)):.3f} GPa\n")
                f.write(f"Optimization converged: {optimizer.converged()}\n")
            
            # Save optimized structure
            xyz_file = os.path.join(self.results_dir, f'{formula}_{termination}_optimized.xyz')
            write(xyz_file, atoms)
            
            results = {
                'formula': formula,
                'termination': termination,
                'energy': energy,
                'forces': forces,
                'stress': stress,
                'converged': optimizer.converged(),
                'steps': len(fmax_history),
                'energy_history': energy_history,
                'fmax_history': fmax_history,
                'stress_history': stress_history
            }
            
            return results
            
        except Exception as e:
            print(f"Error during optimization: {str(e)}")
            return None

    def print_results(self, results):
        """Print calculation results"""
        if results is None:
            return
        
        print(f"\nResults for {results['formula']} ({results['termination']} termination):")
        print(f"Total Energy: {results['energy']:.3f} eV")
        print(f"Maximum Force: {np.max(np.abs(results['forces'])):.3f} eV/Å")
        print(f"Average Force: {np.mean(np.abs(results['forces'])):.3f} eV/Å")
        print(f"Maximum Stress: {np.max(np.abs(results['stress'])):.3f} GPa")
        print(f"Optimization converged: {results['converged']}")

def main():
    """Main function to run optimizations"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Initialize optimizer
    optimizer = MXeneOptimizer(base_dir)
    
    # List of MXene formulas to optimize
    formulas = ['Ti2CTx', 'Ti2NTx', 'Mn2CTx']
    terminations = ['O', 'F', 'OH']
    
    # Run optimizations
    for formula in formulas:
        for termination in terminations:
            results = optimizer.run_optimization(formula, termination)
            optimizer.print_results(results)

if __name__ == "__main__":
    main()
