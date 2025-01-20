#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read, write
from ase.visualize import view
from ase.io.trajectory import Trajectory
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec

class DFTVisualizer:
    def __init__(self, results_dir):
        """Initialize visualizer with path to DFT results"""
        self.results_dir = results_dir
        self.output_dir = os.path.join(results_dir, 'visualizations')
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
    def view_structure(self, formula, termination):
        """View optimized structure using ASE GUI"""
        traj_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.traj')
        
        try:
            # Read the last frame from trajectory
            traj = Trajectory(traj_file)
            atoms = traj[-1]
            
            # Add name to atoms object
            atoms.info['name'] = f"{formula}-{termination} Optimized Structure"
            
            # View structure in ASE GUI
            view(atoms, viewer='ase')
            
            # Print info for user
            print(f"Viewing: {formula}-{termination} Optimized Structure")
            
        except Exception as e:
            print(f"Error viewing structure for {formula}_{termination}: {str(e)}")

    def view_optimization(self, formula, termination):
        """View optimization trajectory in ASE GUI"""
        traj_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.traj')
        
        try:
            # Read trajectory
            traj = Trajectory(traj_file)
            
            # Add name to each frame
            for atoms in traj:
                atoms.info['name'] = f"{formula}-{termination} Optimization"
            
            # View trajectory in ASE GUI
            view(traj, viewer='ase')
            
            # Print info for user
            print(f"Viewing: {formula}-{termination} Optimization Trajectory")
            
        except Exception as e:
            print(f"Error viewing trajectory for {formula}_{termination}: {str(e)}")

    def save_structure_image(self, formula, termination, rotation='45x,45y,0z'):
        """Save structure image with given rotation"""
        from ase.io import write
        traj_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.traj')
        
        try:
            # Read the last frame from trajectory
            traj = Trajectory(traj_file)
            atoms = traj[-1]
            
            # Save image
            png_file = os.path.join(self.output_dir, f'{formula}_{termination}_structure.png')
            write(png_file, atoms, rotation=rotation)
            print(f"Saved structure image to: {png_file}")
        except Exception as e:
            print(f"Error saving structure image for {formula}_{termination}: {str(e)}")
        
    def create_optimization_animation(self, formula, termination, interval=200):
        """Create animation of optimization trajectory"""
        traj_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.traj')
        traj = Trajectory(traj_file)
        
        # Setup figure
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        def update(frame):
            ax.clear()
            atoms = traj[frame]
            positions = atoms.get_positions()
            chemical_symbols = atoms.get_chemical_symbols()
            
            # Plot atoms
            for i, (pos, symbol) in enumerate(zip(positions, chemical_symbols)):
                ax.scatter(*pos, label=symbol if i == 0 else "", s=100)
            
            # Plot cell
            cell = atoms.get_cell()
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        origin = np.array([i, j, k]) @ cell
                        for axis in range(3):
                            direction = np.zeros(3)
                            direction[axis] = 1
                            end = origin + direction @ cell
                            ax.plot3D(*zip(origin, end), 'k-', alpha=0.3)
            
            ax.set_title(f'Step {frame}')
            ax.legend()
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
        
        # Create animation
        anim = animation.FuncAnimation(fig, update, frames=len(traj), 
                                     interval=interval, blit=False)
        
        # Save animation
        mp4_file = os.path.join(self.output_dir, f'{formula}_{termination}_optimization.mp4')
        anim.save(mp4_file, writer='ffmpeg')
        plt.close()
        print(f"Saved optimization animation to: {mp4_file}")
        
    def plot_optimization_progress(self, formula, termination):
        """Plot energy and force evolution during optimization"""
        log_file = os.path.join(self.results_dir, f'{formula}_{termination}_opt.log')
        
        # Read log file
        steps = []
        times = []
        energies = []
        forces = []
        
        with open(log_file, 'r') as f:
            # Skip header
            for line in f:
                if 'Step' in line and 'Time' in line:
                    break
            
            # Read data
            for line in f:
                if not line.startswith('BFGS:'):
                    continue
                    
                parts = line.split()
                if len(parts) < 4:
                    continue
                    
                try:
                    step = int(parts[1])
                    time = parts[2]
                    energy = float(parts[3])
                    fmax = float(parts[4]) if len(parts) > 4 else None
                    
                    steps.append(step)
                    times.append(time)
                    energies.append(energy)
                    if fmax is not None:
                        forces.append(fmax)
                except:
                    continue
        
        if not steps:
            print(f"No data found in log file: {log_file}")
            return
        
        # Create figure
        fig = plt.figure(figsize=(12, 5))
        gs = GridSpec(1, 2, figure=fig)
        
        # Energy plot
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(steps, energies, 'b-', label='Energy')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Energy (eV)')
        ax1.set_title('Energy Evolution')
        ax1.grid(True)
        
        # Force plot
        if forces:
            ax2 = fig.add_subplot(gs[1])
            ax2.plot(steps[:len(forces)], forces, 'r-', label='Max Force')
            ax2.axhline(y=0.04, color='k', linestyle='--', label='Force Tolerance')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Force (eV/Å)')
            ax2.set_title('Force Evolution')
            ax2.grid(True)
            ax2.legend()
        
        plt.tight_layout()
        
        # Save plot
        png_file = os.path.join(self.output_dir, f'{formula}_{termination}_optimization.png')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved optimization progress plot to: {png_file}")
        
    def compare_terminations(self, formula):
        """Compare different terminations for the same formula"""
        terminations = ['O', 'F', 'OH']
        
        # Read final energies and forces
        results = {}
        for term in terminations:
            log_file = os.path.join(self.results_dir, f'{formula}_{term}_opt.log')
            
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                    # Get the last complete line with energy and force
                    energies = []
                    forces = []
                    for line in lines:
                        if line.startswith('BFGS:'):
                            parts = line.split()
                            if len(parts) >= 4:
                                try:
                                    energy = float(parts[3])
                                    force = float(parts[4]) if len(parts) > 4 else None
                                    energies.append(energy)
                                    if force is not None:
                                        forces.append(force)
                                except:
                                    continue
                    
                    if energies:
                        results[term] = {
                            'energy': energies[-1],
                            'force': forces[-1] if forces else None
                        }
                    
            except Exception as e:
                print(f"Error reading log file for {formula}_{term}: {str(e)}")
                continue
        
        if not results:
            print(f"No data found for {formula}")
            return
        
        # Create comparison plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Energy comparison
        terms = list(results.keys())
        energies = [results[term]['energy'] for term in terms]
        ax1.bar(terms, energies)
        ax1.set_title('Final Energy')
        ax1.set_ylabel('Energy (eV)')
        
        # Force comparison
        forces = [results[term]['force'] for term in terms if results[term]['force'] is not None]
        if forces:
            terms_with_forces = [term for term in terms if results[term]['force'] is not None]
            ax2.bar(terms_with_forces, forces)
            ax2.axhline(y=0.04, color='k', linestyle='--', label='Force Tolerance')
            ax2.set_title('Final Force')
            ax2.set_ylabel('Force (eV/Å)')
            ax2.legend()
        
        plt.suptitle(f'Comparison of Terminations for {formula}')
        plt.tight_layout()
        
        # Save plot
        png_file = os.path.join(self.output_dir, f'{formula}_termination_comparison.png')
        plt.savefig(png_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved termination comparison plot to: {png_file}")

def main():
    """Main function to demonstrate visualization capabilities"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'dft', 'stability')
    
    # Initialize visualizer
    visualizer = DFTVisualizer(results_dir)
    
    # List of MXene formulas
    formulas = ['Ti2CTx', 'Ti2NTx', 'Mn2CTx']
    terminations = ['O', 'F', 'OH']
    
    # Compare all terminations first
    for formula in formulas:
        print(f"\nProcessing {formula}...")
        visualizer.compare_terminations(formula)
        
        for termination in terminations:
            print(f"\nProcessing {formula} with {termination} termination...")
            visualizer.plot_optimization_progress(formula, termination)

def view_specific_structure(formula, termination):
    """View a specific structure and its optimization trajectory"""
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(base_dir, 'results', 'dft', 'stability')
    
    # Initialize visualizer
    visualizer = DFTVisualizer(results_dir)
    
    print(f"\nViewing {formula} with {termination} termination...")
    visualizer.view_structure(formula, termination)
    visualizer.view_optimization(formula, termination)

if __name__ == "__main__":
    # Run the main visualization pipeline
    main()
    
    # Example usage of view_specific_structure:
    # view_specific_structure('Ti2CTx', 'O')
