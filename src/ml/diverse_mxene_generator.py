import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import json
from itertools import combinations, product
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import time

class DiverseMXeneGenerator:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data', 'processed')
        self.model_dir = os.path.join(base_dir, 'models', 'diverse_generator')
        
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        
        # Element grupları
        self.metal_elements = ['Ti', 'V', 'Cr', 'Mn']
        self.x_elements = ['C', 'N', 'B']
        self.termination_elements = ['O', 'F', 'OH']
        
        # Element özellikleri
        self.element_properties = {
            'Ti': {'radius': 1.47, 'electronegativity': 1.54, 'electron_affinity': 0.079},
            'V':  {'radius': 1.34, 'electronegativity': 1.63, 'electron_affinity': 0.525},
            'Cr': {'radius': 1.28, 'electronegativity': 1.66, 'electron_affinity': 0.666},
            'Mn': {'radius': 1.27, 'electronegativity': 1.55, 'electron_affinity': -1.0},
            'C':  {'radius': 0.67, 'electronegativity': 2.55, 'electron_affinity': 1.263},
            'N':  {'radius': 0.56, 'electronegativity': 3.04, 'electron_affinity': -0.07},
            'B':  {'radius': 0.87, 'electronegativity': 2.04, 'electron_affinity': 0.277},
        }
    
    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic MXene data"""
        synthetic_data = []
        
        # Tüm olası M₂X ve M₃X₂ kombinasyonları
        for m in self.metal_elements:
            for x in self.x_elements:
                # M₂X yapıları
                base_props = self._calculate_base_properties(m, x, ratio='2:1')
                synthetic_data.extend(
                    self._add_variations(base_props, m, x, '2:1', n_samples//8)
                )
                
                # M₃X₂ yapıları
                base_props = self._calculate_base_properties(m, x, ratio='3:2')
                synthetic_data.extend(
                    self._add_variations(base_props, m, x, '3:2', n_samples//8)
                )
        
        return pd.DataFrame(synthetic_data)
    
    def _calculate_base_properties(self, metal, x_element, ratio):
        """Calculate base properties for a given composition"""
        m_props = self.element_properties[metal]
        x_props = self.element_properties[x_element]
        
        if ratio == '2:1':
            m_factor, x_factor = 2, 1
        else:  # '3:2'
            m_factor, x_factor = 3, 2
        
        # Basit özellik tahminleri
        e_coh = -100 * (m_props['electronegativity'] * x_props['electronegativity'])
        wf = 4.5 + m_props['electron_affinity'] + x_props['electron_affinity']
        w_sur = 150 + 50 * (m_props['radius'] + x_props['radius'])
        
        return {
            'metal': metal,
            'x_element': x_element,
            'ratio': ratio,
            'E_coh': e_coh,
            'WF': wf,
            'W_sur': w_sur,
            'E_coh_norm': e_coh / (m_factor + x_factor),
            'density_sur': (m_factor * m_props['radius'] + x_factor * x_props['radius']) / 2,
            'l_M-X': (m_props['radius'] + x_props['radius']) * 0.8,
            'l_M2-X': (m_props['radius'] + x_props['radius']) * 0.9
        }
    
    def _add_variations(self, base_props, metal, x_element, ratio, n_variations):
        """Add variations to base properties"""
        variations = []
        
        for _ in range(n_variations):
            # Rastgele varyasyonlar ekle
            variation = base_props.copy()
            
            # Özelliklere rastgele gürültü ekle
            variation['E_coh'] *= np.random.normal(1, 0.1)
            variation['WF'] *= np.random.normal(1, 0.05)
            variation['W_sur'] *= np.random.normal(1, 0.15)
            variation['E_coh_norm'] *= np.random.normal(1, 0.1)
            variation['density_sur'] *= np.random.normal(1, 0.05)
            variation['l_M-X'] *= np.random.normal(1, 0.03)
            variation['l_M2-X'] *= np.random.normal(1, 0.03)
            
            variations.append(variation)
        
        return variations
    
    def train_diverse_generator(self, n_epochs=100, batch_size=32):
        """Train a generator that promotes diversity"""
        # Generate synthetic data
        data = self.generate_synthetic_data()
        
        # Prepare data for training
        X = data[['E_coh', 'WF', 'W_sur', 'E_coh_norm', 
                 'density_sur', 'l_M-X', 'l_M2-X']].values
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Save scaler
        joblib.dump(scaler, os.path.join(self.model_dir, 'diverse_scaler.pkl'))
        
        # Convert to torch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Create data loader
        dataset = torch.utils.data.TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize models
        latent_dim = 32
        self.encoder = Encoder(input_dim=X.shape[1], latent_dim=latent_dim)
        self.decoder = Decoder(latent_dim=latent_dim, output_dim=X.shape[1])
        
        # Training
        optimizer = optim.Adam(list(self.encoder.parameters()) + 
                             list(self.decoder.parameters()))
        
        for epoch in range(n_epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                # Forward pass
                z = self.encoder(batch_x)
                x_recon = self.decoder(z)
                
                # Reconstruction loss
                recon_loss = nn.MSELoss()(x_recon, batch_x)
                
                # Diversity loss (encourage spread in latent space)
                diversity_loss = -torch.std(z, dim=0).mean()
                
                # Total loss
                loss = recon_loss + 0.1 * diversity_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{n_epochs}], Loss: {total_loss/len(dataloader):.4f}')
        
        # Save models
        torch.save(self.encoder.state_dict(), 
                  os.path.join(self.model_dir, 'diverse_encoder.pth'))
        torch.save(self.decoder.state_dict(), 
                  os.path.join(self.model_dir, 'diverse_decoder.pth'))
    
    def generate_diverse_mxenes(self, n_samples=10, temperature=1.0):
        """Generate diverse MXene designs"""
        # Load scaler
        scaler = joblib.load(os.path.join(self.model_dir, 'diverse_scaler.pkl'))
        
        # Generate samples from latent space
        with torch.no_grad():
            # Sample from normal distribution with temperature
            z = torch.randn(n_samples, self.encoder.latent_dim) * temperature
            
            # Generate properties
            x_generated = self.decoder(z)
            x_generated = x_generated.numpy()
            
            # Calculate reconstruction error
            x_recon = self.decoder(self.encoder(torch.FloatTensor(x_generated)))
            recon_error = torch.nn.MSELoss()(torch.FloatTensor(x_generated), x_recon).item() * 1000
            
            # Inverse transform
            properties = scaler.inverse_transform(x_generated)
        
        # Convert to MXene designs
        designs = []
        for props in properties:
            # Find closest base composition
            best_design = None
            min_diff = float('inf')
            
            for m in self.metal_elements:
                for x in self.x_elements:
                    for ratio in ['2:1', '3:2']:
                        base = self._calculate_base_properties(m, x, ratio)
                        
                        # Calculate difference from generated properties
                        diff = np.mean([
                            abs(base['E_coh'] - props[0]) / abs(props[0]),
                            abs(base['WF'] - props[1]) / abs(props[1]),
                            abs(base['W_sur'] - props[2]) / abs(props[2])
                        ])
                        
                        if diff < min_diff:
                            min_diff = diff
                            best_design = {
                                'composition': {
                                    'metal': m,
                                    'x_element': x,
                                    'ratio': ratio
                                },
                                'properties': {
                                    'E_coh': props[0],
                                    'WF': props[1],
                                    'W_sur': props[2],
                                    'E_coh_norm': props[3],
                                    'density_sur': props[4],
                                    'l_M-X': props[5],
                                    'l_M2-X': props[6]
                                },
                                'reconstruction_error': recon_error
                            }
            
            designs.append(best_design)
        
        return designs
    
    def save_diverse_designs(self, designs, output_dir=None):
        """Save generated diverse designs to CSV and JSON files"""
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, 'results', 'diverse_designs')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate timestamp for filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        designs_for_json = []
        for design in designs:
            design_copy = {
                'composition': {k: v for k, v in design['composition'].items()},
                'properties': {k: float(v) for k, v in design['properties'].items()},
                'reconstruction_error': float(design['reconstruction_error'])
            }
            designs_for_json.append(design_copy)
        
        # Prepare data for CSV
        csv_data = []
        for i, design in enumerate(designs, 1):
            row = {
                'design_id': i,
                'timestamp': timestamp,
                'formula': (f"{design['composition']['metal']}₂{design['composition']['x_element']}Tₓ" 
                          if design['composition']['ratio'] == '2:1'
                          else f"{design['composition']['metal']}₃{design['composition']['x_element']}₂Tₓ"),
                'metal': design['composition']['metal'],
                'x_element': design['composition']['x_element'],
                'ratio': design['composition']['ratio']
            }
            
            # Add properties
            for prop_name, value in design['properties'].items():
                row[prop_name] = float(value)
            
            row['reconstruction_error'] = float(design['reconstruction_error'])
            
            csv_data.append(row)
            
        # Save to CSV
        csv_file = os.path.join(output_dir, f'diverse_mxene_designs_{timestamp}.csv')
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Save to JSON (includes full design details)
        json_file = os.path.join(output_dir, f'diverse_mxene_designs_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'designs': designs_for_json,
                'parameters': {
                    'temperature': float(self.temperature),
                    'model_version': '1.0',
                    'training_epochs': self.n_epochs
                }
            }, f, indent=2)
            
        print(f"\nResults saved to:")
        print(f"CSV: {csv_file}")
        print(f"JSON: {json_file}")

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, latent_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.1),
            
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.1),
            
            nn.Linear(128, output_dim)
        )
    
    def forward(self, z):
        return self.decoder(z)

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Initialize generator
    generator = DiverseMXeneGenerator(base_dir)
    generator.n_epochs = 100
    generator.temperature = 1.2
    
    # Train generator
    print("Training diverse generator...")
    generator.train_diverse_generator(n_epochs=generator.n_epochs)
    
    # Generate diverse designs
    print("\nGenerating diverse MXene designs...")
    designs = generator.generate_diverse_mxenes(n_samples=10, temperature=generator.temperature)
    
    # Save results
    generator.save_diverse_designs(designs)
    
    # Print results
    print("\nGenerated MXene Designs:")
    print("=" * 50)
    
    for i, design in enumerate(designs, 1):
        print(f"\nDesign {i}:")
        print("-" * 30)
        comp = design['composition']
        props = design['properties']
        
        if comp['ratio'] == '2:1':
            formula = f"{comp['metal']}₂{comp['x_element']}Tₓ"
        else:
            formula = f"{comp['metal']}₃{comp['x_element']}₂Tₓ"
        
        print(f"Formula: {formula}")
        print(f"Composition: {comp['metal']}-{comp['x_element']} ({comp['ratio']})")
        print("\nPredicted Properties:")
        print(f"  E_coh: {props['E_coh']:.2f} eV")
        print(f"  WF: {props['WF']:.2f} eV")
        print(f"  W_sur: {props['W_sur']:.2f} meV/Å²")
        print(f"  E_coh_norm: {props['E_coh_norm']:.2f}")
        print(f"  density_sur: {props['density_sur']:.2f}")
        print(f"  l_M-X: {props['l_M-X']:.3f} Å")
        print(f"  l_M2-X: {props['l_M2-X']:.3f} Å")
        print(f"\nReconstruction Error: {design['reconstruction_error']:.3f}")
