import torch
import numpy as np
import pandas as pd
import joblib
import os
import json
from train_feature_extractor import Autoencoder
from train_inverse_design import PropertyConditionedGenerator
import time

class MXeneDesigner:
    def __init__(self, base_dir):
        """Initialize designer with paths"""
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, 'data', 'processed')
        self.feature_extractor_dir = os.path.join(base_dir, 'models', 'feature_extractor')
        self.inverse_design_dir = os.path.join(base_dir, 'models', 'inverse_design')
        
        # Load models and preprocessing
        self.load_models()
        self.load_preprocessing()
        
        # Load element properties
        self.element_properties = {
            'atomic_radius': {
                'Cr': 1.28, 'Mn': 1.27, 'V': 1.34, 'Ti': 1.47,
                'B': 0.87, 'C': 0.67, 'N': 0.56, 'S': 1.02, 'F': 0.42
            },
            'electronegativity': {
                'Cr': 1.66, 'Mn': 1.55, 'V': 1.63, 'Ti': 1.54,
                'B': 2.04, 'C': 2.55, 'N': 3.04, 'S': 2.58, 'F': 3.98
            },
            'electron_affinity': {
                'Cr': 0.666, 'Mn': -1.0, 'V': 0.525, 'Ti': 0.079,
                'B': 0.277, 'C': 1.263, 'N': -0.07, 'S': 2.077, 'F': 3.399
            }
        }
    
    def load_models(self):
        """Load feature extractor and inverse design models"""
        # Load feature information
        with open(os.path.join(self.data_dir, 'feature_info.json'), 'r') as f:
            self.feature_info = json.load(f)
        
        # Initialize and load feature extractor
        input_dim = (self.feature_info['composition_size'] + 
                    self.feature_info['element_prop_size'] + 
                    self.feature_info['property_size'])
        
        self.feature_extractor = Autoencoder(input_dim, 16)
        self.feature_extractor.load_state_dict(
            torch.load(os.path.join(self.feature_extractor_dir, 
                                  'feature_extractor_best.pth'))
        )
        self.feature_extractor.eval()
        
        # Initialize and load inverse design model
        self.inverse_model = PropertyConditionedGenerator(
            property_dim=self.feature_info['property_size']
        )
        self.inverse_model.load_state_dict(
            torch.load(os.path.join(self.inverse_design_dir, 
                                  'inverse_design_best.pth'))
        )
        self.inverse_model.eval()
    
    def load_preprocessing(self):
        """Load preprocessing objects"""
        # Define elements
        self.elements = ['Ti', 'V', 'Cr', 'Mn', 'B', 'C', 'N', 'S', 'F']
        
        # Load scalers
        self.prop_scaler = joblib.load(
            os.path.join(self.data_dir, 'property_scaler.pkl'))
        self.inverse_prop_scaler = joblib.load(
            os.path.join(self.inverse_design_dir, 'inverse_property_scaler.pkl'))
    
    def get_element_features(self, elements):
        """Get element properties as features"""
        features = []
        for prop in self.element_properties.values():
            values = [prop.get(elem, 0) for elem in elements]
            features.extend([np.mean(values), np.std(values)])
        return features
    
    def design_mxene(self, target_properties, num_samples=5, noise_scale=0.3):
        """Design MXene with target properties
        
        Args:
            target_properties: dict with keys ['E_coh', 'WF', 'W_sur', 'E_coh_norm', 
                                             'density_sur', 'l_M-X', 'l_M2-X']
            num_samples: number of designs to generate
            noise_scale: scale of random noise to add for diversity
        
        Returns:
            List of dictionaries containing predicted compositions and properties
        """
        # Convert target properties to array and scale
        target_props = np.array([[
            target_properties['E_coh'],
            target_properties['WF'],
            target_properties['W_sur'],
            target_properties['E_coh_norm'],
            target_properties['density_sur'],
            target_properties['l_M-X'],
            target_properties['l_M2-X']
        ]])
        
        scaled_props = self.inverse_prop_scaler.transform(target_props)
        
        # Generate multiple designs with different noise patterns
        designs = []
        with torch.no_grad():
            for i in range(num_samples):
                # Add different noise scales for different properties
                property_noise = np.random.normal(0, noise_scale, scaled_props.shape)
                property_noise[0, 0] *= 1.5  # E_coh
                property_noise[0, 1] *= 0.8  # WF
                property_noise[0, 2] *= 1.2  # W_sur
                
                noisy_props = scaled_props + property_noise
                props_tensor = torch.FloatTensor(noisy_props)
                
                # Generate latent representation with noise
                _, latent = self.inverse_model(props_tensor)
                
                # Add structured noise to latent space
                latent_noise = torch.randn_like(latent)
                latent_noise *= noise_scale * (1 + 0.2 * i)  # Increase noise with each sample
                latent = latent + latent_noise
                
                # Decode latent representation
                reconstructed = self.feature_extractor.decoder(latent)
                reconstructed = reconstructed.numpy()[0]
                
                # Add noise to property predictions
                property_start = (self.feature_info['composition_size'] + 
                                self.feature_info['element_prop_size'])
                property_end = len(reconstructed)
                
                # Scale noise differently for each property
                property_noise = np.random.normal(0, noise_scale * 0.5, 
                                               property_end - property_start)
                property_noise[0] *= 1.2  # E_coh
                property_noise[1] *= 0.8  # WF
                property_noise[2] *= 1.0  # W_sur
                property_noise[3] *= 1.1  # E_coh_norm
                property_noise[4] *= 0.9  # density_sur
                property_noise[5] *= 0.7  # l_M-X
                property_noise[6] *= 0.7  # l_M2-X
                
                reconstructed[property_start:property_end] += property_noise
                
                # Extract composition and properties
                comp_reconstructed = reconstructed[:self.feature_info['composition_size']]
                
                # Apply softmax with temperature
                temperature = 1.0 + 0.2 * i  # Increase temperature with each sample
                comp_logits = comp_reconstructed / temperature
                comp_probs = np.exp(comp_logits) / np.sum(np.exp(comp_logits))
                
                # Get predicted elements with dynamic threshold
                threshold = 0.1 * (1 - 0.1 * i)  # Decrease threshold with each sample
                pred_elements = [elem for i, elem in enumerate(self.elements) 
                               if comp_probs[i] > threshold]
                
                # Ensure at least one metal element
                metal_elements = ['Ti', 'V', 'Cr', 'Mn']
                if not any(elem in pred_elements for elem in metal_elements):
                    metal_probs = [comp_probs[self.elements.index(elem)] for elem in metal_elements]
                    best_metal = metal_elements[np.argmax(metal_probs)]
                    pred_elements.append(best_metal)
                
                # Ensure at least one non-metal element
                nonmetal_elements = ['B', 'C', 'N', 'S', 'F']
                if not any(elem in pred_elements for elem in nonmetal_elements):
                    nonmetal_probs = [comp_probs[self.elements.index(elem)] for elem in nonmetal_elements]
                    best_nonmetal = nonmetal_elements[np.argmax(nonmetal_probs)]
                    pred_elements.append(best_nonmetal)
                
                # Get probabilities for selected elements
                selected_probs = [comp_probs[self.elements.index(elem)] for elem in pred_elements]
                
                # Normalize selected probabilities
                selected_probs = np.array(selected_probs)
                selected_probs = selected_probs / np.sum(selected_probs)
                
                elem_prop_reconstructed = reconstructed[
                    self.feature_info['composition_size']:
                    self.feature_info['composition_size'] + 
                    self.feature_info['element_prop_size']
                ]
                prop_reconstructed = self.prop_scaler.inverse_transform(
                    reconstructed[
                        self.feature_info['composition_size'] + 
                        self.feature_info['element_prop_size']:
                    ].reshape(1, -1)
                )[0]
                
                # Calculate property errors
                prop_errors = np.abs(prop_reconstructed - target_props[0]) / np.abs(target_props[0]) * 100
                
                designs.append({
                    'composition': {
                        'elements': pred_elements,
                        'probabilities': selected_probs.tolist()
                    },
                    'properties': {
                        'predicted': {
                            'E_coh': prop_reconstructed[0],
                            'WF': prop_reconstructed[1],
                            'W_sur': prop_reconstructed[2],
                            'E_coh_norm': prop_reconstructed[3],
                            'density_sur': prop_reconstructed[4],
                            'l_M-X': prop_reconstructed[5],
                            'l_M2-X': prop_reconstructed[6]
                        },
                        'errors': {
                            'E_coh': prop_errors[0],
                            'WF': prop_errors[1],
                            'W_sur': prop_errors[2],
                            'E_coh_norm': prop_errors[3],
                            'density_sur': prop_errors[4],
                            'l_M-X': prop_errors[5],
                            'l_M2-X': prop_errors[6]
                        }
                    },
                    'reconstruction_error': np.mean((reconstructed - 
                        np.concatenate([comp_reconstructed, elem_prop_reconstructed, 
                                      scaled_props.flatten()]))**2)
                })
        
        # Sort designs by average property error
        designs.sort(key=lambda x: np.mean(list(x['properties']['errors'].values())))
        return designs

    def save_designs(self, designs, output_dir=None):
        """Save generated designs to CSV and JSON files"""
        if output_dir is None:
            output_dir = os.path.join(self.base_dir, 'results', 'designs')
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate timestamp for filenames
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # Convert numpy types to Python types for JSON serialization
        designs_for_json = []
        for design in designs:
            design_copy = {
                'composition': {
                    'elements': design['composition']['elements'],
                    'probabilities': [float(p) for p in design['composition']['probabilities']]
                },
                'properties': {
                    'predicted': {k: float(v) for k, v in design['properties']['predicted'].items()},
                    'errors': {k: float(v) for k, v in design['properties']['errors'].items()}
                }
            }
            designs_for_json.append(design_copy)
        
        # Prepare data for CSV
        csv_data = []
        for i, design in enumerate(designs, 1):
            row = {
                'design_id': i,
                'timestamp': timestamp
            }
            
            # Add composition
            for elem, prob in zip(design['composition']['elements'], 
                                design['composition']['probabilities']):
                row[f'composition_{elem}'] = float(prob)
                
            # Add properties
            for prop_name, value in design['properties']['predicted'].items():
                row[prop_name] = float(value)
                
            csv_data.append(row)
            
        # Save to CSV
        csv_file = os.path.join(output_dir, f'mxene_designs_{timestamp}.csv')
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        # Save to JSON (includes full design details)
        json_file = os.path.join(output_dir, f'mxene_designs_{timestamp}.json')
        with open(json_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'designs': designs_for_json,
                'parameters': {
                    'noise_scale': float(0.3),
                    'model_version': '1.0'
                }
            }, f, indent=2)
            
        print(f"\nResults saved to:")
        print(f"CSV: {csv_file}")
        print(f"JSON: {json_file}")

def print_design_results(designs):
    """Print design results in a formatted way"""
    print("\nGenerated MXene Designs:")
    print("=" * 50)
    
    for i, design in enumerate(designs):
        print(f"\nDesign {i+1}:")
        print("-" * 30)
        
        print("Predicted Composition:")
        elements = design['composition']['elements']
        probs = design['composition']['probabilities']
        for elem, prob in zip(elements, probs):
            print(f"  {elem}: {prob:.3f}")
        
        print("\nPredicted Properties:")
        for prop, value in design['properties']['predicted'].items():
            error = design['properties']['errors'][prop]
            print(f"  {prop:>10}: {value:>10.4f} (Error: {error:.2f}%)")
        
        print(f"\nReconstruction Error: {design['reconstruction_error']:.6f}")
        print("-" * 30)

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Example target properties
    target_properties = {
        'E_coh': -120.0,
        'WF': 4.8,
        'W_sur': 200.0,
        'E_coh_norm': -6.0,
        'density_sur': 2.5,
        'l_M-X': 2.1,
        'l_M2-X': 2.0
    }
    
    # Generate designs
    designer = MXeneDesigner(base_dir)
    designs = designer.design_mxene(target_properties, num_samples=5, noise_scale=0.3)
    
    # Save results
    designer.save_designs(designs)
    
    # Print results
    print_design_results(designs)
