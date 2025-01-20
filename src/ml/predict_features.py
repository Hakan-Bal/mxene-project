import torch
import numpy as np
import pandas as pd
import joblib
import os
import json
from train_feature_extractor import Autoencoder

class MXenePredictor:
    def __init__(self, base_dir):
        """Initialize predictor with paths"""
        self.data_dir = os.path.join(base_dir, 'data', 'processed')
        self.model_dir = os.path.join(base_dir, 'models', 'feature_extractor')
        
        # Load feature information
        with open(os.path.join(self.data_dir, 'feature_info.json'), 'r') as f:
            self.feature_info = json.load(f)
        
        # Load model
        self.load_model()
        
        # Load scalers and mappings
        self.load_preprocessing()
        
        # Load original data
        self.load_original_data()
    
    def load_model(self):
        """Load trained autoencoder"""
        # Get input dimension
        input_dim = (self.feature_info['composition_size'] + 
                    self.feature_info['element_prop_size'] + 
                    self.feature_info['property_size'])
        
        # Initialize model architecture
        self.model = Autoencoder(input_dim, 16)
        
        # Load trained weights
        model_path = os.path.join(self.model_dir, 'feature_extractor_best.pth')
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
    
    def load_preprocessing(self):
        """Load preprocessing objects"""
        # Load element mapping
        self.elements = pd.read_csv(
            os.path.join(self.data_dir, 'element_mapping.csv'))['0'].tolist()
        
        # Load scalers
        self.prop_scaler = joblib.load(
            os.path.join(self.data_dir, 'property_scaler.pkl'))
        
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
    
    def load_original_data(self):
        """Load original dataset"""
        self.main_df = pd.read_csv(os.path.join(self.data_dir, '../raw/dataset.csv'))
        self.property_cols = ['E_coh', 'WF', 'W_sur', 'E_coh_norm', 'density_sur', 'l_M-X', 'l_M2-X']
    
    def get_element_features(self, elements):
        """Get element properties as features"""
        features = []
        for prop in self.element_properties.values():
            values = [prop.get(elem, 0) for elem in elements]
            features.extend([np.mean(values), np.std(values)])
        return features
    
    def predict_mxene(self, mxene_name):
        """Predict features for a given MXene"""
        mxene_data = self.main_df[self.main_df['Mxenes'] == mxene_name]
        
        if len(mxene_data) == 0:
            raise ValueError(f"MXene {mxene_name} not found in dataset")
        
        # Create composition features
        comp_feat = np.zeros(len(self.elements))
        comp_elements = [elem for elem in mxene_name.split('-')[0] if elem.isalpha()]
        for i, elem in enumerate(self.elements):
            if elem in comp_elements:
                comp_feat[i] = 1
        
        # Create element property features
        elem_prop_feat = self.get_element_features(comp_elements)
        
        # Get and scale properties
        properties = mxene_data[self.property_cols].values
        prop_feat = self.prop_scaler.transform(properties)
        
        # Combine features
        features = np.concatenate([comp_feat, elem_prop_feat, prop_feat.flatten()])
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Get predictions
        with torch.no_grad():
            reconstructed, latent = self.model(features_tensor)
        
        # Convert reconstructed features back
        reconstructed = reconstructed.numpy()[0]
        comp_reconstructed = reconstructed[:len(self.elements)]
        elem_prop_reconstructed = reconstructed[len(self.elements):len(self.elements)+len(elem_prop_feat)]
        prop_reconstructed = self.prop_scaler.inverse_transform(
            reconstructed[len(self.elements)+len(elem_prop_feat):].reshape(1, -1))[0]
        
        # Original properties
        original_props = properties[0]
        
        return {
            'composition': {
                'original_elements': comp_elements,
                'predicted_elements': [elem for i, elem in enumerate(self.elements) 
                                    if comp_reconstructed[i] > 0.5]
            },
            'properties': {
                'original': {name: value for name, value in zip(self.property_cols, original_props)},
                'predicted': {name: value for name, value in zip(self.property_cols, prop_reconstructed)}
            },
            'latent_features': latent.numpy()[0],
            'reconstruction_error': np.mean((features - reconstructed)**2)
        }

def print_prediction_results(mxene_name, results):
    """Print prediction results in a formatted way"""
    print(f"\nPredictions for {mxene_name}:")
    print("\n1. Composition Analysis:")
    print(f"Original elements: {', '.join(results['composition']['original_elements'])}")
    print(f"Predicted elements: {', '.join(results['composition']['predicted_elements'])}")
    
    print("\n2. Property Predictions:")
    print("Original vs Predicted values:")
    for prop in results['properties']['original'].keys():
        orig = results['properties']['original'][prop]
        pred = results['properties']['predicted'][prop]
        error = abs(orig - pred) / abs(orig) * 100  # Percentage error
        print(f"{prop:>10}: {orig:>10.4f} vs {pred:>10.4f} (Error: {error:.2f}%)")
    
    print(f"\n3. Reconstruction Error: {results['reconstruction_error']:.6f}")
    
    print("\n4. Latent Space Representation:")
    print("16-dimensional feature vector:")
    latent = results['latent_features']
    for i in range(0, 16, 4):
        print(f"Dims {i:2d}-{i+3:2d}: " + 
              " ".join(f"{x:>8.4f}" for x in latent[i:i+4]))

if __name__ == '__main__':
    # Example usage
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    predictor = MXenePredictor(base_dir)
    
    # Test with different MXenes
    test_mxenes = ["CrCrBF2-1", "MnTiNS2-1", "VCrCS2-1"]
    
    for mxene in test_mxenes:
        try:
            results = predictor.predict_mxene(mxene)
            print_prediction_results(mxene, results)
        except ValueError as e:
            print(f"\nError with {mxene}: {e}")
