import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
import os
import json

class MXeneDataProcessor:
    def __init__(self, raw_data_dir, processed_data_dir):
        """Initialize data processor with directory paths"""
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        
        # Element özellikleri
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
        
        # Create processed directory if it doesn't exist
        if not os.path.exists(processed_data_dir):
            os.makedirs(processed_data_dir)
    
    def load_data(self):
        """Load raw datasets"""
        # Load main dataset
        self.main_df = pd.read_csv(os.path.join(self.raw_data_dir, 'dataset.csv'))
        
        # Load deltaG_H dataset
        self.dg_df = pd.read_csv(os.path.join(self.raw_data_dir, 'deltaG_H.csv'))
        
        print(f"Loaded {len(self.main_df)} entries from main dataset")
        print(f"Loaded {len(self.dg_df)} entries from deltaG_H dataset")
    
    def get_element_features(self, elements):
        """Get element properties as features"""
        features = []
        for prop in self.element_properties.values():
            # Her element için özelliğin ortalaması ve std'si
            values = [prop.get(elem, 0) for elem in elements]
            features.extend([np.mean(values), np.std(values)])
        return features
    
    def preprocess_composition(self):
        """Process composition data and extract features"""
        # Extract unique elements and their positions
        elements = set()
        for comp in self.main_df['Mxenes']:
            elements.update([elem for elem in comp.split('-')[0] if elem.isalpha()])
        self.elements = sorted(list(elements))
        
        # Create features
        comp_features = []
        element_prop_features = []
        
        for comp in self.main_df['Mxenes']:
            # One-hot encoding
            feat = np.zeros(len(self.elements))
            comp_elements = [elem for elem in comp.split('-')[0] if elem.isalpha()]
            for i, elem in enumerate(self.elements):
                if elem in comp_elements:
                    feat[i] = 1
            comp_features.append(feat)
            
            # Element properties
            elem_features = self.get_element_features(comp_elements)
            element_prop_features.append(elem_features)
        
        self.comp_features = np.array(comp_features)
        self.element_prop_features = np.array(element_prop_features)
        
        print(f"Created composition features with shape: {self.comp_features.shape}")
        print(f"Created element property features with shape: {self.element_prop_features.shape}")
    
    def preprocess_properties(self):
        """Process property data and normalize"""
        # Select relevant properties
        property_cols = [
            'E_coh', 'WF', 'W_sur',  # Ana özellikler
            'E_coh_norm', 'density_sur',  # Normalize edilmiş ve yüzey özellikleri
            'l_M-X', 'l_M2-X'  # Bağ uzunlukları
        ]
        
        # Create property features
        self.prop_scaler = RobustScaler()  # Aykırı değerlere karşı daha dayanıklı
        self.prop_features = self.prop_scaler.fit_transform(self.main_df[property_cols])
        
        print(f"Created property features with shape: {self.prop_features.shape}")
    
    def merge_deltaG(self):
        """Merge deltaG_H data with main dataset"""
        # Merge based on composition
        merged_df = pd.merge(self.main_df, self.dg_df, on='Mxenes', how='inner')
        
        # Scale deltaG_H values
        self.dg_scaler = RobustScaler()
        self.deltaG_H = self.dg_scaler.fit_transform(merged_df[['deltaG_H']])
        
        print(f"Merged dataset contains {len(merged_df)} entries")
        return merged_df
    
    def save_processed_data(self):
        """Save processed data"""
        # Save features
        np.save(os.path.join(self.processed_data_dir, 'composition_features.npy'), 
                self.comp_features)
        np.save(os.path.join(self.processed_data_dir, 'element_prop_features.npy'), 
                self.element_prop_features)
        np.save(os.path.join(self.processed_data_dir, 'property_features.npy'), 
                self.prop_features)
        np.save(os.path.join(self.processed_data_dir, 'deltaG_H.npy'), 
                self.deltaG_H)
        
        # Save element mapping
        pd.Series(self.elements).to_csv(
            os.path.join(self.processed_data_dir, 'element_mapping.csv'),
            index=False
        )
        
        # Save scalers
        import joblib
        joblib.dump(self.prop_scaler, 
                   os.path.join(self.processed_data_dir, 'property_scaler.pkl'))
        joblib.dump(self.dg_scaler, 
                   os.path.join(self.processed_data_dir, 'deltaG_scaler.pkl'))
        
        # Save feature information
        feature_info = {
            'composition_size': self.comp_features.shape[1],
            'element_prop_size': self.element_prop_features.shape[1],
            'property_size': self.prop_features.shape[1]
        }
        with open(os.path.join(self.processed_data_dir, 'feature_info.json'), 'w') as f:
            json.dump(feature_info, f)
        
        print("Saved all processed data and scalers")

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_data_dir = os.path.join(base_dir, 'data', 'raw')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    
    # Initialize processor
    processor = MXeneDataProcessor(raw_data_dir, processed_data_dir)
    
    # Process data
    processor.load_data()
    processor.preprocess_composition()
    processor.preprocess_properties()
    processor.merge_deltaG()
    processor.save_processed_data()

if __name__ == '__main__':
    main()
