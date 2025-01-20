import os
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from collections import defaultdict

class MXeneNoveltyChecker:
    def __init__(self, base_dir):
        """Initialize checker with paths"""
        self.data_dir = os.path.join(base_dir, 'data', 'processed')
        self.known_mxenes = self._load_known_mxenes()
        
    def _load_known_mxenes(self):
        """Load known MXenes from our dataset"""
        try:
            # Load composition data
            compositions = pd.read_csv(os.path.join(self.data_dir, 'compositions.csv'))
            return set(compositions['composition'].tolist())
        except FileNotFoundError:
            print("Warning: compositions.csv not found in data directory")
            return set()
    
    def _format_mxene_formula(self, metal, x_element, composition):
        """Format MXene chemical formula"""
        metal_ratio = composition[metal]
        x_ratio = composition[x_element]
        
        # Determine M:X ratio (usually 2:1 or 3:2)
        if abs(metal_ratio / x_ratio - 2) < abs(metal_ratio / x_ratio - 1.5):
            return f"{metal}2{x_element}Tx"
        else:
            return f"{metal}3{x_element}2Tx"
    
    def search_google_scholar(self, formula):
        """Search Google Scholar for MXene formula
        Note: This is a simple implementation. For production, consider using official APIs
        """
        base_url = "https://scholar.google.com/scholar"
        query = f"{formula} MXene"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        try:
            response = requests.get(
                base_url,
                params={'q': query, 'hl': 'en'},
                headers=headers
            )
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', class_='gs_ri')
                return len(results)
            else:
                print(f"Warning: Could not access Google Scholar (Status code: {response.status_code})")
                return None
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            return None
    
    def check_novelty(self, designs):
        """Check novelty of designed MXenes"""
        results = []
        
        for i, design in enumerate(designs):
            elements = design['composition']['elements']
            probabilities = design['composition']['probabilities']
            
            # Create composition dictionary
            composition = dict(zip(elements, probabilities))
            
            # Find metal and X elements
            metals = [elem for elem in elements if elem in ['Ti', 'V', 'Cr', 'Mn']]
            x_elements = [elem for elem in elements if elem in ['B', 'C', 'N']]
            
            if not metals or not x_elements:
                print(f"Warning: Design {i+1} missing metal or X element")
                continue
            
            # Get primary metal and X element (highest probability)
            primary_metal = max(metals, key=lambda m: composition[m])
            primary_x = max(x_elements, key=lambda x: composition[x])
            
            # Format chemical formula
            formula = self._format_mxene_formula(primary_metal, primary_x, composition)
            
            # Check if in our dataset
            in_dataset = formula in self.known_mxenes
            
            # Search literature (with delay to avoid rate limiting)
            time.sleep(2)  # Be nice to Google Scholar
            citation_count = self.search_google_scholar(formula)
            
            results.append({
                'design_number': i + 1,
                'formula': formula,
                'composition': composition,
                'properties': design['properties']['predicted'],
                'in_dataset': in_dataset,
                'literature_citations': citation_count,
                'novelty_score': 0 if in_dataset else (0 if citation_count is None else 1 / (1 + citation_count))
            })
        
        return results

def print_novelty_results(results):
    """Print novelty analysis results"""
    print("\nMXene Novelty Analysis")
    print("=" * 50)
    
    for result in results:
        print(f"\nDesign {result['design_number']}: {result['formula']}")
        print("-" * 30)
        
        print("Composition:")
        for elem, prob in result['composition'].items():
            print(f"  {elem}: {prob:.3f}")
        
        print("\nKey Properties:")
        print(f"  E_coh: {result['properties']['E_coh']:.2f} eV")
        print(f"  WF: {result['properties']['WF']:.2f} eV")
        print(f"  W_sur: {result['properties']['W_sur']:.2f} meV/Å²")
        
        print("\nNovelty Assessment:")
        print(f"  In Dataset: {'Yes' if result['in_dataset'] else 'No'}")
        if result['literature_citations'] is not None:
            print(f"  Literature Citations: {result['literature_citations']}")
            print(f"  Novelty Score: {result['novelty_score']:.3f}")
        else:
            print("  Literature Search Failed")
        
        print(f"  Recommendation: ", end="")
        if not result['in_dataset'] and (result['literature_citations'] is None or result['literature_citations'] < 5):
            print("Potentially novel MXene! Consider for synthesis.")
        else:
            print("Known MXene structure.")

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Add src directory to Python path
    import sys
    sys.path.append(os.path.join(base_dir, 'src', 'ml'))
    
    # Load designs from previous run
    from design_mxene import MXeneDesigner
    
    # Example target properties (based on a good HER catalyst)
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
    
    # Check novelty
    checker = MXeneNoveltyChecker(base_dir)
    results = checker.check_novelty(designs)
    
    # Print results
    print_novelty_results(results)
