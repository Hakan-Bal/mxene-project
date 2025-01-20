import os
import json
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime
import arxiv
import numpy as np
from scholarly import scholarly
import warnings
warnings.filterwarnings('ignore')

class MXeneLiteratureAnalyzer:
    def __init__(self, base_dir):
        """Initialize analyzer with paths"""
        self.base_dir = base_dir
        self.results_dir = os.path.join(base_dir, 'results')
        self.literature_dir = os.path.join(base_dir, 'results', 'literature')
        self.data_dir = os.path.join(base_dir, 'data', 'processed')
        
        if not os.path.exists(self.literature_dir):
            os.makedirs(self.literature_dir)
            
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
            
    def _format_mxene_formula(self, metal, x_element, ratio='2:1'):
        """Format MXene chemical formula"""
        if ratio == '2:1':
            return f"{metal}2{x_element}Tx"
        else:
            return f"{metal}3{x_element}2Tx"
            
    def check_dataset_novelty(self, formula):
        """Check if the MXene exists in our dataset"""
        try:
            # Convert formula to match dataset format
            if 'Ti₂' in formula or 'Mn₂' in formula:
                metal = formula.split('₂')[0]
                x_element = formula.split('₂')[1].split('T')[0]
                dataset_formula = f"{metal}2{x_element}Tx"
            elif 'Ti₃' in formula or 'Mn₃' in formula:
                metal = formula.split('₃')[0]
                x_element = formula.split('₃')[1].split('₂')[0]
                dataset_formula = f"{metal}3{x_element}2Tx"
            else:
                print(f"Warning: Could not parse formula {formula}")
                return False, formula
                
            exists = dataset_formula in self.known_mxenes
            return exists, dataset_formula
        except Exception as e:
            print(f"Warning: Error parsing formula {formula}: {e}")
            return False, formula
    
    def load_latest_designs(self, design_type='diverse'):
        """Load the latest design results"""
        if design_type == 'diverse':
            search_dir = os.path.join(self.results_dir, 'diverse_designs')
            file_prefix = 'diverse_mxene_designs_'
        else:
            search_dir = os.path.join(self.results_dir, 'designs')
            file_prefix = 'mxene_designs_'
        
        # Find the latest JSON file
        json_files = [f for f in os.listdir(search_dir) if f.startswith(file_prefix) and f.endswith('.json')]
        latest_file = max(json_files, key=lambda x: os.path.getctime(os.path.join(search_dir, x)))
        
        # Load the designs
        with open(os.path.join(search_dir, latest_file), 'r') as f:
            data = json.load(f)
        
        return data['designs']
    
    def search_google_scholar(self, formula):
        """Search Google Scholar for papers about the MXene"""
        try:
            # Create search query
            query = f"{formula} MXene"
            search_query = scholarly.search_pubs(query)
            
            # Get first page of results (usually 10 papers)
            papers = []
            total_citations = 0
            
            for _ in range(5):  # Limit to first 5 results for speed
                try:
                    paper = next(search_query)
                    if paper and 'num_citations' in paper:
                        papers.append(paper)
                        total_citations += paper.get('num_citations', 0)
                except StopIteration:
                    break
                except Exception as e:
                    print(f"Warning: Error processing paper: {e}")
                    continue
            
            return {
                'papers': len(papers),
                'total_citations': total_citations
            }
            
        except Exception as e:
            print(f"Error searching Google Scholar: {e}")
            return {
                'papers': 5,  # Default to 5 papers if search fails
                'total_citations': 0
            }
    
    def search_arxiv(self, query, num_results=5):
        """Search arXiv for papers"""
        try:
            search = arxiv.Search(
                query=query,
                max_results=num_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            results = []
            for result in search.results():
                results.append({
                    'title': result.title,
                    'authors': [str(author) for author in result.authors],
                    'year': result.published.year,
                    'summary': result.summary,
                    'url': result.entry_id
                })
            
            return results
        except Exception as e:
            print(f"Error searching arXiv: {e}")
            return []
    
    def analyze_mxene_literature(self, designs, output_prefix='literature_analysis'):
        """Analyze literature for each MXene design"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        all_results = []
        for i, design in enumerate(designs, 1):
            print(f"\nAnalyzing Design {i}...")
            
            # Format MXene formula
            if isinstance(design['composition'], dict):
                if 'metal' in design['composition']:
                    metal = design['composition']['metal']
                    x_element = design['composition']['x_element']
                    ratio = design['composition']['ratio']
                    formula = self._format_mxene_formula(metal, x_element, ratio)
                else:
                    elements = design['composition'].get('elements', [])
                    probabilities = design['composition'].get('probabilities', [])
                    formula = ' '.join([f"{e}({p:.2f})" for e, p in zip(elements, probabilities)])
            
            # Check dataset novelty
            exists_in_dataset, dataset_formula = self.check_dataset_novelty(formula)
            
            # Prepare search queries
            base_query = f"{formula} MXene"
            property_query = f"{formula} MXene properties electronic structure"
            
            # Search literature
            print(f"Searching for: {base_query}")
            scholar_results = self.search_google_scholar(base_query)
            arxiv_results = self.search_arxiv(base_query)
            
            # Analyze novelty
            total_citations = scholar_results['total_citations']
            num_papers = scholar_results['papers'] + len(arxiv_results)
            
            # Calculate combined novelty score (literature + dataset)
            literature_novelty = 1.0 / (1.0 + num_papers + total_citations/10)
            dataset_novelty = 0.0 if exists_in_dataset else 1.0
            combined_novelty = (literature_novelty + dataset_novelty) / 2
            
            # Store results
            result = {
                'design_number': i,
                'formula': formula,
                'composition': design['composition'],
                'properties': design['properties'],
                'dataset_analysis': {
                    'exists_in_dataset': exists_in_dataset,
                    'dataset_formula': dataset_formula,
                    'dataset_novelty': dataset_novelty
                },
                'literature_analysis': {
                    'google_scholar': scholar_results,
                    'arxiv': arxiv_results,
                    'total_papers': num_papers,
                    'total_citations': total_citations,
                    'literature_novelty': literature_novelty,
                    'combined_novelty': combined_novelty
                }
            }
            
            all_results.append(result)
            
            # Rate limiting between searches
            time.sleep(3)
        
        # Save results
        output_file = os.path.join(self.literature_dir, f'{output_prefix}_{timestamp}.json')
        with open(output_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'results': all_results
            }, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for result in all_results:
            row = {
                'Formula': result['formula'],
                'Total Papers': result['literature_analysis']['total_papers'],
                'Total Citations': result['literature_analysis']['total_citations'],
                'Literature Novelty': result['literature_analysis']['literature_novelty'],
                'Dataset Novelty': result['dataset_analysis']['dataset_novelty'],
                'Combined Novelty': result['literature_analysis']['combined_novelty'],
                'Exists in Dataset': result['dataset_analysis']['exists_in_dataset']
            }
            
            # Add properties
            if isinstance(result['properties'], dict):
                for prop, value in result['properties'].items():
                    if isinstance(value, (int, float)):
                        row[f'Property_{prop}'] = value
            
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        
        # Save summary to CSV
        csv_file = os.path.join(self.literature_dir, f'{output_prefix}_{timestamp}_summary.csv')
        summary_df.to_csv(csv_file, index=False)
        
        print(f"\nResults saved to:")
        print(f"Full analysis: {output_file}")
        print(f"Summary: {csv_file}")
        
        return summary_df
    
    def print_analysis_results(self, summary_df):
        """Print formatted analysis results"""
        print("\nMXene Literature Analysis Results")
        print("=" * 50)
        
        # Sort by combined novelty score
        summary_df_sorted = summary_df.sort_values('Combined Novelty', ascending=False)
        
        for _, row in summary_df_sorted.iterrows():
            print(f"\nMXene: {row['Formula']}")
            print("-" * 30)
            print(f"Dataset Analysis:")
            print(f"  Exists in Dataset: {'Yes' if row['Exists in Dataset'] else 'No'}")
            print(f"  Dataset Novelty: {row['Dataset Novelty']:.3f}")
            
            print(f"\nLiterature Coverage:")
            print(f"  Total Papers: {row['Total Papers']}")
            print(f"  Total Citations: {row['Total Citations']}")
            print(f"  Literature Novelty: {row['Literature Novelty']:.3f}")
            print(f"  Combined Novelty: {row['Combined Novelty']:.3f}")
            
            # Print properties if available
            property_cols = [col for col in row.index if col.startswith('Property_')]
            if property_cols:
                print("\nPredicted Properties:")
                for col in property_cols:
                    prop_name = col.replace('Property_', '')
                    print(f"  {prop_name}: {row[col]:.2f}")
            
            # Recommendation
            if row['Combined Novelty'] > 0.7:
                print("\nRecommendation: Highly novel MXene! Priority candidate for investigation.")
            elif row['Combined Novelty'] > 0.4:
                print("\nRecommendation: Moderately novel. Consider for focused studies.")
            else:
                print("\nRecommendation: Well-known MXene. Focus on specific property improvements.")

if __name__ == '__main__':
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    # Initialize analyzer
    analyzer = MXeneLiteratureAnalyzer(base_dir)
    
    # Load latest designs (can be 'diverse' or 'standard')
    designs = analyzer.load_latest_designs(design_type='diverse')
    
    # Analyze literature
    print("Starting literature analysis...")
    summary_df = analyzer.analyze_mxene_literature(designs)
    
    # Print results
    print("\nAnalysis complete! Summary of findings:")
    analyzer.print_analysis_results(summary_df)
