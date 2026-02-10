"""
DRKG Disease Identifier Diagnostic
===================================

This script investigates how autoimmune diseases are actually encoded in DRKG
and fixes the extraction pattern.

Author: Yatin
Date: February 9, 2026
"""

import pandas as pd
from pathlib import Path
import re

class DRKGDiagnostic:
    """Diagnose disease identifier patterns in DRKG"""
    
    def __init__(self, drkg_path='datasets/raw/drkg/drkg.tsv'):
        self.drkg_path = Path(drkg_path)
        
        # Our target disease names
        self.disease_keywords = [
            'rheumatoid',
            'lupus',
            'sclerosis',
            'multiple sclerosis',
            'crohn',
            'sjogren',
            'sjögren',
            'myasthenia',
            'biliary',
            'hepatitis'
        ]
    
    def load_sample(self, nrows=100000):
        """Load a sample of DRKG for quick analysis"""
        print("Loading DRKG sample (first 100K rows)...")
        
        self.drkg_sample = pd.read_csv(
            self.drkg_path,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail'],
            nrows=nrows
        )
        
        print(f"✓ Loaded {len(self.drkg_sample):,} triples")
        return self.drkg_sample
    
    def find_disease_entities(self):
        """Find all disease entities in DRKG sample"""
        print("\nFinding disease entities...")
        
        # Get all unique entities
        all_entities = set(self.drkg_sample['head']) | set(self.drkg_sample['tail'])
        
        # Filter for Disease entities
        disease_entities = [e for e in all_entities if e.startswith('Disease::')]
        
        print(f"✓ Found {len(disease_entities)} unique disease entities in sample")
        
        # Show examples
        print("\nSample disease entities:")
        for i, entity in enumerate(sorted(disease_entities)[:10]):
            print(f"  {i+1}. {entity}")
        
        return disease_entities
    
    def search_for_autoimmune_diseases(self, disease_entities):
        """Search for our autoimmune diseases"""
        print("\n" + "="*80)
        print("SEARCHING FOR AUTOIMMUNE DISEASES")
        print("="*80)
        
        found_diseases = {}
        
        for keyword in self.disease_keywords:
            print(f"\nSearching for '{keyword}'...")
            
            matches = [
                d for d in disease_entities 
                if keyword.lower() in d.lower()
            ]
            
            if matches:
                print(f"  ✓ Found {len(matches)} matches:")
                for match in matches[:5]:  # Show first 5
                    print(f"    • {match}")
                if len(matches) > 5:
                    print(f"    ... and {len(matches) - 5} more")
                
                found_diseases[keyword] = matches
            else:
                print(f"  ✗ No matches found")
        
        return found_diseases
    
    def analyze_disease_format(self, disease_entities):
        """Analyze how diseases are formatted"""
        print("\n" + "="*80)
        print("DISEASE ID FORMAT ANALYSIS")
        print("="*80)
        
        # Extract ID patterns
        patterns = {}
        
        for entity in disease_entities[:50]:  # Analyze first 50
            # Extract the ID part after 'Disease::'
            id_part = entity.replace('Disease::', '')
            
            # Check what database it's from
            if id_part.startswith('DOID:'):
                pattern = 'DOID'
            elif id_part.startswith('MESH:'):
                pattern = 'MESH'
            elif id_part.startswith('UMLS:'):
                pattern = 'UMLS'
            else:
                pattern = 'Other'
            
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        print("\nID Format Distribution (sample):")
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pattern:10s}: {count:>3} entities")
        
        return patterns
    
    def full_scan_for_diseases(self):
        """Full scan of DRKG for our specific diseases"""
        print("\n" + "="*80)
        print("FULL DRKG SCAN FOR AUTOIMMUNE DISEASES")
        print("="*80)
        print("\nThis will take 1-2 minutes...")
        
        # Load full DRKG
        drkg_full = pd.read_csv(
            self.drkg_path,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail']
        )
        
        print(f"✓ Loaded full DRKG: {len(drkg_full):,} triples")
        
        # Get all disease entities
        all_entities = set(drkg_full['head']) | set(drkg_full['tail'])
        disease_entities = [e for e in all_entities if e.startswith('Disease::')]
        
        print(f"✓ Total disease entities: {len(disease_entities):,}")
        
        # Search for each keyword
        print("\nSearching for autoimmune diseases...")
        
        results = {}
        
        for keyword in self.disease_keywords:
            matches = [d for d in disease_entities if keyword.lower() in d.lower()]
            
            if matches:
                # Count triples for each match
                for match in matches:
                    triple_count = len(drkg_full[
                        (drkg_full['head'] == match) | (drkg_full['tail'] == match)
                    ])
                    
                    results[match] = {
                        'keyword': keyword,
                        'entity': match,
                        'triple_count': triple_count
                    }
        
        # Display results
        print("\n" + "="*80)
        print("FOUND AUTOIMMUNE DISEASES IN DRKG")
        print("="*80)
        
        if results:
            results_df = pd.DataFrame.from_dict(results, orient='index')
            results_df = results_df.sort_values('triple_count', ascending=False)
            
            print(f"\nTotal found: {len(results_df)}")
            print("\n" + results_df.to_string())
            
            # Save results
            output_file = Path('datasets/processed/drkg_disease_entities.csv')
            results_df.to_csv(output_file)
            print(f"\n✓ Saved to: {output_file}")
        else:
            print("\n✗ No autoimmune diseases found!")
            print("\nThis suggests DRKG might use different naming conventions.")
            print("Checking alternative approaches...")
        
        return results
    
    def run(self):
        """Run complete diagnostic"""
        print("="*80)
        print("DRKG DISEASE IDENTIFIER DIAGNOSTIC")
        print("="*80)
        print()
        
        # Quick sample analysis
        self.load_sample()
        disease_entities = self.find_disease_entities()
        self.analyze_disease_format(disease_entities)
        found = self.search_for_autoimmune_diseases(disease_entities)
        
        # Full scan
        results = self.full_scan_for_diseases()
        
        print("\n" + "="*80)
        print("DIAGNOSTIC COMPLETE")
        print("="*80)
        print("\nNext: Use the found entity names to fix the extraction script")

def main():
    diagnostic = DRKGDiagnostic()
    diagnostic.run()

if __name__ == '__main__':
    main()
