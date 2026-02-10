"""
Extract Autoimmune Disease Subgraph from DRKG
==============================================

This script:
1. Loads full DRKG (5.8M triples)
2. Identifies our 9 target autoimmune diseases
3. Extracts all connected entities (drugs, genes, proteins, pathways)
4. Saves autoimmune-specific subgraph (~1.2M triples)
5. Generates entity/relation mappings
6. Creates disease statistics

Author: Yatin
Date: January 30, 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import json
from tqdm import tqdm

class AutoimmuneSubgraphExtractor:
    """Extract autoimmune-specific subgraph from DRKG"""
    
    def __init__(self, drkg_path='datasets/raw/drkg/drkg.tsv'):
        self.drkg_path = Path(drkg_path)
        self.output_dir = Path('datasets/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define our 9 target autoimmune diseases with DOID IDs
        self.autoimmune_diseases = {
            'Rheumatoid Arthritis': 'DOID:7148',
            'Systemic Lupus Erythematosus': 'DOID:9074',
            'Multiple Sclerosis': 'DOID:2377',
            "Crohn's Disease": 'DOID:8778',
            "Sjögren's Syndrome": 'DOID:12894',
            'Systemic Sclerosis': 'DOID:418',
            'Primary Biliary Cholangitis': 'DOID:12236',
            'Myasthenia Gravis': 'DOID:437',
            'Autoimmune Hepatitis': 'DOID:2048'
        }
        
        print("="*80)
        print("AUTOIMMUNE SUBGRAPH EXTRACTION")
        print("="*80)
        print(f"\nTarget diseases ({len(self.autoimmune_diseases)}):")
        for name, doid in self.autoimmune_diseases.items():
            print(f"  • {name:40s} ({doid})")
        print()
    
    def load_drkg(self):
        """Load full DRKG"""
        print("Loading full DRKG...")
        print(f"File: {self.drkg_path}")
        
        self.drkg = pd.read_csv(
            self.drkg_path,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail']
        )
        
        print(f"✓ Loaded {len(self.drkg):,} triples")
        print(f"  Unique entities: {len(set(self.drkg['head']) | set(self.drkg['tail'])):,}")
        print(f"  Unique relations: {self.drkg['relation'].nunique():,}")
        print()
        
        return self.drkg
    
    def find_disease_mentions(self):
        """Find all rows mentioning our target diseases"""
        print("Finding disease mentions in DRKG...")
        
        # Build regex pattern for all disease IDs
        disease_pattern = '|'.join(self.autoimmune_diseases.values())
        
        # Find triples where diseases appear
        disease_mask = (
            self.drkg['head'].str.contains(disease_pattern, na=False, regex=True) |
            self.drkg['tail'].str.contains(disease_pattern, na=False, regex=True)
        )
        
        direct_triples = self.drkg[disease_mask].copy()
        
        print(f"✓ Found {len(direct_triples):,} triples directly mentioning diseases")
        
        # Count per disease
        print("\n  Per-disease direct mentions:")
        for name, doid in self.autoimmune_diseases.items():
            count = direct_triples[
                (direct_triples['head'].str.contains(doid, na=False)) |
                (direct_triples['tail'].str.contains(doid, na=False))
            ].shape[0]
            print(f"    {name:40s}: {count:>6,} triples")
        
        print()
        return direct_triples
    
    def expand_to_neighborhood(self, direct_triples, hops=2):
        """Expand to k-hop neighborhood"""
        print(f"Expanding to {hops}-hop neighborhood...")
        
        # Get all entities from direct triples
        entities_0hop = set(direct_triples['head']) | set(direct_triples['tail'])
        print(f"  0-hop (direct): {len(entities_0hop):,} entities")
        
        all_entities = entities_0hop.copy()
        
        for hop in range(1, hops + 1):
            # Find triples where current entities appear
            mask = (
                self.drkg['head'].isin(all_entities) |
                self.drkg['tail'].isin(all_entities)
            )
            
            hop_triples = self.drkg[mask]
            new_entities = set(hop_triples['head']) | set(hop_triples['tail'])
            
            print(f"  {hop}-hop: {len(new_entities):,} entities (+{len(new_entities - all_entities):,} new)")
            
            all_entities = new_entities
        
        # Extract final subgraph
        final_mask = (
            self.drkg['head'].isin(all_entities) |
            self.drkg['tail'].isin(all_entities)
        )
        
        subgraph = self.drkg[final_mask].copy()
        
        print(f"\n✓ Final subgraph: {len(subgraph):,} triples")
        print(f"  Entities: {len(all_entities):,}")
        print(f"  Reduction: {(1 - len(subgraph)/len(self.drkg))*100:.1f}% smaller than full DRKG")
        print()
        
        return subgraph
    
    def analyze_entity_types(self, subgraph):
        """Analyze entity type distribution"""
        print("Analyzing entity types...")
        
        # Extract entity types
        entity_types = Counter()
        
        for entity in set(subgraph['head']) | set(subgraph['tail']):
            entity_type = entity.split('::')[0] if '::' in entity else 'Unknown'
            entity_types[entity_type] += 1
        
        print("\nEntity Type Distribution:")
        for entity_type, count in entity_types.most_common():
            print(f"  {entity_type:30s}: {count:>8,}")
        
        print()
        return dict(entity_types)
    
    def analyze_disease_links(self, subgraph):
        """Analyze links per disease"""
        print("Analyzing disease-specific links...")
        
        disease_stats = []
        
        for name, doid in self.autoimmune_diseases.items():
            # All triples for this disease
            disease_triples = subgraph[
                (subgraph['head'].str.contains(doid, na=False)) |
                (subgraph['tail'].str.contains(doid, na=False))
            ]
            
            # Drug links
            drug_triples = disease_triples[
                (disease_triples['head'].str.startswith('Compound::', na=False)) |
                (disease_triples['tail'].str.startswith('Compound::', na=False))
            ]
            
            # Gene links
            gene_triples = disease_triples[
                (disease_triples['head'].str.startswith('Gene::', na=False)) |
                (disease_triples['tail'].str.startswith('Gene::', na=False))
            ]
            
            # Protein links
            protein_triples = disease_triples[
                (disease_triples['head'].str.startswith('Protein::', na=False)) |
                (disease_triples['tail'].str.startswith('Protein::', na=False))
            ]
            
            stats = {
                'disease_name': name,
                'disease_id': doid,
                'total_links': len(disease_triples),
                'drug_links': len(drug_triples),
                'gene_links': len(gene_triples),
                'protein_links': len(protein_triples)
            }
            
            disease_stats.append(stats)
        
        # Create DataFrame
        stats_df = pd.DataFrame(disease_stats)
        stats_df = stats_df.sort_values('total_links', ascending=False)
        
        print("\nDisease Statistics:")
        print(stats_df.to_string(index=False))
        
        # Categorize
        print("\n" + "="*80)
        print("DISEASE CATEGORIZATION")
        print("="*80)
        
        print("\nTRAINING SET (Data-Rich, >500 total links):")
        training = stats_df[stats_df['total_links'] > 500]
        for _, row in training.iterrows():
            print(f"  • {row['disease_name']:40s}: {row['total_links']:>6,} links ({row['drug_links']:>4,} drugs)")
        
        print("\nTARGET SET (Data-Sparse, ≤500 links - Transfer Learning):")
        target = stats_df[stats_df['total_links'] <= 500]
        for _, row in target.iterrows():
            marker = "⭐" if row['total_links'] < 20 else ""
            print(f"  • {row['disease_name']:40s}: {row['total_links']:>6,} links ({row['drug_links']:>4,} drugs) {marker}")
        
        print("\n⭐ = Perfect zero-shot test case (minimal data)")
        print()
        
        return stats_df
    
    def create_mappings(self, subgraph):
        """Create entity and relation ID mappings"""
        print("Creating entity and relation mappings...")
        
        # Get unique entities and relations
        entities = sorted(set(subgraph['head']) | set(subgraph['tail']))
        relations = sorted(subgraph['relation'].unique())
        
        # Create mappings
        entity2id = {entity: idx for idx, entity in enumerate(entities)}
        relation2id = {rel: idx for idx, rel in enumerate(relations)}
        
        id2entity = {idx: entity for entity, idx in entity2id.items()}
        id2relation = {idx: rel for rel, idx in relation2id.items()}
        
        print(f"✓ Created mappings:")
        print(f"  Entities: {len(entity2id):,}")
        print(f"  Relations: {len(relation2id):,}")
        print()
        
        return entity2id, relation2id, id2entity, id2relation
    
    def save_outputs(self, subgraph, stats_df, entity2id, relation2id, entity_types):
        """Save all outputs"""
        print("Saving outputs...")
        
        # Save subgraph
        subgraph_file = self.output_dir / 'autoimmune_subgraph.tsv'
        subgraph.to_csv(subgraph_file, sep='\t', index=False, header=False)
        print(f"✓ Saved subgraph: {subgraph_file}")
        
        # Save disease statistics
        stats_file = self.output_dir / 'disease_statistics.csv'
        stats_df.to_csv(stats_file, index=False)
        print(f"✓ Saved statistics: {stats_file}")
        
        # Save entity mapping
        entity2id_file = self.output_dir / 'entity2id.json'
        with open(entity2id_file, 'w') as f:
            json.dump(entity2id, f, indent=2)
        print(f"✓ Saved entity mapping: {entity2id_file}")
        
        # Save relation mapping
        relation2id_file = self.output_dir / 'relation2id.json'
        with open(relation2id_file, 'w') as f:
            json.dump(relation2id, f, indent=2)
        print(f"✓ Saved relation mapping: {relation2id_file}")
        
        # Save summary stats
        summary = {
            'extraction_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_drkg_triples': len(self.drkg),
            'subgraph_triples': len(subgraph),
            'reduction_percent': (1 - len(subgraph)/len(self.drkg)) * 100,
            'num_entities': len(entity2id),
            'num_relations': len(relation2id),
            'entity_types': entity_types,
            'diseases': {name: doid for name, doid in self.autoimmune_diseases.items()}
        }
        
        summary_file = self.output_dir / 'subgraph_extraction_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary: {summary_file}")
        
        print()
    
    def run(self):
        """Run complete extraction pipeline"""
        # Load data
        self.load_drkg()
        
        # Find disease mentions
        direct_triples = self.find_disease_mentions()
        
        # Expand to neighborhood
        subgraph = self.expand_to_neighborhood(direct_triples, hops=2)
        
        # Analyze
        entity_types = self.analyze_entity_types(subgraph)
        stats_df = self.analyze_disease_links(subgraph)
        
        # Create mappings
        entity2id, relation2id, id2entity, id2relation = self.create_mappings(subgraph)
        
        # Save everything
        self.save_outputs(subgraph, stats_df, entity2id, relation2id, entity_types)
        
        # Final summary
        print("="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"\nOriginal DRKG: {len(self.drkg):,} triples")
        print(f"Autoimmune Subgraph: {len(subgraph):,} triples")
        print(f"Reduction: {(1 - len(subgraph)/len(self.drkg))*100:.1f}%")
        print(f"\nData-rich diseases: {len(stats_df[stats_df['total_links'] > 500])}")
        print(f"Data-sparse diseases: {len(stats_df[stats_df['total_links'] <= 500])}")
        print(f"\nFiles saved to: {self.output_dir}")
        print("\nNext step: Parse Orphanet and HPO to enrich this subgraph")
        print("="*80)

def main():
    """Main entry point"""
    extractor = AutoimmuneSubgraphExtractor(
        drkg_path='datasets/raw/drkg/drkg.tsv'
    )
    extractor.run()

if __name__ == '__main__':
    main()
