"""
Background Regulatory Graph (BRG) Builder
==========================================

Extracts immune-specific regulatory subgraph from enriched DRKG:
- Gene-gene interactions
- Immune pathway relationships
- Cytokine cascades
- Autoimmune-relevant mechanisms

The BRG provides biological context for path-based reasoning in BioPathNet++.

Author: Yatin
Date: February 9, 2026
"""

import pandas as pd
import json
from pathlib import Path
from collections import Counter

class BRGBuilder:
    """Build autoimmune-specific Background Regulatory Graph"""
    
    def __init__(self, processed_dir='datasets/processed'):
        self.processed_dir = Path(processed_dir)
        
        # Define immune-relevant keywords for filtering
        self.immune_keywords = [
            # Cytokines
            'TNF', 'IL-', 'IFN', 'interleukin', 'interferon', 'tumor necrosis',
            # Signaling pathways
            'JAK', 'STAT', 'NFkB', 'NF-kB', 'MAPK', 'PI3K', 'AKT',
            # Immune cells
            'T-cell', 'B-cell', 'lymphocyte', 'CD4', 'CD8', 'CD20',
            # Processes
            'inflammation', 'immune', 'autoimmune', 'cytokine',
            'chemokine', 'antibody', 'antigen',
            # Complement
            'complement', 'C3', 'C4', 'C5',
            # MHC
            'MHC', 'HLA-',
        ]
        
        print("="*80)
        print("BACKGROUND REGULATORY GRAPH (BRG) BUILDER")
        print("="*80)
        print()
    
    def load_enriched_graph(self):
        """Load enriched knowledge graph"""
        print("Loading enriched graph...")
        
        enriched_file = self.processed_dir / 'enriched_drkg.tsv'
        
        self.graph = pd.read_csv(
            enriched_file,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail']
        )
        
        print(f"  ✓ Loaded {len(self.graph):,} triples")
        print()
        
        return self.graph
    
    def extract_gene_interactions(self):
        """Extract gene-gene interactions"""
        print("Extracting gene-gene interactions...")
        
        # Gene-gene edges (both entities are genes)
        gene_mask = (
            self.graph['head'].str.startswith('Gene::', na=False) &
            self.graph['tail'].str.startswith('Gene::', na=False)
        )
        
        gene_gene = self.graph[gene_mask].copy()
        
        print(f"  ✓ Found {len(gene_gene):,} gene-gene interactions")
        
        # Show relation types
        print("\n  Relation types:")
        rel_counts = gene_gene['relation'].value_counts().head(10)
        for rel, count in rel_counts.items():
            print(f"    {rel[:60]:60s}: {count:>6,}")
        
        print()
        return gene_gene
    
    def extract_immune_edges(self):
        """Extract immune-related edges"""
        print("Extracting immune-relevant edges...")
        
        # Create pattern from keywords
        pattern = '|'.join(self.immune_keywords)
        
        # Filter edges where head OR tail contains immune keywords
        immune_mask = (
            self.graph['head'].str.contains(pattern, case=False, na=False) |
            self.graph['tail'].str.contains(pattern, case=False, na=False)
        )
        
        immune_edges = self.graph[immune_mask].copy()
        
        print(f"  ✓ Found {len(immune_edges):,} immune-related edges")
        
        # Entity type distribution
        entity_types = Counter()
        for entity in set(immune_edges['head']) | set(immune_edges['tail']):
            entity_type = entity.split('::')[0] if '::' in entity else 'Unknown'
            entity_types[entity_type] += 1
        
        print("\n  Entity types in immune subgraph:")
        for etype, count in entity_types.most_common(10):
            print(f"    {etype:30s}: {count:>6,}")
        
        print()
        return immune_edges
    
    def extract_pathway_edges(self):
        """Extract pathway-related edges"""
        print("Extracting pathway edges...")
        
        # Edges involving pathways
        pathway_mask = (
            self.graph['head'].str.startswith('Pathway::', na=False) |
            self.graph['tail'].str.startswith('Pathway::', na=False)
        )
        
        pathway_edges = self.graph[pathway_mask].copy()
        
        print(f"  ✓ Found {len(pathway_edges):,} pathway edges")
        
        # Filter for immune pathways
        immune_pattern = '|'.join(self.immune_keywords)
        immune_pathway_mask = (
            pathway_edges['head'].str.contains(immune_pattern, case=False, na=False) |
            pathway_edges['tail'].str.contains(immune_pattern, case=False, na=False)
        )
        
        immune_pathways = pathway_edges[immune_pathway_mask].copy()
        
        print(f"  ✓ Immune-relevant pathways: {len(immune_pathways):,}")
        print()
        
        return pathway_edges, immune_pathways
    
    def build_brg(self):
        """Combine all BRG components"""
        print("Building complete BRG...")
        
        # Get all components
        gene_gene = self.extract_gene_interactions()
        immune_edges = self.extract_immune_edges()
        pathway_edges, immune_pathways = self.extract_pathway_edges()
        
        # Combine (with deduplication)
        print("  Merging components...")
        brg = pd.concat([
            gene_gene,
            immune_edges,
            immune_pathways
        ], ignore_index=True)
        
        # Remove duplicates
        before = len(brg)
        brg = brg.drop_duplicates(subset=['head', 'relation', 'tail'])
        after = len(brg)
        
        print(f"  ✓ Combined: {before:,} edges")
        print(f"  ✓ After deduplication: {after:,} edges")
        
        if before > after:
            print(f"    Removed {before - after:,} duplicates")
        
        print()
        return brg
    
    def analyze_brg(self, brg):
        """Analyze BRG composition"""
        print("Analyzing BRG...")
        
        # Entity counts
        entities = set(brg['head']) | set(brg['tail'])
        relations = set(brg['relation'])
        
        print(f"\n  BRG Statistics:")
        print(f"    Edges: {len(brg):,}")
        print(f"    Entities: {len(entities):,}")
        print(f"    Relations: {len(relations):,}")
        
        # Entity type distribution
        entity_types = Counter()
        for entity in entities:
            entity_type = entity.split('::')[0] if '::' in entity else 'Unknown'
            entity_types[entity_type] += 1
        
        print(f"\n  Entity Type Distribution:")
        for etype, count in entity_types.most_common():
            print(f"    {etype:30s}: {count:>6,}")
        
        # Degree distribution (skipped - too slow for large graphs)
        # Approximate average degree
        avg_degree = (2 * len(brg)) / len(entities) if len(entities) > 0 else 0
        
        print(f"\n  Degree Statistics:")
        print(f"    Average degree (approx): {avg_degree:.2f}")
        print(f"    (Detailed analysis skipped for performance)")
        
        print()
        
        return {
            'num_edges': len(brg),
            'num_entities': len(entities),
            'num_relations': len(relations),
            'entity_types': dict(entity_types),
            'avg_degree_approx': avg_degree
        }
    
    def save_brg(self, brg, stats):
        """Save BRG"""
        print("Saving BRG...")
        
        # Save graph
        brg_file = self.processed_dir / 'autoimmune_brg.tsv'
        brg.to_csv(brg_file, sep='\t', index=False, header=False)
        print(f"✓ Saved BRG: {brg_file}")
        print(f"  {len(brg):,} edges")
        
        # Save statistics
        stats_file = self.processed_dir / 'brg_statistics.json'
        stats['creation_date'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics: {stats_file}")
        
        print()
    
    def run(self):
        """Run complete BRG construction"""
        # Load graph
        self.load_enriched_graph()
        
        # Build BRG
        brg = self.build_brg()
        
        # Analyze
        stats = self.analyze_brg(brg)
        
        # Save
        self.save_brg(brg, stats)
        
        # Final summary
        print("="*80)
        print("BRG CONSTRUCTION COMPLETE")
        print("="*80)
        print(f"\nBRG edges: {len(brg):,}")
        print(f"BRG entities: {stats['num_entities']:,}")
        print(f"BRG relations: {stats['num_relations']:,}")
        print(f"\nOutput file: {self.processed_dir / 'autoimmune_brg.tsv'}")
        print("\nNext step: Create PyTorch Geometric data structures")
        print("="*80)

def main():
    builder = BRGBuilder()
    builder.run()

if __name__ == '__main__':
    main()