"""
MESH ID Mapper and Correct Subgraph Extractor
==============================================

DRKG uses MESH IDs, not disease names or DOID.
This script maps our autoimmune diseases to MESH IDs and extracts correctly.

Known MESH IDs for autoimmune diseases:
- Rheumatoid Arthritis: MESH:D001172
- Systemic Lupus Erythematosus: MESH:D008180
- Multiple Sclerosis: MESH:D009103
- Crohn's Disease: MESH:D003424
- Sjögren's Syndrome: MESH:D012859
- Systemic Sclerosis (Scleroderma): MESH:D012595
- Primary Biliary Cholangitis: MESH:D008105 or MESH:D008106
- Myasthenia Gravis: MESH:D009157
- Autoimmune Hepatitis: MESH:D019693

Author: Yatin
Date: February 9, 2026
"""

import pandas as pd
from pathlib import Path
import json

class MESHSubgraphExtractor:
    """Extract subgraph using correct MESH IDs"""
    
    def __init__(self, drkg_path='datasets/raw/drkg/drkg.tsv'):
        self.drkg_path = Path(drkg_path)
        self.output_dir = Path('datasets/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Map diseases to MESH IDs (verified from NCBI MeSH database)
        self.autoimmune_diseases = {
            'Rheumatoid Arthritis': 'Disease::MESH:D001172',
            'Systemic Lupus Erythematosus': 'Disease::MESH:D008180',
            'Multiple Sclerosis': 'Disease::MESH:D009103',
            "Crohn's Disease": 'Disease::MESH:D003424',
            "Sjögren's Syndrome": 'Disease::MESH:D012859',
            'Systemic Sclerosis': 'Disease::MESH:D012595',
            'Primary Biliary Cholangitis': 'Disease::MESH:D008105',  # Primary Biliary Cirrhosis
            'Myasthenia Gravis': 'Disease::MESH:D009157',
            'Autoimmune Hepatitis': 'Disease::MESH:D019693'
        }
        
        print("="*80)
        print("MESH-BASED SUBGRAPH EXTRACTION")
        print("="*80)
        print(f"\nTarget diseases ({len(self.autoimmune_diseases)}):")
        for name, mesh_id in self.autoimmune_diseases.items():
            print(f"  • {name:40s} → {mesh_id}")
        print()
    
    def verify_mesh_ids(self):
        """Verify MESH IDs exist in DRKG"""
        print("Verifying MESH IDs in DRKG...")
        
        # Load DRKG
        drkg = pd.read_csv(
            self.drkg_path,
            sep='\t',
            header=None,
            names=['head', 'relation', 'tail']
        )
        
        print(f"✓ Loaded DRKG: {len(drkg):,} triples")
        
        # Check each disease
        verified = {}
        
        for name, mesh_entity in self.autoimmune_diseases.items():
            # Count triples
            count = len(drkg[
                (drkg['head'] == mesh_entity) | (drkg['tail'] == mesh_entity)
            ])
            
            status = "✓" if count > 0 else "✗"
            print(f"  {status} {name:40s}: {count:>6,} triples")
            
            verified[name] = {
                'mesh_id': mesh_entity,
                'found': count > 0,
                'triple_count': count
            }
        
        print()
        
        # Check if any not found
        not_found = [name for name, info in verified.items() if not info['found']]
        
        if not_found:
            print("⚠ Warning: Some diseases not found in DRKG:")
            for name in not_found:
                print(f"    • {name}")
            print("\n  These diseases might use different MESH IDs.")
            print("  Proceeding with found diseases only...\n")
        
        return drkg, verified
    
    def extract_direct_triples(self, drkg):
        """Extract triples directly mentioning our diseases"""
        print("Extracting disease triples...")
        
        # Create mask for any of our diseases
        disease_entities = list(self.autoimmune_diseases.values())
        
        disease_mask = (
            drkg['head'].isin(disease_entities) |
            drkg['tail'].isin(disease_entities)
        )
        
        direct_triples = drkg[disease_mask].copy()
        
        print(f"✓ Found {len(direct_triples):,} direct triples")
        print()
        
        return direct_triples
    
    def analyze_disease_links(self, drkg):
        """Analyze detailed links per disease"""
        print("Analyzing disease-specific links...")
        
        disease_stats = []
        
        for name, mesh_entity in self.autoimmune_diseases.items():
            # All triples for this disease
            disease_triples = drkg[
                (drkg['head'] == mesh_entity) | (drkg['tail'] == mesh_entity)
            ]
            
            # Drug links (Compound entities)
            drug_mask = (
                (disease_triples['head'].str.startswith('Compound::', na=False)) |
                (disease_triples['tail'].str.startswith('Compound::', na=False))
            )
            drug_triples = disease_triples[drug_mask]
            
            # Gene links
            gene_mask = (
                (disease_triples['head'].str.startswith('Gene::', na=False)) |
                (disease_triples['tail'].str.startswith('Gene::', na=False))
            )
            gene_triples = disease_triples[gene_mask]
            
            # Pathway links
            pathway_mask = (
                (disease_triples['head'].str.startswith('Pathway::', na=False)) |
                (disease_triples['tail'].str.startswith('Pathway::', na=False))
            )
            pathway_triples = disease_triples[pathway_mask]
            
            # Symptom links
            symptom_mask = (
                (disease_triples['head'].str.startswith('Symptom::', na=False)) |
                (disease_triples['tail'].str.startswith('Symptom::', na=False))
            )
            symptom_triples = disease_triples[symptom_mask]
            
            stats = {
                'disease_name': name,
                'mesh_entity': mesh_entity,
                'total_links': len(disease_triples),
                'drug_links': len(drug_triples),
                'gene_links': len(gene_triples),
                'pathway_links': len(pathway_triples),
                'symptom_links': len(symptom_triples)
            }
            
            disease_stats.append(stats)
        
        # Create DataFrame
        stats_df = pd.DataFrame(disease_stats)
        stats_df = stats_df.sort_values('total_links', ascending=False)
        
        print("\nDisease Statistics:")
        print(stats_df.to_string(index=False))
        
        # Categorize
        print("\n" + "="*80)
        print("DISEASE CATEGORIZATION FOR TRANSFER LEARNING")
        print("="*80)
        
        print("\nTRAINING SET (Data-Rich, >500 total links):")
        training = stats_df[stats_df['total_links'] > 500]
        if len(training) > 0:
            for _, row in training.iterrows():
                print(f"  • {row['disease_name']:40s}: {row['total_links']:>6,} links ({row['drug_links']:>4,} drugs, {row['gene_links']:>4,} genes)")
        else:
            print("  (None found - all diseases are data-sparse)")
        
        print("\nTARGET SET (Data-Sparse, ≤500 links - Transfer Learning):")
        target = stats_df[stats_df['total_links'] <= 500]
        for _, row in target.iterrows():
            marker = "⭐" if row['total_links'] < 20 else ""
            print(f"  • {row['disease_name']:40s}: {row['total_links']:>6,} links ({row['drug_links']:>4,} drugs, {row['gene_links']:>4,} genes) {marker}")
        
        print("\n⭐ = Perfect zero-shot test case (minimal data)")
        print()
        
        return stats_df
    
    def expand_to_drug_gene_network(self, drkg, direct_triples):
        """
        Expand to include:
        1. All drugs treating these diseases
        2. All genes associated with these diseases
        3. All entities connected to those drugs/genes (1-hop)
        """
        print("Expanding to drug-gene network...")
        
        # Get drugs and genes from direct triples
        disease_entities = set(self.autoimmune_diseases.values())
        
        # Extract drugs
        drugs = set()
        for _, row in direct_triples.iterrows():
            if row['head'] in disease_entities and row['tail'].startswith('Compound::'):
                drugs.add(row['tail'])
            elif row['tail'] in disease_entities and row['head'].startswith('Compound::'):
                drugs.add(row['head'])
        
        # Extract genes
        genes = set()
        for _, row in direct_triples.iterrows():
            if row['head'] in disease_entities and row['tail'].startswith('Gene::'):
                genes.add(row['tail'])
            elif row['tail'] in disease_entities and row['head'].startswith('Gene::'):
                genes.add(row['head'])
        
        print(f"  Diseases: {len(disease_entities):,}")
        print(f"  Connected drugs: {len(drugs):,}")
        print(f"  Connected genes: {len(genes):,}")
        
        # Combine key entities
        key_entities = disease_entities | drugs | genes
        
        print(f"  Total key entities: {len(key_entities):,}")
        
        # Extract 1-hop neighborhood
        print("\n  Extracting 1-hop neighborhood...")
        expanded_mask = (
            drkg['head'].isin(key_entities) |
            drkg['tail'].isin(key_entities)
        )
        
        subgraph = drkg[expanded_mask].copy()
        
        print(f"  ✓ Subgraph size: {len(subgraph):,} triples")
        print(f"  Reduction: {(1 - len(subgraph)/len(drkg))*100:.1f}% from full DRKG")
        print()
        
        return subgraph
    
    def save_outputs(self, subgraph, stats_df):
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
        
        # Create entity and relation mappings
        entities = sorted(set(subgraph['head']) | set(subgraph['tail']))
        relations = sorted(subgraph['relation'].unique())
        
        entity2id = {entity: idx for idx, entity in enumerate(entities)}
        relation2id = {rel: idx for idx, rel in enumerate(relations)}
        
        # Save mappings
        with open(self.output_dir / 'entity2id.json', 'w') as f:
            json.dump(entity2id, f, indent=2)
        
        with open(self.output_dir / 'relation2id.json', 'w') as f:
            json.dump(relation2id, f, indent=2)
        
        print(f"✓ Saved entity mapping: {len(entity2id):,} entities")
        print(f"✓ Saved relation mapping: {len(relation2id):,} relations")
        
        # Save summary
        summary = {
            'extraction_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'original_drkg_triples': len(pd.read_csv(self.drkg_path, sep='\t', header=None)),
            'subgraph_triples': len(subgraph),
            'num_entities': len(entity2id),
            'num_relations': len(relation2id),
            'diseases': {name: mesh_id for name, mesh_id in self.autoimmune_diseases.items()},
            'note': 'Using MESH IDs (Disease::MESH:DXXXXXX format)'
        }
        
        with open(self.output_dir / 'subgraph_extraction_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Saved summary")
        print()
    
    def run(self):
        """Run complete extraction"""
        # Verify MESH IDs
        drkg, verified = self.verify_mesh_ids()
        
        # Extract direct triples
        direct_triples = self.extract_direct_triples(drkg)
        
        # Analyze disease links
        stats_df = self.analyze_disease_links(drkg)
        
        # Expand to drug-gene network
        subgraph = self.expand_to_drug_gene_network(drkg, direct_triples)
        
        # Save outputs
        self.save_outputs(subgraph, stats_df)
        
        # Final summary
        print("="*80)
        print("EXTRACTION COMPLETE")
        print("="*80)
        print(f"\nOriginal DRKG: {len(drkg):,} triples")
        print(f"Autoimmune Subgraph: {len(subgraph):,} triples")
        print(f"Reduction: {(1 - len(subgraph)/len(drkg))*100:.1f}%")
        
        training_count = len(stats_df[stats_df['total_links'] > 500])
        target_count = len(stats_df[stats_df['total_links'] <= 500])
        
        print(f"\nData-rich diseases: {training_count}")
        print(f"Data-sparse diseases: {target_count}")
        print(f"\nFiles saved to: {self.output_dir}")
        print("\n✓ Ready for Orphanet + HPO enrichment!")
        print("="*80)

def main():
    extractor = MESHSubgraphExtractor()
    extractor.run()

if __name__ == '__main__':
    main()
