"""
Dataset Merger - Combine DRKG + Orphanet + HPO
===============================================

Merges:
1. DRKG autoimmune subgraph (1.2M triples)
2. Orphanet disease-phenotype (116K triples)
3. HPO disease-phenotype (281K triples)
4. HPO gene-phenotype (323K triples)

Creates enriched knowledge graph for autoimmune drug repurposing.

Author: Yatin
Date: February 9, 2026
"""

import pandas as pd
from pathlib import Path
import json
from collections import Counter

class DatasetMerger:
    """Merge all datasets into enriched knowledge graph"""
    
    def __init__(self, processed_dir='datasets/processed'):
        self.processed_dir = Path(processed_dir)
        
        print("="*80)
        print("DATASET MERGER")
        print("="*80)
        print()
    
    def load_all_datasets(self):
        """Load all processed datasets"""
        print("Loading all datasets...")
        
        datasets = {}
        
        # 1. DRKG autoimmune subgraph
        drkg_file = self.processed_dir / 'autoimmune_subgraph.tsv'
        if drkg_file.exists():
            datasets['drkg'] = pd.read_csv(
                drkg_file,
                sep='\t',
                header=None,
                names=['head', 'relation', 'tail']
            )
            print(f"  ✓ DRKG subgraph: {len(datasets['drkg']):,} triples")
        else:
            print(f"  ✗ DRKG subgraph not found: {drkg_file}")
        
        # 2. Orphanet disease-phenotype
        orphanet_file = self.processed_dir / 'orphanet_disease_phenotype.tsv'
        if orphanet_file.exists():
            datasets['orphanet'] = pd.read_csv(
                orphanet_file,
                sep='\t',
                header=None,
                names=['head', 'relation', 'tail']
            )
            print(f"  ✓ Orphanet phenotypes: {len(datasets['orphanet']):,} triples")
        else:
            print(f"  ⚠ Orphanet phenotypes not found (skipping)")
        
        # 3. HPO disease-phenotype
        hpo_disease_file = self.processed_dir / 'hpo_disease_phenotype.tsv'
        if hpo_disease_file.exists():
            datasets['hpo_disease'] = pd.read_csv(
                hpo_disease_file,
                sep='\t',
                header=None,
                names=['head', 'relation', 'tail']
            )
            print(f"  ✓ HPO disease-phenotype: {len(datasets['hpo_disease']):,} triples")
        else:
            print(f"  ⚠ HPO disease-phenotype not found (skipping)")
        
        # 4. HPO gene-phenotype
        hpo_gene_file = self.processed_dir / 'hpo_gene_phenotype.tsv'
        if hpo_gene_file.exists():
            datasets['hpo_gene'] = pd.read_csv(
                hpo_gene_file,
                sep='\t',
                header=None,
                names=['head', 'relation', 'tail']
            )
            print(f"  ✓ HPO gene-phenotype: {len(datasets['hpo_gene']):,} triples")
        else:
            print(f"  ⚠ HPO gene-phenotype not found (skipping)")
        
        print()
        return datasets
    
    def merge_datasets(self, datasets):
        """Merge all datasets"""
        print("Merging datasets...")
        
        # Combine all DataFrames
        all_dfs = list(datasets.values())
        
        print(f"  Combining {len(all_dfs)} datasets...")
        merged = pd.concat(all_dfs, ignore_index=True)
        
        print(f"  ✓ Combined: {len(merged):,} triples")
        
        # Check for duplicates
        print("\n  Checking for duplicates...")
        before = len(merged)
        merged = merged.drop_duplicates(subset=['head', 'relation', 'tail'])
        after = len(merged)
        removed = before - after
        
        if removed > 0:
            print(f"  ✓ Removed {removed:,} duplicate triples ({removed/before*100:.2f}%)")
        else:
            print(f"  ✓ No duplicates found")
        
        print(f"\n  Final enriched graph: {len(merged):,} triples")
        print()
        
        return merged
    
    def analyze_enrichment(self, datasets, merged):
        """Analyze enrichment statistics"""
        print("Analyzing enrichment...")
        
        # Original vs enriched
        drkg_size = len(datasets['drkg'])
        enriched_size = len(merged)
        
        added_triples = enriched_size - drkg_size
        percent_increase = (added_triples / drkg_size) * 100
        
        print(f"\n  Original DRKG subgraph: {drkg_size:,} triples")
        print(f"  Enriched graph: {enriched_size:,} triples")
        print(f"  Added: {added_triples:,} triples (+{percent_increase:.1f}%)")
        
        # Entity types
        print("\n  Analyzing entity types...")
        entity_types = Counter()
        
        for entity in set(merged['head']) | set(merged['tail']):
            entity_type = entity.split('::')[0] if '::' in entity else 'Unknown'
            entity_types[entity_type] += 1
        
        print("\n  Entity Type Distribution:")
        for entity_type, count in entity_types.most_common(15):
            print(f"    {entity_type:30s}: {count:>8,}")
        
        # Relation types
        print(f"\n  Total unique relations: {merged['relation'].nunique():,}")
        
        print()
        return {
            'original_size': drkg_size,
            'enriched_size': enriched_size,
            'added_triples': added_triples,
            'percent_increase': percent_increase,
            'entity_types': dict(entity_types)
        }
    
    def analyze_disease_enrichment(self, datasets, merged):
        """Analyze per-disease enrichment"""
        print("Analyzing per-disease enrichment...")
        
        # Load disease statistics
        stats_file = self.processed_dir / 'disease_statistics.csv'
        if not stats_file.exists():
            print("  ⚠ Disease statistics not found (skipping)")
            return {}
        
        stats_df = pd.read_csv(stats_file)
        
        print("\n  Disease-specific enrichment:")
        print(f"  {'Disease':<40s} {'Before':>10s} {'After':>10s} {'Added':>10s} {'Increase':>10s}")
        print("  " + "-"*76)
        
        enrichment_results = {}
        
        for _, row in stats_df.iterrows():
            disease_name = row['disease_name']
            mesh_entity = row['mesh_entity']
            before_count = row['total_links']
            
            # Count in enriched graph
            # Note: Orphanet uses ORPHA IDs, HPO uses OMIM/other IDs
            # For now, we'll focus on MESH entities from DRKG
            after_count = len(merged[
                (merged['head'] == mesh_entity) | (merged['tail'] == mesh_entity)
            ])
            
            added = after_count - before_count
            percent = (added / before_count * 100) if before_count > 0 else 0
            
            print(f"  {disease_name:<40s} {before_count:>10,} {after_count:>10,} {added:>10,} {percent:>9.1f}%")
            
            enrichment_results[disease_name] = {
                'before': before_count,
                'after': after_count,
                'added': added,
                'percent_increase': percent
            }
        
        print()
        return enrichment_results
    
    def create_mappings(self, merged):
        """Create entity and relation mappings for enriched graph"""
        print("Creating entity and relation mappings...")
        
        entities = sorted(set(merged['head']) | set(merged['tail']))
        relations = sorted(merged['relation'].unique())
        
        entity2id = {entity: idx for idx, entity in enumerate(entities)}
        relation2id = {rel: idx for idx, rel in enumerate(relations)}
        
        print(f"  ✓ Entities: {len(entity2id):,}")
        print(f"  ✓ Relations: {len(relation2id):,}")
        print()
        
        return entity2id, relation2id
    
    def save_outputs(self, merged, entity2id, relation2id, stats):
        """Save all outputs"""
        print("Saving outputs...")
        
        # Save enriched graph
        enriched_file = self.processed_dir / 'enriched_drkg.tsv'
        merged.to_csv(enriched_file, sep='\t', index=False, header=False)
        print(f"✓ Saved enriched graph: {enriched_file}")
        print(f"  {len(merged):,} triples")
        
        # Save mappings
        entity2id_file = self.processed_dir / 'enriched_entity2id.json'
        with open(entity2id_file, 'w') as f:
            json.dump(entity2id, f, indent=2)
        print(f"✓ Saved entity mapping: {entity2id_file}")
        
        relation2id_file = self.processed_dir / 'enriched_relation2id.json'
        with open(relation2id_file, 'w') as f:
            json.dump(relation2id, f, indent=2)
        print(f"✓ Saved relation mapping: {relation2id_file}")
        
        # Save statistics
        stats_file = self.processed_dir / 'enrichment_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Saved statistics: {stats_file}")
        
        print()
    
    def run(self):
        """Run complete merge pipeline"""
        # Load all datasets
        datasets = self.load_all_datasets()
        
        if 'drkg' not in datasets:
            print("✗ ERROR: DRKG subgraph not found. Run extraction first.")
            return False
        
        # Merge
        merged = self.merge_datasets(datasets)
        
        # Analyze enrichment
        overall_stats = self.analyze_enrichment(datasets, merged)
        disease_stats = self.analyze_disease_enrichment(datasets, merged)
        
        # Create mappings
        entity2id, relation2id = self.create_mappings(merged)
        
        # Combine stats
        all_stats = {
            'merge_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall': overall_stats,
            'per_disease': disease_stats,
            'num_entities': len(entity2id),
            'num_relations': len(relation2id)
        }
        
        # Save outputs
        self.save_outputs(merged, entity2id, relation2id, all_stats)
        
        # Final summary
        print("="*80)
        print("MERGE COMPLETE")
        print("="*80)
        print(f"\nOriginal DRKG subgraph: {overall_stats['original_size']:,} triples")
        print(f"Enriched graph: {overall_stats['enriched_size']:,} triples")
        print(f"Added: {overall_stats['added_triples']:,} triples (+{overall_stats['percent_increase']:.1f}%)")
        print(f"\nTotal entities: {len(entity2id):,}")
        print(f"Total relations: {len(relation2id):,}")
        print(f"\nEnriched graph saved to: {self.processed_dir / 'enriched_drkg.tsv'}")
        print("\n✓ Ready for model training!")
        print("="*80)
        
        return True

def main():
    merger = DatasetMerger()
    success = merger.run()
    return 0 if success else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
