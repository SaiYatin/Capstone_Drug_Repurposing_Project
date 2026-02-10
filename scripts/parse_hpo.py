"""
HPO (Human Phenotype Ontology) Parser
======================================

Parses HPO files to extract:
1. Disease-phenotype annotations (phenotype.hpoa)
2. Gene-phenotype mappings (genes_to_phenotype.txt)

Converts to triple format compatible with DRKG.

Author: Yatin
Date: February 9, 2026
"""

import pandas as pd
from pathlib import Path
from tqdm import tqdm

class HPOParser:
    """Parse HPO annotation files"""
    
    def __init__(self, hpo_dir='datasets/raw/hpo'):
        self.hpo_dir = Path(hpo_dir)
        self.output_dir = Path('datasets/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("HPO PARSER")
        print("="*80)
        print()
    
    def parse_phenotype_annotations(self, file='phenotype.hpoa'):
        """Parse disease-phenotype annotations"""
        print(f"Parsing phenotype annotations from {file}...")
        
        filepath = self.hpo_dir / file
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        # Read HPO annotation file (skip comment lines starting with #)
        try:
            df = pd.read_csv(
                filepath,
                sep='\t',
                comment='#',
                header=None,
                names=['database_id', 'disease_name', 'qualifier', 'hpo_id',
                       'reference', 'evidence', 'onset', 'frequency', 'sex',
                       'modifier', 'aspect', 'biocuration'],
                low_memory=False
            )
            
            print(f"  ✓ Loaded {len(df):,} annotations")
            
            # Convert to triple format
            triples = []
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
                # Create disease ID
                disease_id = f"Disease::{row['database_id']}"
                
                # Create phenotype ID
                phenotype_id = f"Phenotype::{row['hpo_id']}"
                
                # Create relation with frequency/evidence
                if pd.notna(row['frequency']) and row['frequency']:
                    relation = f"HPO::has_phenotype_freq_{row['frequency']}::Disease:Phenotype"
                else:
                    relation = "HPO::has_phenotype::Disease:Phenotype"
                
                triples.append({
                    'head': disease_id,
                    'relation': relation,
                    'tail': phenotype_id
                })
            
            triple_df = pd.DataFrame(triples)
            
            print(f"  ✓ Created {len(triple_df):,} disease-phenotype triples")
            
            # Show sample
            print("\n  Sample triples:")
            for i, row in triple_df.head(3).iterrows():
                print(f"    {row['head'][:40]:40s} → {row['tail'][:40]:40s}")
            
            # Unique diseases
            unique_diseases = triple_df['head'].nunique()
            print(f"\n  Unique diseases: {unique_diseases:,}")
            
            print()
            return triple_df
            
        except Exception as e:
            print(f"  ✗ Error parsing file: {e}")
            return pd.DataFrame()
    
    def parse_genes_to_phenotype(self, file='genes_to_phenotype.txt'):
        """Parse gene-phenotype mappings"""
        print(f"Parsing gene-phenotype mappings from {file}...")
        
        filepath = self.hpo_dir / file
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        try:
            # Read file - HPO files have headers after comment lines
            # First, read to find where actual data starts
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Find first non-comment line
            data_start = 0
            for i, line in enumerate(lines):
                if not line.startswith('#'):
                    data_start = i
                    break
            
            # Read from data start
            df = pd.read_csv(
                filepath,
                sep='\t',
                skiprows=data_start,  # Skip to first data line (which has headers)
                low_memory=False
            )
            
            print(f"  ✓ Loaded {len(df):,} gene-phenotype mappings")
            print(f"  Columns: {list(df.columns)}")
            
            # Convert to triple format
            # Actual HPO genes_to_phenotype format uses these columns
            triples = []
            
            # The file should have columns like: entrez-gene-id, entrez-gene-symbol, HPO-Term-ID, HPO-Term-Name
            # But column names vary, so we'll use positional
            for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
                # Use first column for gene ID, second for symbol
                # Third for HPO term
                try:
                    # Get gene symbol (usually column 1)
                    gene_col = df.columns[1] if len(df.columns) > 1 else df.columns[0]
                    hpo_col = df.columns[2] if len(df.columns) > 2 else df.columns[0]
                    
                    if pd.notna(row[gene_col]) and str(row[gene_col]).strip():
                        gene_id = f"Gene::{row[gene_col]}"
                    else:
                        continue
                    
                    # Create phenotype ID
                    if pd.notna(row[hpo_col]) and str(row[hpo_col]).strip():
                        phenotype_id = f"Phenotype::{row[hpo_col]}"
                    else:
                        continue
                    
                    triples.append({
                        'head': gene_id,
                        'relation': 'HPO::gene_associated_with_phenotype::Gene:Phenotype',
                        'tail': phenotype_id
                    })
                except Exception as e:
                    # Skip malformed rows
                    continue
            
            triple_df = pd.DataFrame(triples)
            
            print(f"  ✓ Created {len(triple_df):,} gene-phenotype triples")
            
            # Show sample
            print("\n  Sample triples:")
            for i, row in triple_df.head(3).iterrows():
                print(f"    {row['head'][:40]:40s} → {row['tail'][:40]:40s}")
            
            # Unique genes and phenotypes
            unique_genes = triple_df['head'].nunique()
            unique_phenotypes = triple_df['tail'].nunique()
            print(f"\n  Unique genes: {unique_genes:,}")
            print(f"  Unique phenotypes: {unique_phenotypes:,}")
            
            print()
            return triple_df
            
        except Exception as e:
            print(f"  ✗ Error parsing file: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def save_outputs(self, phenotype_df, gene_pheno_df):
        """Save parsed data"""
        print("Saving outputs...")
        
        # Save disease-phenotype
        if len(phenotype_df) > 0:
            output_file = self.output_dir / 'hpo_disease_phenotype.tsv'
            phenotype_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"✓ Saved disease-phenotype: {output_file}")
            print(f"  {len(phenotype_df):,} triples")
        
        # Save gene-phenotype
        if len(gene_pheno_df) > 0:
            output_file = self.output_dir / 'hpo_gene_phenotype.tsv'
            gene_pheno_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"✓ Saved gene-phenotype: {output_file}")
            print(f"  {len(gene_pheno_df):,} triples")
        
        # Summary
        total_triples = len(phenotype_df) + len(gene_pheno_df)
        print(f"\nTotal HPO triples: {total_triples:,}")
        print()
    
    def run(self):
        """Run complete parsing"""
        # Parse phenotype annotations
        phenotype_df = self.parse_phenotype_annotations('phenotype.hpoa')
        
        # Parse gene-phenotype mappings
        gene_pheno_df = self.parse_genes_to_phenotype('genes_to_phenotype.txt')
        
        # Save outputs
        self.save_outputs(phenotype_df, gene_pheno_df)
        
        # Final summary
        print("="*80)
        print("HPO PARSING COMPLETE")
        print("="*80)
        print("\nFiles created:")
        print("  • hpo_disease_phenotype.tsv")
        print("  • hpo_gene_phenotype.tsv")
        print("\nNext: Merge all datasets (DRKG + Orphanet + HPO)")
        print("="*80)

def main():
    parser = HPOParser()
    parser.run()

if __name__ == '__main__':
    main()