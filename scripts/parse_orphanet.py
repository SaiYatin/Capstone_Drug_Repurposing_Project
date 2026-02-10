"""
Orphanet XML Parser
===================

Parses Orphanet XML files to extract:
1. Disease-Gene associations (en_product6.xml)
2. Disease-Phenotype associations (en_product4.xml)

Converts to triple format compatible with DRKG.

Author: Yatin
Date: February 9, 2026
"""

import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class OrphanetParser:
    """Parse Orphanet XML files"""
    
    def __init__(self, orphanet_dir='datasets/raw/orphanet'):
        self.orphanet_dir = Path(orphanet_dir)
        self.output_dir = Path('datasets/processed')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print("="*80)
        print("ORPHANET XML PARSER")
        print("="*80)
        print()
    
    def parse_disease_gene(self, xml_file='en_product6.xml'):
        """Parse disease-gene associations"""
        print(f"Parsing disease-gene associations from {xml_file}...")
        
        filepath = self.orphanet_dir / xml_file
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        # Parse XML
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        triples = []
        
        # Find all disorders
        disorders = root.findall('.//Disorder')
        print(f"  Found {len(disorders):,} disorders")
        
        for disorder in tqdm(disorders, desc="  Processing disorders"):
            # Get OrphaCode
            orpha_code = disorder.find('.//OrphaCode')
            if orpha_code is None or not orpha_code.text:
                continue
            
            disease_id = f'Disease::ORPHA:{orpha_code.text}'
            
            # Get associated genes
            gene_list = disorder.find('.//GeneList')
            if gene_list is not None:
                genes = gene_list.findall('.//Gene')
                
                for gene in genes:
                    # Get gene symbol
                    symbol = gene.find('.//Symbol')
                    if symbol is not None and symbol.text:
                        gene_id = f'Gene::{symbol.text}'
                        
                        # Get association type if available
                        assoc_type = gene.find('.//DisorderGeneAssociationType/Name')
                        relation_type = assoc_type.text if assoc_type is not None else 'associated_with'
                        
                        triples.append({
                            'head': disease_id,
                            'relation': f'Orphanet::{relation_type}::Disease:Gene',
                            'tail': gene_id
                        })
        
        # Create DataFrame
        df = pd.DataFrame(triples)
        
        if len(df) > 0:
            print(f"  ✓ Extracted {len(df):,} disease-gene associations")
            
            # Show sample
            print("\n  Sample associations:")
            for i, row in df.head(3).iterrows():
                print(f"    {row['head']:30s} → {row['relation']:50s} → {row['tail']}")
        else:
            print("  ✗ No associations found")
        
        print()
        return df
    
    def parse_disease_phenotype(self, xml_file='en_product4.xml'):
        """Parse disease-phenotype (HPO) associations"""
        print(f"Parsing disease-phenotype associations from {xml_file}...")
        
        filepath = self.orphanet_dir / xml_file
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return pd.DataFrame()
        
        # Parse XML
        tree = ET.parse(filepath)
        root = tree.getroot()
        
        triples = []
        
        # Find all disorders
        disorders = root.findall('.//Disorder')
        print(f"  Found {len(disorders):,} disorders")
        
        for disorder in tqdm(disorders, desc="  Processing disorders"):
            # Get OrphaCode
            orpha_code = disorder.find('.//OrphaCode')
            if orpha_code is None or not orpha_code.text:
                continue
            
            disease_id = f'Disease::ORPHA:{orpha_code.text}'
            
            # Get HPO associations
            hpo_list = disorder.find('.//HPODisorderAssociationList')
            if hpo_list is not None:
                hpo_assocs = hpo_list.findall('.//HPODisorderAssociation')
                
                for assoc in hpo_assocs:
                    hpo = assoc.find('.//HPO')
                    if hpo is not None:
                        # Get HPO ID
                        hpo_id_elem = hpo.find('.//HPOId')
                        if hpo_id_elem is not None and hpo_id_elem.text:
                            phenotype_id = f'Phenotype::{hpo_id_elem.text}'
                            
                            # Get frequency if available
                            freq = assoc.find('.//HPOFrequency/Name')
                            frequency = freq.text if freq is not None else 'unknown'
                            
                            triples.append({
                                'head': disease_id,
                                'relation': f'Orphanet::has_phenotype::Disease:Phenotype',
                                'tail': phenotype_id,
                                'frequency': frequency
                            })
        
        # Create DataFrame
        df = pd.DataFrame(triples)
        
        if len(df) > 0:
            print(f"  ✓ Extracted {len(df):,} disease-phenotype associations")
            
            # Show frequency distribution
            if 'frequency' in df.columns:
                print("\n  Frequency distribution:")
                freq_counts = df['frequency'].value_counts()
                for freq, count in freq_counts.head(5).items():
                    print(f"    {freq:30s}: {count:>8,}")
            
            # Show sample
            print("\n  Sample associations:")
            for i, row in df.head(3).iterrows():
                print(f"    {row['head']:30s} → {row['tail']:30s} ({row.get('frequency', 'unknown')})")
            
            # Drop frequency column for triple format
            df = df[['head', 'relation', 'tail']]
        else:
            print("  ✗ No associations found")
        
        print()
        return df
    
    def parse_additional_files(self):
        """Parse additional Orphanet files if present"""
        print("Checking for additional Orphanet files...")
        
        additional_files = [
            'en_product3_195.xml',  # Rare immunological diseases
            'en_product3_196.xml',  # Rare systemic/rheumatological diseases
        ]
        
        all_triples = []
        
        for filename in additional_files:
            filepath = self.orphanet_dir / filename
            
            if filepath.exists():
                print(f"\n  Found: {filename}")
                print("  Parsing disease associations...")
                
                try:
                    tree = ET.parse(filepath)
                    root = tree.getroot()
                    
                    disorders = root.findall('.//Disorder')
                    print(f"    Disorders: {len(disorders):,}")
                    
                    # Extract disease-gene associations
                    for disorder in tqdm(disorders, desc="    Processing"):
                        orpha_code = disorder.find('.//OrphaCode')
                        if orpha_code is None or not orpha_code.text:
                            continue
                        
                        disease_id = f'Disease::ORPHA:{orpha_code.text}'
                        
                        # Genes
                        gene_list = disorder.find('.//GeneList')
                        if gene_list is not None:
                            genes = gene_list.findall('.//Gene')
                            for gene in genes:
                                symbol = gene.find('.//Symbol')
                                if symbol is not None and symbol.text:
                                    all_triples.append({
                                        'head': disease_id,
                                        'relation': 'Orphanet::associated_with::Disease:Gene',
                                        'tail': f'Gene::{symbol.text}'
                                    })
                    
                    print(f"    ✓ Extracted {len([t for t in all_triples if filename in str(t)])} associations")
                    
                except Exception as e:
                    print(f"    ✗ Error parsing {filename}: {e}")
            else:
                print(f"  Not found: {filename} (skipping)")
        
        if all_triples:
            return pd.DataFrame(all_triples)
        else:
            return pd.DataFrame()
    
    def save_outputs(self, disease_gene_df, disease_phenotype_df, additional_df=None):
        """Save parsed data"""
        print("Saving outputs...")
        
        # Save disease-gene
        if len(disease_gene_df) > 0:
            output_file = self.output_dir / 'orphanet_disease_gene.tsv'
            disease_gene_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"✓ Saved disease-gene: {output_file}")
            print(f"  {len(disease_gene_df):,} triples")
        
        # Save disease-phenotype
        if len(disease_phenotype_df) > 0:
            output_file = self.output_dir / 'orphanet_disease_phenotype.tsv'
            disease_phenotype_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"✓ Saved disease-phenotype: {output_file}")
            print(f"  {len(disease_phenotype_df):,} triples")
        
        # Save additional
        if additional_df is not None and len(additional_df) > 0:
            output_file = self.output_dir / 'orphanet_additional.tsv'
            additional_df.to_csv(output_file, sep='\t', index=False, header=False)
            print(f"✓ Saved additional: {output_file}")
            print(f"  {len(additional_df):,} triples")
        
        # Summary
        total_triples = len(disease_gene_df) + len(disease_phenotype_df)
        if additional_df is not None:
            total_triples += len(additional_df)
        
        print(f"\nTotal Orphanet triples: {total_triples:,}")
        print()
    
    def run(self):
        """Run complete parsing"""
        # Parse main files
        disease_gene_df = self.parse_disease_gene('en_product6.xml')
        disease_phenotype_df = self.parse_disease_phenotype('en_product4.xml')
        
        # Parse additional files
        additional_df = self.parse_additional_files()
        
        # Save outputs
        self.save_outputs(disease_gene_df, disease_phenotype_df, additional_df)
        
        # Final summary
        print("="*80)
        print("ORPHANET PARSING COMPLETE")
        print("="*80)
        print("\nFiles created:")
        print("  • orphanet_disease_gene.tsv")
        print("  • orphanet_disease_phenotype.tsv")
        if additional_df is not None and len(additional_df) > 0:
            print("  • orphanet_additional.tsv")
        print("\nNext: Parse HPO files")
        print("="*80)

def main():
    parser = OrphanetParser()
    parser.run()

if __name__ == '__main__':
    main()
