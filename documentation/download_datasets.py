"""
Dataset Download Script for Capstone Project
BioPathNet++ for Autoimmune Drug Repurposing

This script downloads:
1. DRKG (Drug Repurposing Knowledge Graph)
2. Orphanet rare disease data
3. HPO (Human Phenotype Ontology)

Run this script to prepare datasets for Feb 3rd review.
"""

import os
import requests
import urllib.request
from pathlib import Path

# Create data directory
data_dir = Path('datasets')
data_dir.mkdir(exist_ok=True)

print("="*80)
print("DATASET DOWNLOAD SCRIPT")
print("="*80)

# ============================================================================
# 1. DRKG (Drug Repurposing Knowledge Graph)
# ============================================================================
print("\n[1/3] Downloading DRKG...")
print("-" * 80)

drkg_url = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
drkg_file = data_dir / "drkg.tar.gz"

print(f"Source: {drkg_url}")
print(f"Destination: {drkg_file}")
print("Size: ~300 MB")
print("\nDownloading...")

try:
    urllib.request.urlretrieve(drkg_url, drkg_file)
    print(f"✓ Downloaded: {drkg_file}")
    
    # Extract
    import tarfile
    with tarfile.open(drkg_file, 'r:gz') as tar:
        tar.extractall(path=data_dir)
    print("✓ Extracted DRKG")
    
except Exception as e:
    print(f"✗ Error downloading DRKG: {e}")
    print("Manual download: https://github.com/gnn4dr/DRKG")

# ============================================================================
# 2. Orphanet Data
# ============================================================================
print("\n[2/3] Downloading Orphanet...")
print("-" * 80)

orphanet_files = {
    'disease_gene': 'http://www.orphadata.org/data/xml/en_product6.xml',
    'disease_phenotype': 'http://www.orphadata.org/data/xml/en_product4.xml',
    'nomenclature': 'http://www.orphadata.org/data/xml/en_product1.xml'
}

for name, url in orphanet_files.items():
    filename = data_dir / f'orphanet_{name}.xml'
    print(f"\nDownloading {name}...")
    print(f"Source: {url}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded: {filename}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Manual download: {url}")

# ============================================================================
# 3. HPO (Human Phenotype Ontology)
# ============================================================================
print("\n[3/3] Downloading HPO...")
print("-" * 80)

hpo_files = {
    'phenotype_annotations': 'http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa',
    'ontology': 'http://purl.obolibrary.org/obo/hp.obo',
    'genes_to_phenotype': 'http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt'
}

for name, url in hpo_files.items():
    ext = url.split('.')[-1]
    filename = data_dir / f'hpo_{name}.{ext}'
    print(f"\nDownloading {name}...")
    print(f"Source: {url}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✓ Downloaded: {filename}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print(f"Manual download: {url}")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("DOWNLOAD SUMMARY")
print("="*80)

print("\nExpected files in", data_dir, ":")
print("  DRKG:")
print("    - drkg.tsv (~300 MB)")
print("  Orphanet:")
print("    - orphanet_disease_gene.xml (~50 MB)")
print("    - orphanet_disease_phenotype.xml (~30 MB)")
print("    - orphanet_nomenclature.xml (~20 MB)")
print("  HPO:")
print("    - hpo_phenotype_annotations.hpoa (~15 MB)")
print("    - hpo_ontology.obo (~10 MB)")
print("    - hpo_genes_to_phenotype.txt (~5 MB)")

print("\nNext steps:")
print("1. Verify all files downloaded successfully")
print("2. Run data_exploration.py to analyze datasets")
print("3. Prepare visualizations for Feb 3rd review")

print("\n" + "="*80)
print("DOWNLOAD COMPLETE")
print("="*80)
