"""
Dataset Verification Script
============================

Verifies all downloaded datasets are complete and valid:
- DRKG: Check triple count
- Orphanet: Verify XML structure
- HPO: Check file formats and sizes

Run this BEFORE processing to ensure data integrity.

Author: Yatin
Date: January 30, 2026
"""

import os
from pathlib import Path
import json

class DatasetVerifier:
    """Verify all downloaded datasets"""
    
    def __init__(self, base_dir='datasets/raw'):
        self.base_dir = Path(base_dir)
        self.results = {
            'drkg': {},
            'orphanet': {},
            'hpo': {}
        }
    
    def verify_drkg(self):
        """Verify DRKG dataset"""
        print("="*80)
        print("VERIFYING DRKG")
        print("="*80)
        
        drkg_dir = self.base_dir / 'drkg'
        
        # Check drkg.tsv
        drkg_tsv = drkg_dir / 'drkg.tsv'
        if not drkg_tsv.exists():
            print("✗ drkg.tsv not found!")
            self.results['drkg']['status'] = 'FAILED'
            self.results['drkg']['error'] = 'File not found'
            return False
        
        print(f"✓ Found: {drkg_tsv}")
        
        # Check file size
        size_gb = drkg_tsv.stat().st_size / (1024**3)
        print(f"  Size: {size_gb:.2f} GB")
        
        if size_gb < 0.5:  # Should be ~1.2 GB
            print(f"  ⚠ Warning: File seems small (expected ~1.2 GB)")
        
        # Count lines (triples)
        print("\nCounting triples (this may take 30-60 seconds)...")
        try:
            with open(drkg_tsv, 'r', encoding='utf-8') as f:
                num_lines = sum(1 for _ in f)
            
            print(f"✓ Total triples: {num_lines:,}")
            
            # Verify count
            expected = 5_874_261
            if num_lines == expected:
                print(f"✓ Perfect! Matches expected count ({expected:,})")
                status = 'PERFECT'
            elif num_lines > 5_000_000:
                print(f"✓ Good! Within acceptable range (>5M)")
                status = 'GOOD'
            else:
                print(f"✗ Too few triples (expected {expected:,})")
                status = 'INCOMPLETE'
            
            # Check format (sample first 5 lines)
            print("\nChecking format...")
            with open(drkg_tsv, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:
                        break
                    parts = line.strip().split('\t')
                    if len(parts) != 3:
                        print(f"✗ Line {i+1} has {len(parts)} columns (expected 3)")
                        status = 'FORMAT_ERROR'
                        break
                else:
                    print("✓ Format verified (3 columns: head, relation, tail)")
            
            # Sample data
            print("\nSample triples:")
            with open(drkg_tsv, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 3:
                        break
                    parts = line.strip().split('\t')
                    print(f"  {i+1}. {parts[0][:40]:40s} → {parts[1][:30]:30s} → {parts[2][:40]:40s}")
            
            self.results['drkg'] = {
                'status': status,
                'file': str(drkg_tsv),
                'size_gb': size_gb,
                'num_triples': num_lines
            }
            
            return status in ['PERFECT', 'GOOD']
            
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            self.results['drkg'] = {
                'status': 'ERROR',
                'error': str(e)
            }
            return False
    
    def verify_orphanet(self):
        """Verify Orphanet files"""
        print("\n" + "="*80)
        print("VERIFYING ORPHANET")
        print("="*80)
        
        orphanet_dir = self.base_dir / 'orphanet'
        
        files_to_check = {
            'en_product6.xml': {
                'description': 'Disease-Gene Associations',
                'min_size_mb': 15,
                'expected_size_mb': 21
            },
            'en_product4.xml': {
                'description': 'Disease-Phenotype Associations',
                'min_size_mb': 35,
                'expected_size_mb': 46
            },
            'en_product1.xml': {
                'description': 'Disease Nomenclature',
                'min_size_mb': 8,
                'expected_size_mb': 17
            }
        }
        
        all_good = True
        
        for filename, info in files_to_check.items():
            print(f"\n--- {info['description']} ---")
            filepath = orphanet_dir / filename
            
            if not filepath.exists():
                print(f"✗ {filename} not found!")
                self.results['orphanet'][filename] = {'status': 'NOT_FOUND'}
                all_good = False
                continue
            
            print(f"✓ Found: {filename}")
            
            # Check size
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"  Size: {size_mb:.2f} MB")
            
            if size_mb < info['min_size_mb']:
                print(f"  ✗ File too small (expected >{info['min_size_mb']} MB)")
                self.results['orphanet'][filename] = {
                    'status': 'INCOMPLETE',
                    'size_mb': size_mb
                }
                all_good = False
                continue
            elif size_mb < info['expected_size_mb'] * 0.9:
                print(f"  ⚠ Warning: Smaller than expected (~{info['expected_size_mb']} MB)")
            
            # Verify XML structure
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(filepath)
                root = tree.getroot()
                
                print(f"  ✓ Valid XML")
                print(f"  Root tag: {root.tag}")
                
                # Count disorders
                disorders = root.findall('.//Disorder')
                print(f"  Disorders: {len(disorders):,}")
                
                self.results['orphanet'][filename] = {
                    'status': 'VALID',
                    'size_mb': size_mb,
                    'num_disorders': len(disorders)
                }
                
            except ET.ParseError as e:
                print(f"  ✗ Invalid XML: {e}")
                self.results['orphanet'][filename] = {
                    'status': 'INVALID_XML',
                    'error': str(e)
                }
                all_good = False
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results['orphanet'][filename] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                all_good = False
        
        return all_good
    
    def verify_hpo(self):
        """Verify HPO files"""
        print("\n" + "="*80)
        print("VERIFYING HPO")
        print("="*80)
        
        hpo_dir = self.base_dir / 'hpo'
        
        files_to_check = {
            'phenotype.hpoa': {
                'description': 'Phenotype Annotations',
                'min_size_mb': 8,
                'expected_lines': 200000
            },
            'hp.obo': {
                'description': 'HPO Ontology',
                'min_size_mb': 5,
                'expected_lines': 100000
            },
            'genes_to_phenotype.txt': {
                'description': 'Gene-Phenotype Mapping',
                'min_size_mb': 2,
                'expected_lines': 50000
            }
        }
        
        all_good = True
        
        for filename, info in files_to_check.items():
            print(f"\n--- {info['description']} ---")
            filepath = hpo_dir / filename
            
            if not filepath.exists():
                print(f"✗ {filename} not found!")
                self.results['hpo'][filename] = {'status': 'NOT_FOUND'}
                all_good = False
                continue
            
            print(f"✓ Found: {filename}")
            
            # Check size
            size_mb = filepath.stat().st_size / (1024**2)
            print(f"  Size: {size_mb:.2f} MB")
            
            if size_mb < info['min_size_mb']:
                print(f"  ✗ File too small (expected >{info['min_size_mb']} MB)")
                self.results['hpo'][filename] = {
                    'status': 'INCOMPLETE',
                    'size_mb': size_mb
                }
                all_good = False
                continue
            
            # Count lines
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    num_lines = sum(1 for _ in f)
                
                print(f"  Lines: {num_lines:,}")
                
                if num_lines < info['expected_lines'] * 0.5:
                    print(f"  ⚠ Warning: Fewer lines than expected (~{info['expected_lines']:,})")
                
                # Show sample
                print(f"  Sample (first 3 non-comment lines):")
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    count = 0
                    for line in f:
                        if not line.startswith('#'):
                            print(f"    {line.strip()[:100]}")
                            count += 1
                            if count >= 3:
                                break
                
                self.results['hpo'][filename] = {
                    'status': 'VALID',
                    'size_mb': size_mb,
                    'num_lines': num_lines
                }
                
            except Exception as e:
                print(f"  ✗ Error: {e}")
                self.results['hpo'][filename] = {
                    'status': 'ERROR',
                    'error': str(e)
                }
                all_good = False
        
        return all_good
    
    def generate_report(self):
        """Generate verification report"""
        print("\n" + "="*80)
        print("VERIFICATION SUMMARY")
        print("="*80)
        
        # DRKG
        drkg_status = self.results['drkg'].get('status', 'UNKNOWN')
        print(f"\n✓ DRKG: {drkg_status}")
        if drkg_status in ['PERFECT', 'GOOD']:
            print(f"  Triples: {self.results['drkg']['num_triples']:,}")
            print(f"  Size: {self.results['drkg']['size_gb']:.2f} GB")
        
        # Orphanet
        print(f"\n✓ ORPHANET:")
        orphanet_valid = sum(1 for v in self.results['orphanet'].values() 
                            if v.get('status') == 'VALID')
        orphanet_total = len(self.results['orphanet'])
        print(f"  Files: {orphanet_valid}/{orphanet_total} valid")
        
        for filename, result in self.results['orphanet'].items():
            status = result.get('status', 'UNKNOWN')
            print(f"    {filename}: {status}")
        
        # HPO
        print(f"\n✓ HPO:")
        hpo_valid = sum(1 for v in self.results['hpo'].values() 
                       if v.get('status') == 'VALID')
        hpo_total = len(self.results['hpo'])
        print(f"  Files: {hpo_valid}/{hpo_total} valid")
        
        for filename, result in self.results['hpo'].items():
            status = result.get('status', 'UNKNOWN')
            print(f"    {filename}: {status}")
        
        # Overall
        all_good = (drkg_status in ['PERFECT', 'GOOD'] and 
                   orphanet_valid == orphanet_total and 
                   hpo_valid == hpo_total)
        
        print("\n" + "="*80)
        if all_good:
            print("✓✓✓ ALL DATASETS VERIFIED SUCCESSFULLY ✓✓✓")
            print("\nYou're ready to proceed with data processing!")
        else:
            print("⚠ SOME ISSUES DETECTED - REVIEW ABOVE")
            print("\nFix issues before proceeding with processing.")
        print("="*80)
        
        # Save report
        report_file = self.base_dir / 'verification_report.json'
        with open(report_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n✓ Detailed report saved: {report_file}")
        
        return all_good
    
    def run(self):
        """Run complete verification"""
        print("\n" + "="*80)
        print("DATASET VERIFICATION SCRIPT")
        print("="*80)
        print("\nThis script will verify:")
        print("  1. DRKG (drkg.tsv)")
        print("  2. Orphanet (3 XML files)")
        print("  3. HPO (3 files)")
        print()
        
        # Verify each dataset
        drkg_ok = self.verify_drkg()
        orphanet_ok = self.verify_orphanet()
        hpo_ok = self.verify_hpo()
        
        # Generate report
        all_ok = self.generate_report()
        
        return all_ok

def main():
    """Main entry point"""
    verifier = DatasetVerifier(base_dir='datasets/raw')
    success = verifier.run()
    
    return 0 if success else 1

if __name__ == '__main__':
    import sys
    sys.exit(main())
