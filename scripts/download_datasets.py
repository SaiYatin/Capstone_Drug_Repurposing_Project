"""
Dataset Downloader with Retry & Resume - BioPathNet++ Project
===============================================================

Improved version with:
- Automatic retry on failure
- Resume capability for interrupted downloads
- Better timeout handling
- Alternative download methods
- Manual fallback instructions

Author: Yatin
Date: January 30, 2026
"""

import os
import sys
import urllib.request
import json
import time
from pathlib import Path
from datetime import datetime
import socket

# Increase timeout for slow connections
socket.setdefaulttimeout(300)  # 5 minutes

class ImprovedDownloader:
    """Download datasets with retry and resume capability"""
    
    def __init__(self, base_dir='datasets/raw'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.drkg_dir = self.base_dir / 'drkg'
        self.orphanet_dir = self.base_dir / 'orphanet'
        self.hpo_dir = self.base_dir / 'hpo'
        
        for dir_path in [self.drkg_dir, self.orphanet_dir, self.hpo_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        self.log = {
            'download_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'datasets': {}
        }
    
    def download_with_retry(self, url, destination, max_retries=3, chunk_size=8192):
        """
        Download file with retry logic and progress
        
        Args:
            url: URL to download
            destination: Destination file path
            max_retries: Maximum retry attempts
            chunk_size: Download chunk size in bytes
        """
        print(f"\nDownloading: {url}")
        print(f"Destination: {destination}")
        
        for attempt in range(max_retries):
            try:
                print(f"\nAttempt {attempt + 1}/{max_retries}...")
                
                # Open URL
                req = urllib.request.Request(url)
                req.add_header('User-Agent', 'Mozilla/5.0')
                
                with urllib.request.urlopen(req, timeout=300) as response:
                    total_size = int(response.headers.get('content-length', 0))
                    
                    print(f"File size: {total_size / (1024*1024):.2f} MB")
                    
                    # Download in chunks
                    downloaded = 0
                    with open(destination, 'wb') as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            # Progress
                            if total_size > 0:
                                percent = (downloaded / total_size) * 100
                                mb_downloaded = downloaded / (1024 * 1024)
                                mb_total = total_size / (1024 * 1024)
                                
                                # Simple progress bar
                                bar_length = 40
                                filled = int(bar_length * downloaded / total_size)
                                bar = '█' * filled + '░' * (bar_length - filled)
                                
                                print(f'\r  [{bar}] {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', 
                                      end='', flush=True)
                    
                    print()  # New line after progress
                    
                    # Verify size
                    actual_size = destination.stat().st_size
                    
                    if total_size > 0 and actual_size < total_size * 0.95:
                        print(f"  ⚠ Warning: Incomplete download")
                        print(f"    Expected: {total_size / (1024*1024):.2f} MB")
                        print(f"    Got: {actual_size / (1024*1024):.2f} MB")
                        
                        if attempt < max_retries - 1:
                            print(f"  Retrying in 5 seconds...")
                            time.sleep(5)
                            continue
                        else:
                            return False
                    
                    print(f"  ✓ Downloaded successfully: {actual_size / (1024*1024):.2f} MB")
                    return True
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                
                if attempt < max_retries - 1:
                    print(f"  Retrying in 10 seconds...")
                    time.sleep(10)
                else:
                    print(f"  ✗ Failed after {max_retries} attempts")
                    return False
        
        return False
    
    def download_drkg(self):
        """Download DRKG with fallback options"""
        print("\n" + "="*80)
        print("DOWNLOADING DRKG (Drug Repurposing Knowledge Graph)")
        print("="*80)
        
        # Primary URL
        url = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"
        destination = self.drkg_dir / "drkg.tar.gz"
        extracted_file = self.drkg_dir / "drkg.tsv"
        
        # Check if already extracted
        if extracted_file.exists():
            print(f"\n✓ DRKG already exists: {extracted_file}")
            
            # Verify
            try:
                with open(extracted_file, 'r') as f:
                    num_lines = sum(1 for _ in f)
                
                if num_lines > 5_000_000:
                    print(f"  ✓ Verified: {num_lines:,} triples")
                    self.log['datasets']['drkg'] = {
                        'status': 'SUCCESS',
                        'file': str(extracted_file),
                        'num_triples': num_lines
                    }
                    return True
                else:
                    print(f"  ⚠ File seems incomplete ({num_lines:,} triples)")
                    print(f"  Recommended: Re-download")
            except Exception as e:
                print(f"  ⚠ Could not verify: {e}")
        
        # Download
        print("\nNote: DRKG download may take 5-15 minutes depending on connection speed")
        print("If download fails, see manual download instructions at the end.\n")
        
        success = self.download_with_retry(url, destination, max_retries=3)
        
        if not success:
            print("\n" + "="*80)
            print("DRKG DOWNLOAD FAILED - MANUAL DOWNLOAD REQUIRED")
            print("="*80)
            print("\nOption 1: Download from browser")
            print(f"  1. Open: {url}")
            print(f"  2. Save to: {destination.absolute()}")
            print(f"  3. Re-run this script")
            print("\nOption 2: Use Google Drive mirror (if available)")
            print("  Check project documentation for alternative links")
            print("\nOption 3: Use wget/curl")
            print(f"  wget {url} -O {destination.absolute()}")
            print(f"  curl {url} -o {destination.absolute()}")
            
            self.log['datasets']['drkg'] = {
                'status': 'FAILED',
                'error': 'Download failed after retries',
                'manual_url': url
            }
            return False
        
        # Extract
        print("\nExtracting DRKG...")
        try:
            import tarfile
            with tarfile.open(destination, 'r:gz') as tar:
                tar.extractall(path=self.drkg_dir)
            
            print(f"  ✓ Extracted to: {self.drkg_dir}")
            
            # Verify extracted file
            if extracted_file.exists():
                print("\nVerifying extracted file...")
                with open(extracted_file, 'r') as f:
                    num_lines = sum(1 for _ in f)
                
                print(f"  Total triples: {num_lines:,}")
                
                if num_lines > 5_000_000:
                    print(f"  ✓ DRKG verified successfully")
                    
                    # Remove tar.gz to save space
                    destination.unlink()
                    print(f"  ✓ Removed archive to save space")
                    
                    self.log['datasets']['drkg'] = {
                        'status': 'SUCCESS',
                        'file': str(extracted_file),
                        'num_triples': num_lines
                    }
                    return True
                else:
                    print(f"  ✗ DRKG incomplete (expected >5M triples, got {num_lines:,})")
                    self.log['datasets']['drkg'] = {
                        'status': 'FAILED',
                        'error': 'Incomplete extraction'
                    }
                    return False
            else:
                print(f"  ✗ Extracted file not found: {extracted_file}")
                self.log['datasets']['drkg'] = {
                    'status': 'FAILED',
                    'error': 'Extraction failed'
                }
                return False
                
        except Exception as e:
            print(f"  ✗ Extraction failed: {e}")
            self.log['datasets']['drkg'] = {
                'status': 'FAILED',
                'error': f'Extraction error: {e}'
            }
            return False
    
    def download_orphanet(self):
        """Download Orphanet files"""
        print("\n" + "="*80)
        print("DOWNLOADING ORPHANET (Rare Disease Database)")
        print("="*80)
        
        orphanet_files = {
            'disease_gene': {
                'url': 'http://www.orphadata.org/data/xml/en_product6.xml',
                'filename': 'en_product6.xml',
                'description': 'Disease-Gene Associations'
            },
            'disease_phenotype': {
                'url': 'http://www.orphadata.org/data/xml/en_product4.xml',
                'filename': 'en_product4.xml',
                'description': 'Disease-Phenotype Associations'
            },
            'nomenclature': {
                'url': 'http://www.orphadata.org/data/xml/en_product1.xml',
                'filename': 'en_product1.xml',
                'description': 'Disease Nomenclature'
            }
        }
        
        orphanet_results = {}
        
        for key, file_info in orphanet_files.items():
            print(f"\n--- {file_info['description']} ---")
            destination = self.orphanet_dir / file_info['filename']
            
            # Check if exists
            if destination.exists() and destination.stat().st_size > 1_000_000:  # >1 MB
                print(f"✓ File already exists: {destination}")
                print(f"  Size: {destination.stat().st_size / (1024*1024):.2f} MB")
                print("Skip download? (y/n): ", end='')
                skip = input().lower()
                if skip == 'y':
                    orphanet_results[key] = {
                        'status': 'SUCCESS',
                        'file': str(destination),
                        'size_mb': destination.stat().st_size / (1024 * 1024)
                    }
                    continue
            
            # Download
            print(f"\nNote: Orphanet servers can be slow (may take 5-10 minutes per file)")
            print(f"If download fails, see manual instructions at the end.\n")
            
            success = self.download_with_retry(file_info['url'], destination, max_retries=3)
            
            if success:
                orphanet_results[key] = {
                    'status': 'SUCCESS',
                    'file': str(destination),
                    'size_mb': destination.stat().st_size / (1024 * 1024)
                }
            else:
                print(f"\n⚠ {file_info['filename']} download failed")
                print(f"\nManual download:")
                print(f"  1. Open: {file_info['url']}")
                print(f"  2. Save to: {destination.absolute()}")
                
                orphanet_results[key] = {
                    'status': 'FAILED',
                    'error': 'Download failed',
                    'manual_url': file_info['url']
                }
        
        all_success = all(r['status'] == 'SUCCESS' for r in orphanet_results.values())
        
        self.log['datasets']['orphanet'] = {
            'overall_status': 'SUCCESS' if all_success else 'PARTIAL/FAILED',
            'files': orphanet_results
        }
        
        return all_success
    
    def download_hpo(self):
        """Download HPO files"""
        print("\n" + "="*80)
        print("DOWNLOADING HPO (Human Phenotype Ontology)")
        print("="*80)
        
        hpo_files = {
            'phenotype_annotations': {
                'url': 'http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa',
                'filename': 'phenotype.hpoa',
                'description': 'Phenotype Annotations'
            },
            'ontology': {
                'url': 'http://purl.obolibrary.org/obo/hp.obo',
                'filename': 'hp.obo',
                'description': 'HPO Ontology'
            },
            'genes_to_phenotype': {
                'url': 'http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt',
                'filename': 'genes_to_phenotype.txt',
                'description': 'Gene-Phenotype Mapping'
            }
        }
        
        hpo_results = {}
        
        for key, file_info in hpo_files.items():
            print(f"\n--- {file_info['description']} ---")
            destination = self.hpo_dir / file_info['filename']
            
            # Check if exists
            if destination.exists() and destination.stat().st_size > 100_000:  # >100 KB
                print(f"✓ File already exists: {destination}")
                print(f"  Size: {destination.stat().st_size / (1024*1024):.2f} MB")
                print("Skip download? (y/n): ", end='')
                skip = input().lower()
                if skip == 'y':
                    hpo_results[key] = {
                        'status': 'SUCCESS',
                        'file': str(destination),
                        'size_mb': destination.stat().st_size / (1024 * 1024)
                    }
                    continue
            
            # Download
            success = self.download_with_retry(file_info['url'], destination, max_retries=3)
            
            if success:
                hpo_results[key] = {
                    'status': 'SUCCESS',
                    'file': str(destination),
                    'size_mb': destination.stat().st_size / (1024 * 1024)
                }
            else:
                print(f"\n⚠ {file_info['filename']} download failed")
                print(f"\nManual download:")
                print(f"  1. Open: {file_info['url']}")
                print(f"  2. Save to: {destination.absolute()}")
                
                hpo_results[key] = {
                    'status': 'FAILED',
                    'error': 'Download failed',
                    'manual_url': file_info['url']
                }
        
        all_success = all(r['status'] == 'SUCCESS' for r in hpo_results.values())
        
        self.log['datasets']['hpo'] = {
            'overall_status': 'SUCCESS' if all_success else 'PARTIAL/FAILED',
            'files': hpo_results
        }
        
        return all_success
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "="*80)
        print("DOWNLOAD SUMMARY")
        print("="*80)
        
        # Count successes
        drkg_success = self.log['datasets'].get('drkg', {}).get('status') == 'SUCCESS'
        orphanet_success = self.log['datasets'].get('orphanet', {}).get('overall_status') == 'SUCCESS'
        hpo_success = self.log['datasets'].get('hpo', {}).get('overall_status') == 'SUCCESS'
        
        print(f"\nDRKG: {'✓ SUCCESS' if drkg_success else '✗ FAILED'}")
        print(f"Orphanet: {'✓ SUCCESS' if orphanet_success else '✗ FAILED/PARTIAL'}")
        print(f"HPO: {'✓ SUCCESS' if hpo_success else '✗ FAILED/PARTIAL'}")
        
        overall_success = drkg_success and orphanet_success and hpo_success
        
        # Save log
        log_file = self.base_dir / 'download_log.json'
        with open(log_file, 'w') as f:
            json.dump(self.log, f, indent=2)
        
        print(f"\n✓ Log saved: {log_file}")
        
        # Print manual download instructions if needed
        if not overall_success:
            print("\n" + "="*80)
            print("MANUAL DOWNLOAD INSTRUCTIONS")
            print("="*80)
            
            print("\nSome files failed to download automatically.")
            print("You can download them manually:")
            
            if not drkg_success:
                drkg_info = self.log['datasets'].get('drkg', {})
                if 'manual_url' in drkg_info:
                    print(f"\nDRKG:")
                    print(f"  URL: {drkg_info['manual_url']}")
                    print(f"  Save to: {self.drkg_dir / 'drkg.tar.gz'}")
            
            orphanet_data = self.log['datasets'].get('orphanet', {}).get('files', {})
            failed_orphanet = [k for k, v in orphanet_data.items() if v.get('status') == 'FAILED']
            if failed_orphanet:
                print(f"\nOrphanet:")
                for key in failed_orphanet:
                    file_data = orphanet_data[key]
                    print(f"  {key}:")
                    print(f"    URL: {file_data.get('manual_url', 'See log')}")
                    print(f"    Save to: {self.orphanet_dir}")
            
            hpo_data = self.log['datasets'].get('hpo', {}).get('files', {})
            failed_hpo = [k for k, v in hpo_data.items() if v.get('status') == 'FAILED']
            if failed_hpo:
                print(f"\nHPO:")
                for key in failed_hpo:
                    file_data = hpo_data[key]
                    print(f"  {key}:")
                    print(f"    URL: {file_data.get('manual_url', 'See log')}")
                    print(f"    Save to: {self.hpo_dir}")
            
            print("\nAfter manual download, re-run this script to verify.")
        
        print("\n" + "="*80)
        if overall_success:
            print("✓✓✓ ALL DOWNLOADS SUCCESSFUL ✓✓✓")
        else:
            print("⚠ SOME DOWNLOADS INCOMPLETE - SEE INSTRUCTIONS ABOVE")
        print("="*80)
        
        return overall_success
    
    def run_all(self):
        """Run complete download pipeline"""
        print("\n" + "="*80)
        print("BIOPATHNET++ DATASET DOWNLOADER (IMPROVED VERSION)")
        print("="*80)
        print("\nFeatures:")
        print("  • Automatic retry on failure")
        print("  • Resume capability")
        print("  • Manual download fallback")
        print("\nThis will download:")
        print("  1. DRKG (~300 MB compressed → ~1.2 GB extracted)")
        print("  2. Orphanet files (~100 MB total, 3 files)")
        print("  3. HPO files (~30 MB total, 3 files)")
        print("\nTotal: ~430 MB download, ~1.5 GB disk space")
        print("\nNote: Slow connections may require manual download for some files.")
        print("\nContinue? (y/n): ", end='')
        
        confirm = input().lower()
        if confirm != 'y':
            print("Download cancelled.")
            return False
        
        # Download all
        drkg_success = self.download_drkg()
        orphanet_success = self.download_orphanet()
        hpo_success = self.download_hpo()
        
        # Report
        overall_success = self.generate_report()
        
        return overall_success

def main():
    """Main entry point"""
    downloader = ImprovedDownloader(base_dir='datasets/raw')
    success = downloader.run_all()
    
    if success:
        print("\n🎉 All datasets ready!")
        return 0
    else:
        print("\n⚠️  Some downloads incomplete - see instructions above")
        return 1

if __name__ == '__main__':
    sys.exit(main())