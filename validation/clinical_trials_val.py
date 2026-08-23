import os
import glob
import pandas as pd
import requests
import time
import json
import logging
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DISEASE_SEARCH_NAMES = {
    'Sjögren\'s Syndrome': ['Sjogren syndrome', 'Sjogrens syndrome', 'Sjogren\'s disease'],
    'Systemic Sclerosis': ['systemic sclerosis', 'scleroderma'],
    'Primary Biliary Cholangitis': ['primary biliary cholangitis', 'primary biliary cirrhosis'],
    'Myasthenia Gravis': ['myasthenia gravis'],
    'Autoimmune Hepatitis': ['autoimmune hepatitis'],
    'Crohn\'s Disease': ['Crohn disease', 'Crohns disease'],
}

API_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
PREDICTIONS_DIR = r"c:\D_Games\capstone\implementations\outputs\predictions"
OUTPUT_DIR = r"c:\D_Games\capstone\implementations\outputs\validation_v2"
JSON_OUTPUT = os.path.join(OUTPUT_DIR, "clinical_trials_validation.json")
MD_OUTPUT = os.path.join(OUTPUT_DIR, "clinical_trials_validation.md")

def query_clinical_trials(drug: str, disease_terms: List[str]) -> Dict[str, Any]:
    # Construct query term
    disease_query = " OR ".join(f'"{term}"' for term in disease_terms)
    query_term = f'"{drug}" AND ({disease_query})'
    
    params = {
        'query.term': query_term,
        'filter.overallStatus': 'RECRUITING,ACTIVE_NOT_RECRUITING,COMPLETED',
        'pageSize': 100
    }
    
    try:
        response = requests.get(API_BASE_URL, params=params, timeout=10)
        response.raise_for_status()
        time.sleep(0.5)  # Rate limit
        return response.json()
    except Exception as e:
        logger.error(f"Error querying API for {drug} and {disease_terms}: {e}")
        time.sleep(1) # Extra wait on error
        return {}

def classify_trials(studies: List[Dict]) -> str:
    if not studies:
        return "NO_TRIAL"
    
    highest_phase = 0
    for study in studies:
        protocol = study.get("protocolSection", {})
        design = protocol.get("designModule", {})
        phases = design.get("phases", [])
        
        for phase in phases:
            if phase in ["PHASE2", "PHASE3", "PHASE4"]:
                return "STRONG_CLINICAL"
            elif phase in ["PHASE1", "EARLY_PHASE1"]:
                highest_phase = max(highest_phase, 1)
            elif phase == "NA": # observational or other
                pass
                
    if highest_phase >= 1 or len(studies) > 0:
        return "MODERATE_CLINICAL"
        
    return "NO_TRIAL"

def process_disease_predictions():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    results = {}
    summary = {
        "overall_match_rate": 0.0,
        "disease_match_rates": {},
        "total_pairs": 0,
        "matched_pairs": 0,
        "nct_ids": []
    }
    
    # Explicit filename mapping for tricky names
    FILENAME_MAP = {
        'sjogrens_top50.csv': "Sjögren's Syndrome",
        'sjogrens_syndrome_top50.csv': "Sjögren's Syndrome",
        'crohns_disease_top50.csv': "Crohn's Disease",
    }
    
    csv_files = glob.glob(os.path.join(PREDICTIONS_DIR, "*.csv"))
    if not csv_files:
        logger.warning(f"No CSV files found in {PREDICTIONS_DIR}")
        
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        
        # Try explicit map first
        matched_key = FILENAME_MAP.get(filename)
        
        if not matched_key:
            # Match against our dictionary keys (simple heuristic)
            for key in DISEASE_SEARCH_NAMES.keys():
                if key.lower().replace("'", "").replace("ö", "o").replace(" ", "") in filename.lower().replace("_", ""):
                    matched_key = key
                    break
                
        if not matched_key:
            matched_key = next((k for k in DISEASE_SEARCH_NAMES.keys() if k.lower() in filename.lower()), None)
            if not matched_key:
                logger.warning(f"Could not map filename {filename} to disease search terms. Skipping.")
                continue
                
        search_terms = DISEASE_SEARCH_NAMES[matched_key]
        logger.info(f"Processing {matched_key} from {filename}")
        
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
            if 'drug' not in df.columns:
                df.rename(columns={df.columns[0]: 'drug'}, inplace=True) # Assume first column is drug if not named
            
            top_drugs = df['drug'].head(20).tolist()
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
            
        disease_results = []
        disease_matched = 0
        
        for drug in top_drugs:
            data = query_clinical_trials(drug, search_terms)
            studies = data.get("studies", [])
            
            classification = classify_trials(studies)
            
            drug_nct_ids = []
            for s in studies:
                nct_id = s.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if nct_id:
                    drug_nct_ids.append(nct_id)
                    summary["nct_ids"].append(nct_id)
            
            if classification != "NO_TRIAL":
                disease_matched += 1
                summary["matched_pairs"] += 1
                
            disease_results.append({
                "drug": drug,
                "classification": classification,
                "trial_count": len(studies),
                "nct_ids": drug_nct_ids
            })
            summary["total_pairs"] += 1
            
        results[matched_key] = disease_results
        
        match_rate = disease_matched / len(top_drugs) if top_drugs else 0
        summary["disease_match_rates"][matched_key] = match_rate
        
    if summary["total_pairs"] > 0:
        summary["overall_match_rate"] = summary["matched_pairs"] / summary["total_pairs"]
        
    # Remove duplicate NCT IDs
    summary["nct_ids"] = list(set(summary["nct_ids"]))
    
    # Save JSON
    with open(JSON_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump({"results": results, "summary": summary}, f, indent=4)
        
    # Generate MD
    generate_markdown_report(results, summary)
    
def generate_markdown_report(results, summary):
    md = "# Clinical Trials Validation Report\n\n"
    
    md += "## Overall Summary\n"
    md += f"- **Total Predictions Analyzed**: {summary['total_pairs']}\n"
    md += f"- **Overall Match Rate**: {summary['overall_match_rate']:.2%}\n"
    md += f"- **Total Unique Clinical Trials Found**: {len(summary['nct_ids'])}\n\n"
    
    md += "## Match Rates per Disease\n"
    for disease, rate in summary['disease_match_rates'].items():
        md += f"- **{disease}**: {rate:.2%}\n"
        
    md += "\n## Detailed Results\n\n"
    
    for disease, drugs in results.items():
        md += f"### {disease}\n"
        md += "| Drug | Classification | Trial Count | NCT IDs |\n"
        md += "|---|---|---|---|\n"
        for d in drugs:
            nct_str = ", ".join(d['nct_ids'][:3]) + ("..." if len(d['nct_ids']) > 3 else "")
            if not nct_str:
                nct_str = "None"
            md += f"| {d['drug']} | {d['classification']} | {d['trial_count']} | {nct_str} |\n"
        md += "\n"
        
    with open(MD_OUTPUT, 'w', encoding='utf-8') as f:
        f.write(md)
    logger.info(f"Saved reports to {JSON_OUTPUT} and {MD_OUTPUT}")

if __name__ == "__main__":
    process_disease_predictions()
