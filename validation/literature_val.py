"""
Literature Validation via PubMed
==================================

Validates top drug predictions by querying PubMed for supporting evidence.
Classifies each prediction as:
  - STRONG: ≥3 papers support the drug-disease association
  - MODERATE: 1-2 papers
  - WEAK: 0 papers (novel prediction)

Usage:
    python validation/literature_val.py

Author: Yatin
Date: April 2026
"""

import sys
import io
from pathlib import Path

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

from models.disease_mappings import (
    get_disease_name, get_drug_name, get_disease_short_name,
    TEST_DISEASE_ENTITIES,
)


# ============================================================================
# PubMed Search
# ============================================================================

PUBMED_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

# Rate limiting: max 3 requests/sec without API key, 10/sec with
REQUEST_DELAY = 0.4  # seconds between requests


def search_pubmed(query, max_results=20):
    """
    Search PubMed for a query string. Returns count and PMIDs.

    Args:
        query: PubMed search query string
        max_results: max PMIDs to return

    Returns:
        dict with 'count', 'pmids', 'query'
    """
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'xml',
    }

    url = f"{PUBMED_SEARCH_URL}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            xml_data = response.read().decode('utf-8')

        root = ET.fromstring(xml_data)
        count = int(root.findtext('.//Count', default='0'))
        pmids = [id_elem.text for id_elem in root.findall('.//IdList/Id')]

        return {
            'count': count,
            'pmids': pmids[:max_results],
            'query': query,
        }
    except Exception as e:
        print(f"    ⚠ PubMed search failed for '{query}': {e}")
        return {'count': 0, 'pmids': [], 'query': query}


def classify_evidence(paper_count):
    """Classify evidence level based on number of supporting papers."""
    if paper_count >= 3:
        return "STRONG"
    elif paper_count >= 1:
        return "MODERATE"
    else:
        return "WEAK"


# ============================================================================
# Drug Name Lookup (for PubMed queries)
# ============================================================================

# Extended drug name lookup - tries to extract meaningful names
# from compound IDs for better PubMed search
MESH_COMPOUND_NAMES = {
    "MESH:C500947": "Iguratimod",
    "CHEBI:33567": "Cepharanthine",
    "MESH:D008775": "Methylprednisolone",
    "MESH:D003078": "Cocoa",
    "MESH:D006886": "Hydroxychloroquine",
    "MESH:D000068800": "Etanercept",
    "MESH:D000069285": "Infliximab",
    "MESH:D000069283": "Rituximab",
    "MESH:D000068879": "Adalimumab",
}


def get_searchable_drug_name(drug_entity):
    """
    Get a drug name suitable for PubMed search.
    Tries multiple strategies for best coverage.
    """
    # First try the standard lookup
    name = get_drug_name(drug_entity)

    # If we got back the raw ID, try MESH compound lookup
    drug_id = drug_entity.split("::", 1)[1] if "::" in drug_entity else drug_entity
    if name == drug_id and drug_id in MESH_COMPOUND_NAMES:
        name = MESH_COMPOUND_NAMES[drug_id]

    # If it's still a DB/MESH/CHEBI ID, it won't give good PubMed results
    if name.startswith("DB") or name.startswith("MESH:") or name.startswith("CHEBI:"):
        return None  # Skip - won't find useful results

    return name


def get_searchable_disease_name(disease_entity):
    """Get disease name for PubMed search."""
    name = get_disease_name(disease_entity)
    # Clean up for PubMed
    name = name.replace("'", "'")  # Normalize quotes
    return name


# ============================================================================
# Main Validation Pipeline
# ============================================================================

def validate_predictions(predictions_dir='outputs/predictions', top_n=10, use_pubmed=True):
    """
    Validate top-N predictions per disease against PubMed literature.

    Args:
        predictions_dir: directory containing prediction CSVs
        top_n: number of top predictions to validate per disease
        use_pubmed: if True, query PubMed; if False, use local evidence only

    Returns:
        DataFrame with validation results
    """
    predictions_dir = Path(predictions_dir)

    all_results = []

    for disease_entity in TEST_DISEASE_ENTITIES:
        disease_name = get_disease_name(disease_entity)
        short_name = get_disease_short_name(disease_entity)
        csv_path = predictions_dir / f"{short_name}_top50.csv"

        if not csv_path.exists():
            print(f"  ⚠ Predictions not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        top_predictions = df.head(top_n)

        print(f"\n{'─' * 60}")
        print(f"Validating: {disease_name} (top {top_n} predictions)")
        print(f"{'─' * 60}")

        for _, row in top_predictions.iterrows():
            drug_name = get_searchable_drug_name(row['drug_entity'])
            result = {
                'disease_entity': disease_entity,
                'disease_name': disease_name,
                'drug_entity': row['drug_entity'],
                'drug_id': row['drug_id'],
                'drug_name': row.get('drug_name', drug_name or row['drug_id']),
                'rank': row['rank'],
                'score': row['score'],
                'known_drug': row.get('known_drug', False),
            }

            if use_pubmed and drug_name:
                # Query 1: Drug AND Disease
                query1 = f'"{drug_name}" AND "{get_searchable_disease_name(disease_entity)}"'
                pubmed1 = search_pubmed(query1)
                time.sleep(REQUEST_DELAY)

                # Query 2: Drug AND Disease AND (treatment OR therapy)
                query2 = f'"{drug_name}" AND "{get_searchable_disease_name(disease_entity)}" AND (treatment OR therapy OR repurposing)'
                pubmed2 = search_pubmed(query2)
                time.sleep(REQUEST_DELAY)

                result['pubmed_total_papers'] = pubmed1['count']
                result['pubmed_treatment_papers'] = pubmed2['count']
                result['pubmed_query'] = query1
                result['evidence_level'] = classify_evidence(pubmed1['count'])
                result['top_pmids'] = ','.join(pubmed1['pmids'][:5])

                icon = {"STRONG": "🟢", "MODERATE": "🟡", "WEAK": "🔴"}[result['evidence_level']]
                print(f"  {icon} Rank {row['rank']}: {drug_name or row['drug_id']} → "
                      f"{result['evidence_level']} ({pubmed1['count']} papers)")
            else:
                result['pubmed_total_papers'] = -1
                result['pubmed_treatment_papers'] = -1
                result['pubmed_query'] = ''
                result['evidence_level'] = 'SKIPPED'
                result['top_pmids'] = ''

                if drug_name:
                    print(f"  ⏭ Rank {row['rank']}: {drug_name} (PubMed disabled)")
                else:
                    print(f"  ⏭ Rank {row['rank']}: {row['drug_id']} (no searchable name)")

            all_results.append(result)

    return pd.DataFrame(all_results)


def compute_literature_summary(results_df):
    """Compute summary statistics from literature validation results."""
    # Filter to only those that were actually searched
    searched = results_df[results_df['evidence_level'] != 'SKIPPED']

    if len(searched) == 0:
        return {
            'total_validated': 0,
            'strong': 0,
            'moderate': 0,
            'weak': 0,
            'support_rate': 0.0,
        }

    summary = {
        'total_validated': len(searched),
        'strong': len(searched[searched['evidence_level'] == 'STRONG']),
        'moderate': len(searched[searched['evidence_level'] == 'MODERATE']),
        'weak': len(searched[searched['evidence_level'] == 'WEAK']),
    }
    summary['support_rate'] = (
        (summary['strong'] + summary['moderate']) / summary['total_validated'] * 100
    )

    # Per-disease summary
    per_disease = {}
    for disease in searched['disease_name'].unique():
        d_df = searched[searched['disease_name'] == disease]
        d_summary = {
            'validated': len(d_df),
            'strong': len(d_df[d_df['evidence_level'] == 'STRONG']),
            'moderate': len(d_df[d_df['evidence_level'] == 'MODERATE']),
            'weak': len(d_df[d_df['evidence_level'] == 'WEAK']),
        }
        d_summary['support_rate'] = (
            (d_summary['strong'] + d_summary['moderate']) / d_summary['validated'] * 100
            if d_summary['validated'] > 0 else 0
        )
        per_disease[disease] = d_summary

    summary['per_disease'] = per_disease

    return summary


def main():
    print("=" * 80)
    print("LITERATURE VALIDATION (PubMed)")
    print("=" * 80)

    # Check if predictions exist
    pred_dir = Path('outputs/predictions')
    if not pred_dir.exists():
        print("⚠ Predictions not found! Run predict_drugs.py first.")
        return

    # Run validation
    print("\nQuerying PubMed for top-10 predictions per disease...")
    print("(This may take a few minutes due to rate limiting)")

    results = validate_predictions(
        predictions_dir='outputs/predictions',
        top_n=10,
        use_pubmed=True,
    )

    # Compute summary
    summary = compute_literature_summary(results)

    # Print summary
    print(f"\n{'=' * 80}")
    print("LITERATURE VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  Total predictions validated: {summary['total_validated']}")
    print(f"  🟢 STRONG evidence (≥3 papers):  {summary['strong']}")
    print(f"  🟡 MODERATE evidence (1-2 papers): {summary['moderate']}")
    print(f"  🔴 WEAK evidence (0 papers):       {summary['weak']}")
    print(f"  Overall support rate: {summary['support_rate']:.1f}%")
    print(f"  (Target: ≥70%)")

    if 'per_disease' in summary:
        print(f"\n  Per-Disease Support Rates:")
        for disease, d_summary in summary['per_disease'].items():
            print(f"    {disease}: {d_summary['support_rate']:.1f}% "
                  f"({d_summary['strong']}S / {d_summary['moderate']}M / {d_summary['weak']}W)")

    # Save results
    output_dir = Path('outputs/validation')
    output_dir.mkdir(parents=True, exist_ok=True)

    results.to_csv(output_dir / 'literature_validation.csv', index=False)
    with open(output_dir / 'literature_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}")


if __name__ == '__main__':
    main()
