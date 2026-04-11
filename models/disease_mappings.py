"""
Disease & Drug Mappings (v2 - Fixed)
=====================================

Maps entity IDs to human-readable names:
- Entrez Gene IDs → Gene Symbols (for pathway analysis)
- DrugBank IDs → Drug Names (for reporting)
- MESH IDs → Disease Names

Author: Yatin
Date: April 2026
"""

import json
import urllib.request
import urllib.parse
import time
from pathlib import Path


# ============================================================================
# MESH ID → Disease Name
# ============================================================================

MESH_TO_DISEASE = {
    "MESH:D001172": "Rheumatoid Arthritis",
    "MESH:D009103": "Multiple Sclerosis",
    "MESH:D008180": "Systemic Lupus Erythematosus",
    "MESH:D012859": "Sjögren's Syndrome",
    "MESH:D012595": "Systemic Sclerosis",
    "MESH:D008105": "Primary Biliary Cholangitis",
    "MESH:D009157": "Myasthenia Gravis",
    "MESH:D019693": "Autoimmune Hepatitis",
    "MESH:D003424": "Crohn's Disease",
    "MESH:D012162": "Retinal Detachment",
}

DISEASE_SHORT_NAMES = {
    "MESH:D001172": "rheumatoid_arthritis",
    "MESH:D009103": "multiple_sclerosis",
    "MESH:D008180": "systemic_lupus",
    "MESH:D012859": "sjogrens",
    "MESH:D012595": "systemic_sclerosis",
    "MESH:D008105": "primary_biliary_cholangitis",
    "MESH:D009157": "myasthenia_gravis",
    "MESH:D019693": "autoimmune_hepatitis",
    "MESH:D003424": "crohns_disease",
}

TRAIN_DISEASE_ENTITIES = [
    "Disease::MESH:D001172",
    "Disease::MESH:D009103",
    "Disease::MESH:D008180",
]

TEST_DISEASE_ENTITIES = [
    "Disease::MESH:D012859",  # Sjögren's
    "Disease::MESH:D012595",  # Systemic Sclerosis
    "Disease::MESH:D008105",  # PBC
    "Disease::MESH:D009157",  # Myasthenia Gravis
    "Disease::MESH:D019693",  # Autoimmune Hepatitis
    "Disease::MESH:D003424",  # Crohn's Disease
]

ALL_DISEASE_ENTITIES = TRAIN_DISEASE_ENTITIES + TEST_DISEASE_ENTITIES


# ============================================================================
# Entrez Gene ID → Gene Symbol (comprehensive immune genes)
# ============================================================================

ENTREZ_TO_SYMBOL = {
    # JAK-STAT pathway
    "3716": "JAK1", "3717": "JAK2", "3718": "JAK3", "7297": "TYK2",
    "6772": "STAT1", "6773": "STAT2", "6774": "STAT3", "6775": "STAT4",
    "6776": "STAT5A", "6777": "STAT5B", "6778": "STAT6",

    # TNF signaling
    "7124": "TNF", "7132": "TNFRSF1A", "7133": "TNFRSF1B",
    "8743": "TNFSF10", "10673": "TNFSF13B",
    "8717": "TRADD", "7186": "TRAF2", "8737": "RIPK1",
    "4790": "NFKB1", "5970": "RELA",
    "7292": "TNFSF4",

    # Interleukins
    "3552": "IL1A", "3553": "IL1B", "3554": "IL1R1", "3557": "IL1RN",
    "3558": "IL2", "3559": "IL2RA", "3560": "IL2RB",
    "3569": "IL6", "3570": "IL6R", "3572": "IL6ST",
    "3586": "IL10", "3587": "IL10RA", "3588": "IL10RB",
    "3605": "IL17A", "112744": "IL17F", "23765": "IL17RA",
    "51561": "IL23A", "149233": "IL23R",
    "3592": "IL12A", "3593": "IL12B", "3594": "IL12RB1",
    "3565": "IL4", "3567": "IL5", "3596": "IL13",
    "3600": "IL15", "3601": "IL15RA",
    "3606": "IL18",
    "50616": "IL22",
    "29949": "IL33",
    "90865": "IL33",  # alternate
    "3574": "IL7",

    # B-cell markers
    "930": "CD19", "931": "MS4A1", "933": "CD22",
    "939": "CD27", "952": "CD38", "958": "CD40",
    "3500": "IGHG1", "3507": "IGHM", "29760": "BLNK", "695": "BTK",

    # T-cell markers
    "915": "CD3D", "916": "CD3E", "917": "CD3G",
    "920": "CD4", "925": "CD8A", "926": "CD8B",
    "1493": "CTLA4", "5133": "PDCD1", "940": "CD28", "29851": "ICOS",
    "50943": "FOXP3", "2625": "GATA3", "30009": "TBX21", "6097": "RORC",

    # Cytokines & Chemokines
    "3458": "IFNG", "3439": "IFNG",  # both used
    "3454": "IFNA1", "3456": "IFNB1",
    "1437": "CSF2", "1435": "CSF1", "3576": "CXCL8",
    "6347": "CCL2", "6352": "CCL5", "3627": "CXCL10",

    # NF-kB pathway
    "4791": "NFKB2", "5971": "RELB",
    "3551": "IKBKB", "8517": "IKBKG", "1147": "CHUK",
    "4792": "NFKBIA", "4793": "NFKBIB",

    # Additional important immune genes
    "942": "CD86", "941": "CD80", "943": "CD96",
    "3383": "ICAM1", "7412": "VCAM1",
    "1436": "CSF1R", "2353": "FOS", "3725": "JUN",
    "5594": "MAPK1", "5595": "MAPK3",
    "2475": "MTOR", "5290": "PIK3CA",
    "207": "AKT1", "572": "BAD", "581": "BAX",
    "836": "CASP3", "840": "CASP7",
    "4609": "MYC", "7157": "TP53",
    "1029": "CDKN2A", "1026": "CDKN1A",
    "79679": "VTCN1",  # B7-H4 (immune checkpoint)

    # Additional genes found in autoimmune KG
    "6688": "SPI1", "6689": "SPIB",
    "3551": "IKBKB",
    "1956": "EGFR", "2064": "ERBB2",
    "5743": "PTGS2",  # COX-2
    "5742": "PTGS1",  # COX-1
    "301": "ANXA1",
    "1401": "CRP",
    "5468": "PPARG",
    "5467": "PPARD",
    "7040": "TGFB1",
    "7042": "TGFB2",
    "7043": "TGFB3",
    "3082": "HGF",
    "7422": "VEGFA",
    "2919": "CXCL1",
    "6354": "CCL7",
    "6363": "CCL19",
    "6366": "CCL21",
}

# Build reverse mapping symbol → entrez
SYMBOL_TO_ENTREZ = {v: k for k, v in ENTREZ_TO_SYMBOL.items()}


# ============================================================================
# Immune Pathway Genes (using Entrez IDs)
# ============================================================================

IMMUNE_PATHWAY_GENES = {
    "JAK-STAT": ["3716", "3717", "3718", "7297",
                 "6772", "6773", "6774", "6775", "6776", "6777", "6778"],
    "TNF_signaling": ["7124", "7132", "7133", "8743", "10673",
                      "8717", "7186", "8737", "4790", "5970", "7292"],
    "Interleukins": ["3552", "3553", "3554", "3557",
                     "3558", "3559", "3560",
                     "3569", "3570", "3572",
                     "3586", "3587", "3588",
                     "3605", "112744", "23765",
                     "51561", "149233",
                     "3592", "3593", "3594",
                     "3565", "3567", "3596",
                     "3606", "50616", "29949", "90865"],
    "B_cell_markers": ["930", "931", "933", "939", "952", "958",
                       "3500", "3507", "29760", "695"],
    "T_cell_markers": ["915", "916", "917", "920", "925", "926",
                       "1493", "5133", "940", "29851",
                       "50943", "2625", "30009", "6097"],
    "Cytokine_cascade": ["3458", "3439", "3454", "3456",
                         "1437", "1435", "3576",
                         "6347", "6352", "3627"],
    "NF-kB_pathway": ["4790", "4791", "5970", "5971",
                      "3551", "8517", "1147",
                      "4792", "4793"],
}

ALL_IMMUNE_GENE_IDS = set()
for pathway, genes in IMMUNE_PATHWAY_GENES.items():
    ALL_IMMUNE_GENE_IDS.update(genes)


# ============================================================================
# DrugBank ID → Drug Name (comprehensive)
# ============================================================================

DRUGBANK_TO_NAME = {
    # Known Sjögren's drugs
    "DB00635": "Prednisone", "DB00860": "Prednisolone",
    "DB01611": "Hydroxychloroquine", "DB01085": "Pilocarpine",
    "DB00305": "Mitomycin",
    # TNF inhibitors
    "DB00051": "Adalimumab", "DB00065": "Infliximab",
    "DB00073": "Rituximab", "DB00081": "Trastuzumab",
    "DB00002": "Cetuximab", "DB01041": "Thalidomide",
    "DB00608": "Chloroquine", "DB05676": "Ustekinumab",
    "DB06674": "Golimumab", "DB08904": "Certolizumab",
    "DB00005": "Etanercept",
    # JAK inhibitors
    "DB08895": "Tofacitinib", "DB11817": "Baricitinib",
    "DB12323": "Upadacitinib", "DB12010": "Filgotinib",
    # DMARDs
    "DB00563": "Methotrexate", "DB00795": "Sulfasalazine",
    "DB01050": "Ibuprofen", "DB00482": "Celecoxib",
    "DB00945": "Aspirin", "DB00784": "Mefenamic Acid",
    "DB00339": "Pyridostigmine", "DB00211": "Midodrine",
    # Immunosuppressants
    "DB00091": "Cyclosporine", "DB00993": "Azathioprine",
    "DB01234": "Dexamethasone", "DB00244": "Mesalazine",
    "DB01033": "Mercaptopurine", "DB00688": "Mycophenolate",
    "DB00966": "Teriflunomide", "DB01033": "Mercaptopurine",
    "DB00087": "Alemtuzumab",
    # Corticosteroids
    "DB00959": "Methylprednisolone", "DB01384": "Paramethasone",
    "DB00838": "Clocortolone", "DB00443": "Betamethasone",
    "DB00324": "Fluorometholone", "DB01013": "Clobetasol",
    "DB00620": "Triamcinolone", "DB04630": "Aldosterone",
    "DB00180": "Flunisolide", "DB00394": "Beclomethasone",
    "DB14645": "Budesonide",
    # Other common drugs
    "DB00891": "Sulfamethoxazole", "DB01077": "Etidronic Acid",
    "DB00720": "Clodronic Acid", "DB00650": "Leucovorin",
    "DB00400": "Griseofulvin", "DB02527": "Cyclic AMP",
    "DB05381": "Histamine", "DB04540": "Cholesterol",
    "DB05804": "Vaccine Components", "DB01257": "Spectinomycin",
    "DB01103": "Quinacrine", "DB01393": "Bezafibrate",
    "DB01586": "Ursodeoxycholic Acid",
    "DB06662": "Sparsentan", "DB01698": "Rutin",
    "DB00290": "Bleomycin", "DB01436": "Alfacalcidol",
    "DB09322": "Zinc Sulfate",
    # MS drugs
    "DB00120": "Phenylalanine", "DB01610": "Valganciclovir",
    "DB00115": "Cyanocobalamin", "DB00715": "Paroxetine",
    "DB01039": "Fenofibrate",
    # More common drugs
    "DB00100": "Hyaluronidase", "DB00594": "Amiloride",
    "DB01065": "Melatonin", "DB00811": "Ribavirin",
    "DB00630": "Alendronate", "DB01586": "Ursodeoxycholic Acid",
    "DB00943": "Zalcitabine",
    # Crohn's / GI drugs
    "DB00741": "Hydrocortisone", "DB01014": "Balsalazide",
    "DB00806": "Pentoxifylline",
    # Additional
    "DB00619": "Imatinib", "DB01259": "Lapatinib",
    "DB00530": "Erlotinib", "DB04868": "Nilotinib",
    "DB00398": "Sorafenib", "DB01268": "Sunitinib",
    "DB08896": "Ruxolitinib",
    "DB05259": "Omalizumab",
    "DB04557": "Arachidonic Acid",
    "DB00290": "Bleomycin",
    "DB09322": "Zinc Sulfate",
}

# ============================================================================
# PubChem Drug Name Lookup (for unknowns)
# ============================================================================

_drug_name_cache = {}
_drug_name_cache_file = Path('outputs/.drug_name_cache.json')


def _load_drug_cache():
    global _drug_name_cache
    if _drug_name_cache_file.exists():
        with open(_drug_name_cache_file) as f:
            _drug_name_cache = json.load(f)


def _save_drug_cache():
    _drug_name_cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(_drug_name_cache_file, 'w') as f:
        json.dump(_drug_name_cache, f, indent=2)


def lookup_drug_name_pubchem(drugbank_id):
    """Look up drug name via PubChem REST API."""
    if drugbank_id in _drug_name_cache:
        return _drug_name_cache[drugbank_id]

    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{drugbank_id}/property/Title/JSON"
        req = urllib.request.Request(url, headers={'User-Agent': 'BioPathNet/1.0'})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            name = data['PropertyTable']['Properties'][0]['Title']
            _drug_name_cache[drugbank_id] = name
            return name
    except Exception:
        _drug_name_cache[drugbank_id] = None
        return None


# ============================================================================
# Utility Functions
# ============================================================================

def get_disease_name(entity_str):
    if "::" in entity_str:
        mesh_id = entity_str.split("::", 1)[1]
    else:
        mesh_id = entity_str
    return MESH_TO_DISEASE.get(mesh_id, mesh_id)


def get_disease_short_name(entity_str):
    if "::" in entity_str:
        mesh_id = entity_str.split("::", 1)[1]
    else:
        mesh_id = entity_str
    return DISEASE_SHORT_NAMES.get(mesh_id, mesh_id.replace(":", "_"))


def get_drug_name(entity_str):
    """Get drug name from entity string. Uses local dict + PubChem fallback."""
    if "::" in entity_str:
        drug_id = entity_str.split("::", 1)[1]
    else:
        drug_id = entity_str

    # Try local dictionary first
    if drug_id in DRUGBANK_TO_NAME:
        return DRUGBANK_TO_NAME[drug_id]

    # Try MESH compound names
    if drug_id.startswith("MESH:") or drug_id.startswith("CHEBI:"):
        return drug_id  # Keep as-is for non-DrugBank IDs

    return drug_id


def get_drug_name_with_lookup(entity_str):
    """Get drug name with PubChem API fallback."""
    name = get_drug_name(entity_str)
    drug_id = entity_str.split("::", 1)[1] if "::" in entity_str else entity_str

    if name == drug_id and drug_id.startswith("DB"):
        # Try PubChem lookup
        pubchem_name = lookup_drug_name_pubchem(drug_id)
        if pubchem_name:
            return pubchem_name

    return name


def get_gene_symbol(entity_str):
    """Convert 'Gene::3569' → 'IL6'"""
    if "::" in entity_str:
        entrez_id = entity_str.split("::", 1)[1]
    else:
        entrez_id = entity_str
    return ENTREZ_TO_SYMBOL.get(entrez_id, f"Gene_{entrez_id}")


def get_mesh_id(entity_str):
    if "::" in entity_str:
        return entity_str.split("::", 1)[1]
    return entity_str


def get_drug_id(entity_str):
    if "::" in entity_str:
        return entity_str.split("::", 1)[1]
    return entity_str


def load_entity_mappings(processed_dir="datasets/processed"):
    e2id_path = Path(processed_dir) / "enriched_entity2id.json"
    with open(e2id_path, "r") as f:
        entity2id = json.load(f)
    id2entity = {v: k for k, v in entity2id.items()}
    return entity2id, id2entity


def get_compound_entities(entity2id):
    return [k for k in entity2id if k.startswith("Compound::")]


def get_disease_entities(entity2id):
    return [k for k in entity2id if k.startswith("Disease::")]


def get_gene_entities(entity2id):
    return [k for k in entity2id if k.startswith("Gene::")]


def check_immune_pathway(gene_entity):
    """
    Check if a gene belongs to a known immune pathway.
    Uses Entrez Gene IDs for matching.
    Returns list of matching pathways, or empty list.
    """
    if "::" in gene_entity:
        entrez_id = gene_entity.split("::", 1)[1]
    else:
        entrez_id = gene_entity

    matched = []
    for pathway, gene_ids in IMMUNE_PATHWAY_GENES.items():
        if entrez_id in gene_ids:
            matched.append(pathway)
    return matched


if __name__ == "__main__":
    print("=== Disease Mappings ===")
    for entity in ALL_DISEASE_ENTITIES:
        name = get_disease_name(entity)
        short = get_disease_short_name(entity)
        print(f"  {entity} → {name} ({short})")

    print(f"\n=== Gene Symbol Mapping Test ===")
    test_genes = ["Gene::3569", "Gene::7124", "Gene::920", "Gene::3439",
                  "Gene::50943", "Gene::3627", "Gene::79679", "Gene::7292"]
    for g in test_genes:
        sym = get_gene_symbol(g)
        pathways = check_immune_pathway(g)
        print(f"  {g} → {sym} (pathways: {pathways})")

    print(f"\n=== Sjögren's Direct Neighbors (mapped) ===")
    sj_genes = ["Gene::3627", "Gene::79679", "Gene::50943",
                "Gene::90865", "Gene::7292", "Gene::920",
                "Gene::3569", "Gene::7124", "Gene::3439"]
    for g in sj_genes:
        sym = get_gene_symbol(g)
        pathways = check_immune_pathway(g)
        print(f"  {g} → {sym}: {pathways}")

    sj_drugs = ["Compound::DB00635", "Compound::DB00860",
                "Compound::DB01611", "Compound::DB01085", "Compound::DB00305"]
    print(f"\n=== Sjögren's Known Drugs ===")
    for d in sj_drugs:
        print(f"  {d} → {get_drug_name(d)}")

    print(f"\n=== Immune Pathway Genes (Entrez IDs) ===")
    for pathway, genes in IMMUNE_PATHWAY_GENES.items():
        syms = [ENTREZ_TO_SYMBOL.get(g, g) for g in genes]
        print(f"  {pathway}: {', '.join(syms[:5])}... ({len(genes)} genes)")
    print(f"  Total unique immune gene IDs: {len(ALL_IMMUNE_GENE_IDS)}")
