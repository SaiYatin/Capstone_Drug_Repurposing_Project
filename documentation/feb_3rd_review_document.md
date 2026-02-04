# Dataset Enrichment Strategy for Feb 3rd Review

## BioPathNet++ for Autoimmune Drug Repurposing

---

## **STRATEGIC DECISION: Which Datasets to Add?**

### **RECOMMENDATION: Orphanet + HPO (Not GARD)**

| Dataset | Include? | Reason |
|---------|----------|--------|
| **Orphanet** | ✅ YES | 170% sensitivity improvement (BMC Med Inform 2019) |
| **HPO** | ✅ YES | 250,000 phenotype annotations, essential for rare diseases |
| **GARD** | ⚠️ OPTIONAL | 70% overlap with Orphanet, add later if needed |

**For Feb 3rd review:** Focus on **Orphanet + HPO** enrichment strategy.

---

## **1. BASE DATASET: DRKG**

### Download Information
- **Source:** https://github.com/gnn4dr/DRKG
- **Direct Link:** https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz
- **Size:** ~300 MB (compressed)
- **Format:** TSV (tab-separated values)

### Statistics
- **Total triples:** 5,874,261
- **Unique entities:** 97,238
  - Compounds/Drugs: 24,313
  - Genes/Proteins: 37,430
  - Diseases: 5,103
  - Pathways: 1,457
  - Biological Processes: 28,880
- **Unique relations:** 107 types

### Autoimmune Disease Coverage

| Disease | Total Links | Drug Links | Gene Links | Category |
|---------|------------|------------|------------|----------|
| Rheumatoid Arthritis | 1,695 | 426 | 1,161 | **TRAINING** |
| Systemic Lupus Erythematosus | 657 | 193 | 384 | **TRAINING** |
| Multiple Sclerosis | 689 | 226 | 345 | **TRAINING** |
| Crohn's Disease | 132 | 67 | 64 | **TRAINING** |
| Sjögren's Syndrome | **14** | **5** | **9** | **TARGET** ⭐ |
| Systemic Sclerosis | 173 | 87 | 79 | **TARGET** |
| Primary Biliary Cholangitis | 207 | 64 | 116 | **TARGET** |
| Myasthenia Gravis | 109 | 70 | 39 | **TARGET** |
| Autoimmune Hepatitis | 56 | 36 | 19 | **TARGET** |

**Key Insight:** Sjögren's with only 14 links is the PERFECT test case for zero-shot transfer learning.

---

## **2. ENRICHMENT DATASET: Orphanet**

### Download Information
- **Source:** https://www.orphadata.com/data/xml/
- **Files to download:**
  1. **Disease-Gene associations:** http://www.orphadata.org/data/xml/en_product6.xml (~50 MB)
  2. **Disease-Phenotype (HPO) associations:** http://www.orphadata.org/data/xml/en_product4.xml (~30 MB)
  3. **Disease nomenclature:** http://www.orphadata.org/data/xml/en_product1.xml (~20 MB)
- **Format:** XML
- **Total size:** ~100 MB

### Statistics
- **Total rare diseases:** 9,000+
- **Disease-gene associations:** ~150,000
- **Disease-phenotype associations:** ~85,000
- **Rare autoimmune diseases:** ~45
- **Unique genes:** ~4,500

### Why Orphanet?
**Scientific Evidence:** 
> "HPO-Orphanet enrichment improved sensitivity from 0.17 to 0.46 (170% improvement) and maintained specificity at 0.52"  
> — Zhu et al., BMC Medical Informatics and Decision Making (2019)

### Autoimmune Coverage
- Sjögren's Syndrome: ORPHA:289390
- Systemic Sclerosis: ORPHA:90291
- Autoimmune Hepatitis: ORPHA:2137
- Primary Biliary Cholangitis: ORPHA:186
- Myasthenia Gravis: ORPHA:589

---

## **3. ENRICHMENT DATASET: HPO**

### Download Information
- **Source:** https://hpo.jax.org/app/download/annotation
- **Files to download:**
  1. **Phenotype annotations:** http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa (~15 MB)
  2. **HPO ontology (OBO format):** http://purl.obolibrary.org/obo/hp.obo (~10 MB)
  3. **Disease-Phenotype annotations:** http://purl.obolibrary.org/obo/hp/hpoa/genes_to_phenotype.txt (~5 MB)
- **Format:** OBO (text) or TSV
- **Total size:** ~30 MB

### Statistics
- **Total phenotype terms:** 16,000+
- **Disease-phenotype annotations:** ~250,000
- **Annotated diseases:** 7,278
- **Hierarchical relationships:** ~45,000
- **Autoimmune-relevant phenotypes:** ~1,200

### Autoimmune-Relevant HPO Terms (Examples)
- **Autoimmunity:** HP:0002960
- **Autoimmune antibody positivity:** HP:0030057
- **Arthritis:** HP:0001369
- **Dry mouth:** HP:0000217 (Sjögren's marker)
- **Dry eye:** HP:0000488 (Sjögren's marker)
- **Raynaud phenomenon:** HP:0030880 (Systemic Sclerosis marker)

---

## **4. ENRICHMENT IMPACT**

### Expected Statistics

| Metric | Original DRKG | + Orphanet | + HPO | Enriched Total | Increase |
|--------|---------------|------------|-------|----------------|----------|
| **Total Triples** | 5,874,261 | +150,000 | +250,000 | 6,274,261 | **+6.8%** |
| **Disease Coverage** | 5,103 | +4,000 | +2,175 | 11,278 | **+121%** |
| **Phenotype Links** | Limited | +85,000 | +250,000 | +335,000 | **NEW** |

### Impact on Rare Diseases (Example: Sjögren's Syndrome)

| Source | Links Added | Type |
|--------|-------------|------|
| Original DRKG | 14 | Drug + Gene |
| + Orphanet | ~35 | Gene + Phenotype |
| + HPO | ~50 | Phenotype annotations |
| **TOTAL** | **~99** | **All types** |
| **Increase** | **+607%** | - |

---

## **5. KEY PAPERS TO CITE**

### Primary Citation (MUST INCLUDE)
**Title:** "Rare disease knowledge enrichment through a data-driven approach"  
**Authors:** Zhu et al.  
**Journal:** BMC Medical Informatics and Decision Making (2019)  
**DOI:** 10.1186/s12911-019-0752-9  
**Key Finding:** HPO-Orphanet enrichment improved sensitivity from 0.17 to 0.46 (170%)

### Supporting Citations

1. **"Clustering rare diseases within an ontology-enriched knowledge graph"**  
   - JAMIA (2023)
   - Demonstrates effective rare disease clustering with HPO+Orphanet

2. **"Application of knowledge graphs in rare disease research"**  
   - Frontiers in Public Health (2025)
   - Shows KG integration significantly improves rare disease discoverability

3. **"A Hypergraph powered approach to Phenotype-driven Gene Prioritization"**  
   - Scientific Reports (2025)
   - Uses 2,130 diseases, 4,655 genes, 9,541 phenotypes from Orphanet+HPO
   - Outperforms state-of-the-art tools

---

## **6. IMPLEMENTATION TIMELINE**

### Week 1-2 (Feb 3-16): Data Collection
- ✅ Download DRKG
- ✅ Download Orphanet XML files
- ✅ Download HPO annotations
- ✅ Verify file integrity

### Week 3-4 (Feb 17-Mar 2): Data Processing
- Parse Orphanet XML → Extract disease-gene, disease-phenotype triples
- Parse HPO files → Extract phenotype annotations
- Entity alignment (map Orphanet IDs to DRKG entities)
- Quality validation

### Week 5-6 (Mar 3-16): Data Merging
- Merge Orphanet triples with DRKG
- Merge HPO triples with enriched graph
- Deduplicate overlapping entities
- Validate enriched graph structure

### Week 7-8 (Mar 17-30): Validation & Analysis
- Statistical analysis of enrichment
- Autoimmune subgraph extraction
- Prepare visualizations
- Document enrichment process

---

## **7. EXPECTED OUTCOMES FOR REVIEW**

### Deliverables for Feb 3rd
1. ✅ **Dataset Overview Presentation**
   - DRKG statistics
   - Orphanet statistics
   - HPO statistics
   - Enrichment strategy

2. ✅ **Visualizations**
   - Autoimmune disease data distribution
   - Enrichment impact chart
   - Rare disease coverage improvement

3. ✅ **Technical Justification**
   - Scientific papers supporting enrichment
   - Expected performance improvement (20-30%)
   - Biological relevance for autoimmune transfer learning

### Questions Panel May Ask

**Q: Why not use GARD?**  
**A:** GARD has ~70% overlap with Orphanet. We'll add it in Phase 2 if needed, but Orphanet + HPO provide sufficient enrichment for our autoimmune focus.

**Q: How do you integrate these different formats (XML, TSV, OBO)?**  
**A:** All convert to triple format (entity1, relation, entity2). We parse XML/OBO using standard libraries (xml.etree, obonet) and merge into unified TSV.

**Q: What about data quality / noise?**  
**A:** Orphanet and HPO are curated databases with quality control. We'll validate entity alignment and use confidence thresholds for associations.

**Q: Can BioPathNet++ handle this enriched graph?**  
**A:** Yes, BioPathNet++ is designed for heterogeneous knowledge graphs. Enrichment adds ~7% more triples, well within capacity.

---

## **8. CONCLUSION**

### Summary
- **Base:** DRKG (5.8M triples, 97K entities)
- **Enrichment:** Orphanet (~150K triples) + HPO (~250K triples)
- **Total:** ~6.3M triples (+6.8% increase)
- **Impact on rare diseases:** +300-600% coverage per disease
- **Scientific backing:** 170% sensitivity improvement (published research)

### Recommendation
**Proceed with Orphanet + HPO enrichment for the following reasons:**
1. ✅ Strong scientific evidence (BMC 2019 paper)
2. ✅ Manageable implementation (2-4 weeks)
3. ✅ Significant impact on rare autoimmune diseases
4. ✅ Compatible with BioPathNet++ architecture
5. ✅ Addresses data scarcity problem for transfer learning

### Next Steps
1. Get panel approval on Feb 3rd
2. Download datasets (Week 1-2)
3. Process and merge (Week 3-6)
4. Validate enrichment (Week 7-8)
5. Begin BioPathNet++ training on enriched graph

---

**Prepared by:** Capstone Team  
**Date:** January 29, 2026  
**Next Review:** February 3, 2026
