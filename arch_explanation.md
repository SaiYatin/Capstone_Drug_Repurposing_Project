# BioPathNet++ Architecture: Complete Step-by-Step Explanation

## Project: Path-Based Graph Neural Networks for Autoimmune Drug Repurposing

---

## **ARCHITECTURE OVERVIEW**

This architecture implements a **multi-disease transfer learning framework** using BioPathNet++ to predict drugs for rare autoimmune diseases with minimal training data.

**The Problem We Solve:**
- Sjögren's Syndrome has only **14 drug associations** in DRKG
- Traditional models fail with such sparse data
- We need to transfer knowledge from data-rich diseases (RA: 1,695 links) to data-sparse diseases

**Our Solution:**
- Use path-based reasoning (not just node embeddings)
- Learn from multiple autoimmune diseases simultaneously
- Transfer shared immune mechanisms to rare diseases
- Provide interpretable predictions with biological paths

---

## **COMPLETE ARCHITECTURE FLOW**

### **PHASE 1: DATA PREPARATION & ENRICHMENT**

---

#### **STEP 1: Load Base DRKG**
**What:** Load the Drug Repurposing Knowledge Graph (DRKG)

**Why:**
- DRKG contains 5.8M biomedical triples
- Includes drugs, genes, diseases, pathways, proteins
- Industry-standard knowledge graph for drug repurposing

**Input:**
- Raw DRKG file: `drkg.tsv`
- Format: `(entity1, relation, entity2)`

**Output:**
- Loaded knowledge graph: 97,238 entities, 5.8M triples

**Code Concept:**
```python
drkg = pd.read_csv('drkg.tsv', sep='\t')
# Result: 5,874,261 triples
```

**Visualization in Architecture:**
```
┌─────────────────────┐
│   Load DRKG Base    │
│  97K entities       │
│  5.8M triples       │
└─────────────────────┘
```

---

#### **STEP 2: Extract Autoimmune Subgraph**
**What:** Filter DRKG to keep only autoimmune disease-related entities

**Why:**
- Full DRKG contains ALL diseases (cancer, diabetes, etc.)
- We only need autoimmune-related data
- Reduces computational cost by ~80%
- Focuses model on relevant domain

**Input:**
- Full DRKG (5.8M triples)
- List of 9 autoimmune diseases

**Process:**
1. Identify autoimmune disease nodes
2. Find all connected drugs, genes, pathways
3. Extract 1-hop and 2-hop neighborhoods
4. Keep only relevant entities

**Output:**
- Autoimmune subgraph: ~50,000 entities, ~1.2M triples

**Code Concept:**
```python
autoimmune_diseases = [
    'DOID:7148',  # Rheumatoid Arthritis
    'DOID:9074',  # Systemic Lupus Erythematosus
    'DOID:12894', # Sjögren's Syndrome
    # ... 6 more
]

subgraph = drkg[
    drkg['head'].isin(autoimmune_diseases) | 
    drkg['tail'].isin(autoimmune_diseases)
]
# Expand to 2-hop neighborhood
```

**Visualization in Architecture:**
```
┌─────────────────────┐
│Extract Autoimmune   │
│   Subgraph          │
│ 9 target diseases   │
│ ~50K entities       │
│ ~1.2M triples       │
└─────────────────────┘
```

---

#### **STEP 3: Enrichment with Orphanet + HPO**
**What:** Add rare disease metadata from Orphanet and phenotype data from HPO

**Why:**
- DRKG alone: Sjögren's has only 14 links (too sparse!)
- Orphanet adds: ~35 disease-gene associations
- HPO adds: ~50 phenotype annotations (dry mouth, dry eye, etc.)
- **Result:** 14 → ~99 links (+607% coverage)

**Why This Works:**
- Orphanet provides rare disease genetics
- HPO provides clinical phenotypes
- Phenotypes help connect similar diseases
- Published research shows 20-30% improvement (BMC 2019)

**Input:**
- Autoimmune subgraph (1.2M triples)
- Orphanet XML files (~150K disease-gene triples)
- HPO annotations (~250K phenotype triples)

**Process:**
1. Parse Orphanet XML → Extract disease-gene pairs
2. Parse HPO files → Extract disease-phenotype pairs
3. Map Orphanet IDs to DRKG disease entities
4. Merge new triples into subgraph
5. Deduplicate overlapping entities

**Output:**
- Enriched subgraph: ~55K entities, ~1.6M triples (+33% increase)

**Code Concept:**
```python
# Parse Orphanet
orphanet_triples = parse_orphanet_xml('en_product6.xml')
# Result: ~150,000 (disease, associated_with, gene)

# Parse HPO
hpo_triples = parse_hpo('phenotype.hpoa')
# Result: ~250,000 (disease, has_phenotype, phenotype)

# Merge
enriched_graph = merge([subgraph, orphanet_triples, hpo_triples])
```

**Impact on Sjögren's:**
```
Original:  14 links  [████░░░░░░░░░░░░░░░░░░░]
Enriched:  99 links  [████████████████████████]
```

**Visualization in Architecture:**
```
┌──────────────────────────────┐
│  Enrichment Layer            │
│  + Orphanet (150K triples)   │
│  + HPO (250K triples)        │
│  ─────────────────────────   │
│  Enriched: 55K entities      │
│           1.6M triples       │
└──────────────────────────────┘
```

---

### **PHASE 2: FEATURE ENGINEERING**

---

#### **STEP 4: Generate Node Embeddings (BioBERT + ChemBERTa)**
**What:** Convert entities (drugs, diseases, genes) into numerical vector representations

**Why:**
- Neural networks can't process text directly
- Need fixed-size vectors for all entities
- BioBERT: Pre-trained on biomedical literature (diseases, genes)
- ChemBERTa: Pre-trained on chemical structures (drugs)
- These capture semantic meaning (e.g., "TNF-α inhibitor" close to "anti-inflammatory")

**Input:**
- Entity names/descriptions from enriched graph
- Example: "Methotrexate", "Sjögren's Syndrome", "TNF-α"

**Process:**
1. For drugs/compounds: Use ChemBERTa
   - Encode SMILES string → 768-dim vector
2. For diseases/genes/proteins: Use BioBERT
   - Encode entity name + description → 768-dim vector
3. Store embeddings for all 55K entities

**Output:**
- Embedding matrix: 55,000 × 768 dimensions
- Each entity = 768 numbers capturing its meaning

**Code Concept:**
```python
from transformers import AutoModel, AutoTokenizer

# For drugs
chembert = AutoModel.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
drug_embedding = chembert.encode('SMILES_string')  # → 768-dim

# For diseases/genes
biobert = AutoModel.from_pretrained('dmis-lab/biobert-v1.1')
disease_embedding = biobert.encode("Sjögren's Syndrome")  # → 768-dim
```

**Why This Matters:**
- Similar entities have similar vectors
- E.g., IL-6 and TNF-α (both cytokines) will be close in vector space
- Model can generalize: "If drug X inhibits TNF-α, it might also inhibit IL-6"

**Visualization in Architecture:**
```
┌────────────────────────────────┐
│   Node Embedding Layer         │
│                                │
│  BioBERT (Diseases, Genes)     │
│  ChemBERTa (Drugs)             │
│                                │
│  Output: 55K × 768 matrix      │
└────────────────────────────────┘
```

---

#### **STEP 5: Build Autoimmune Background Regulatory Graph (BRG)**
**What:** Create a specialized graph of immune system interactions

**Why:**
- Standard DRKG doesn't have all gene-gene regulations
- Immune system has specific pathways (JAK-STAT, NF-κB)
- Adding these helps the model understand autoimmune mechanisms
- BioPathNet++ uses BRG to enrich information flow

**What We Add:**
1. **Gene-Gene Regulations** (from STRING database)
   - Example: STAT3 → regulates → IL6
2. **Cytokine Interaction Network**
   - Example: TNF-α → activates → IL-6
3. **Immune Pathway Edges**
   - Example: JAK1 → participates → JAK-STAT_pathway
4. **Protein-Protein Interactions** (from IntAct)
   - Example: CD80 → binds → CTLA4

**Input:**
- STRING database (protein interactions)
- Immune pathway databases (KEGG, Reactome)
- Cytokine network data

**Process:**
1. Query STRING for immune gene interactions
2. Extract JAK-STAT, NF-κB, T-cell signaling pathways
3. Build cytokine cascade network
4. Format as triples: (gene1, regulates, gene2)

**Output:**
- Background Regulatory Graph: ~50,000 regulatory edges
- Specifically focused on immune mechanisms

**Code Concept:**
```python
# Extract immune regulatory network
brg_edges = []

# From STRING
string_immune = query_string_db(genes=['JAK1', 'JAK2', 'STAT3', ...])
brg_edges.extend(string_immune)

# Immune pathways
jak_stat_pathway = extract_pathway('JAK-STAT signaling')
brg_edges.extend(jak_stat_pathway)

# Cytokine network
cytokines = ['TNF-α', 'IL-6', 'IL-17', 'IFN-γ']
cytokine_network = build_interaction_network(cytokines)
brg_edges.extend(cytokine_network)

# Result: ~50,000 regulatory edges
```

**Why This Is Novel:**
- **First autoimmune-specific BRG**
- Standard BioPathNet++ uses generic regulatory graphs
- Ours is tailored to immune mechanisms
- Helps model learn autoimmune-specific patterns

**Visualization in Architecture:**
```
┌──────────────────────────────────┐
│ Background Regulatory Graph      │
│ (Autoimmune-Specific)            │
│                                  │
│ - Gene-Gene regulations          │
│ - Cytokine interactions          │
│ - JAK-STAT, NF-κB pathways       │
│                                  │
│ Output: 50K immune edges         │
└──────────────────────────────────┘
```

---

#### **STEP 6: Define Immune-Specific Metapaths**
**What:** Create templates for biologically meaningful paths between drugs and diseases

**Why:**
- Not all paths are therapeutically relevant
- Metapaths constrain the model to learn real biology
- E.g., Drug → Gene → Disease is valid, but Drug → City → Disease is nonsense
- Immune-specific metapaths focus on autoimmune mechanisms

**The 7 Immune Metapaths:**

**Metapath 1: TNF-α Inhibition**
```
Drug → inhibits → TNF-α → regulates → Inflammation → associates → Disease
```
Example: Adalimumab → inhibits → TNF-α → ... → Rheumatoid Arthritis

**Metapath 2: JAK-STAT Modulation**
```
Drug → modulates → JAK1/2 → activates → STAT3 → dysregulated_in → Disease
```
Example: Tofacitinib → modulates → JAK1 → ... → Sjögren's

**Metapath 3: IL-6 Blockade**
```
Drug → blocks → IL-6 → triggers → B-cell_activation → causes → Disease
```
Example: Tocilizumab → blocks → IL-6 → ... → Autoimmune diseases

**Metapath 4: T-cell Targeting**
```
Drug → targets → CD20/CD4 → depletes → T-cells → involved_in → Disease
```
Example: Rituximab → targets → CD20 → ... → Systemic Lupus

**Metapath 5: Cytokine Cascade**
```
Drug → inhibits → Cytokine → activates → Immune_cell → pathogenesis → Disease
```

**Metapath 6: Pathway Intervention**
```
Drug → modulates → Gene → participates_in → Pathway → dysregulated_in → Disease
```

**Metapath 7: Tissue Specificity**
```
Drug → targets → Protein → expressed_in → Tissue → affected_in → Disease
```

**Input:**
- Domain knowledge from immunology
- Literature on autoimmune mechanisms
- Pathway databases (KEGG, Reactome)

**Output:**
- 7 metapath templates (relation sequences)

**Code Concept:**
```python
metapaths = [
    ['inhibits', 'regulates', 'associates'],  # TNF-α inhibition
    ['modulates', 'activates', 'dysregulated_in'],  # JAK-STAT
    ['blocks', 'triggers', 'causes'],  # IL-6 blockade
    ['targets', 'depletes', 'involved_in'],  # T-cell targeting
    ['inhibits', 'activates', 'pathogenesis'],  # Cytokine cascade
    ['modulates', 'participates_in', 'dysregulated_in'],  # Pathway
    ['targets', 'expressed_in', 'affected_in']  # Tissue-specific
]
```

**Why This Is Novel:**
- **First immune-specific metapath set**
- Standard BioPathNet++ uses generic metapaths
- These capture known autoimmune mechanisms
- Biologically interpretable

**Visualization in Architecture:**
```
┌─────────────────────────────────────┐
│  Immune-Specific Metapaths (7)      │
│                                     │
│  1. TNF-α Inhibition                │
│  2. JAK-STAT Modulation             │
│  3. IL-6 Blockade                   │
│  4. T-cell Targeting                │
│  5. Cytokine Cascade                │
│  6. Pathway Intervention            │
│  7. Tissue Specificity              │
└─────────────────────────────────────┘
```

---

### **PHASE 3: MODEL TRAINING (BIOPATHNET++ CORE)**

---

#### **STEP 7: Build BioPathNet++ Model**
**What:** Construct the Neural Bellman-Ford Network architecture

**Why:**
- Learns to find and score biological paths
- Path-based reasoning (not just node embeddings)
- Naturally handles zero-shot scenarios
- Provides interpretable predictions

**Architecture Components:**

**Component 1: Neural Bellman-Ford Layers (NBFNet)**
- Iteratively extends paths from Drug to Disease
- Layer 1: Find 1-hop paths
- Layer 2: Extend to 2-hops
- Layer 3-6: Up to 6-hop paths
- Uses learned relation transformations

**Component 2: Path Scoring Network**
- Takes a path as input
- Scores how likely it's therapeutic
- Uses GRU to model path sequence
- Outputs probability score

**Component 3: Path Attention Mechanism**
- Multiple paths can connect Drug → Disease
- Learns which paths are most important
- Outputs attention weights (α₁, α₂, ...)
- Final score = weighted average

**Input:**
- Enriched graph with embeddings
- Background Regulatory Graph
- Immune-specific metapaths

**Output:**
- Trained BioPathNet++ model

**Code Architecture:**
```python
class BioPathNetAutoimmune(nn.Module):
    def __init__(self):
        # NBFNet core
        self.nbfnet = NBFNetConv(
            num_relations=107,  # DRKG relation types
            hidden_dim=64,
            num_layers=6  # 6-hop paths
        )
        
        # Path scorer
        self.path_gru = nn.GRU(64, 64)
        self.scorer = nn.Linear(64, 1)
        
        # Attention
        self.attention = nn.Linear(64, 1)
    
    def forward(self, drug, disease, graph, brg):
        # Merge main graph + BRG
        full_graph = merge(graph, brg)
        
        # Generate paths (Bellman-Ford iterations)
        paths = self.nbfnet.find_paths(
            source=drug,
            target=disease,
            graph=full_graph,
            metapaths=self.metapaths,
            max_length=6
        )
        
        # Score each path
        path_scores = []
        for path in paths:
            # Encode path sequence
            path_repr = self.path_gru(path.relations)
            score = self.scorer(path_repr)
            path_scores.append(score)
        
        # Attention weighting
        attn_weights = softmax(self.attention(paths))
        
        # Final prediction
        final_score = sum(attn_weights * path_scores)
        
        return final_score, paths, attn_weights
```

**Visualization in Architecture:**
```
┌─────────────────────────────────────────┐
│     BioPathNet++ Core Model             │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Neural Bellman-Ford Network      │  │
│  │ - Layer 1: 1-hop paths           │  │
│  │ - Layer 2: 2-hop paths           │  │
│  │ - Layer 3-6: Up to 6-hop         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Path Scoring (GRU + MLP)         │  │
│  └───────────────────────────────────┘  │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │ Path Attention Layer             │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

---

#### **STEP 8: Multi-Disease Joint Training**
**What:** Train the model simultaneously on 4 data-rich autoimmune diseases

**Why:**
- Single-disease training: Model only learns RA-specific patterns
- Multi-disease training: Model learns SHARED autoimmune mechanisms
- Shared learning enables transfer to rare diseases
- More data = better generalization

**Training Diseases:**
1. **Rheumatoid Arthritis** (1,695 drug-disease pairs)
2. **Systemic Lupus Erythematosus** (657 pairs)
3. **Multiple Sclerosis** (689 pairs)
4. **Crohn's Disease** (132 pairs)

**Total Training Data:** 3,173 positive drug-disease pairs

**What Model Learns:**
- TNF-α inhibition works across RA, SLE, Crohn's
- JAK-STAT modulation effective in RA and MS
- IL-6 blockade therapeutic in multiple diseases
- B-cell/T-cell targeting is cross-disease

**Training Procedure:**
1. **Positive samples:** Known drug-disease pairs
2. **Negative samples:** 
   - Hard negatives from RepoDB (failed clinical trials)
   - Random negatives (non-associated drugs)
   - Ratio: 1 positive : 1 hard negative : 2 easy negatives
3. **Loss function:** Binary cross-entropy
4. **Optimization:** AdamW with learning rate 1e-4
5. **Epochs:** 200 with early stopping

**Input:**
- Drug-disease pairs from 4 training diseases
- Enriched graph + BRG
- Node embeddings

**Output:**
- Trained model weights θ_autoimmune

**Code Concept:**
```python
# Training loop
for epoch in range(200):
    for batch in train_loader:
        # Batch contains: (drug, disease, label)
        drugs, diseases, labels = batch
        
        # Forward pass
        predictions = []
        for drug, disease in zip(drugs, diseases):
            score, paths, attn = model(drug, disease, graph, brg)
            predictions.append(score)
        
        # Loss
        loss = binary_cross_entropy(predictions, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
    
    # Validation
    val_auprc = evaluate(model, val_data)
    if val_auprc > best_auprc:
        save_checkpoint(model)
```

**Visualization in Architecture:**
```
┌───────────────────────────────────────┐
│   Multi-Disease Training              │
│                                       │
│   Training Set (Data-Rich):           │
│   - RA: 1,695 pairs                   │
│   - SLE: 657 pairs                    │
│   - MS: 689 pairs                     │
│   - Crohn's: 132 pairs                │
│                                       │
│   Learn Shared Patterns:              │
│   ✓ TNF-α inhibition                  │
│   ✓ JAK-STAT modulation               │
│   ✓ IL-6 blockade                     │
│   ✓ B/T-cell targeting                │
└───────────────────────────────────────┘
```

---

### **PHASE 4: TRANSFER LEARNING & PREDICTION**

---

#### **STEP 9: Transfer to Rare Diseases (Zero-Shot/Few-Shot)**
**What:** Apply trained model to rare autoimmune diseases with minimal data

**Why:**
- Sjögren's has only 14 known drugs (not enough to train)
- Model learned shared mechanisms from RA/SLE/MS
- Can reuse those patterns for Sjögren's
- This is the CORE innovation: transfer learning for rare diseases

**Target Diseases:**
1. **Sjögren's Syndrome** (14 links) ← Perfect zero-shot test
2. **Systemic Sclerosis** (173 links)
3. **Primary Biliary Cholangitis** (207 links)
4. **Myasthenia Gravis** (109 links)
5. **Autoimmune Hepatitis** (56 links)

**Two Transfer Strategies:**

**Option A: Pure Zero-Shot (Recommended)**
- NO fine-tuning on target disease
- Direct inference using θ_autoimmune
- Tests true transfer capability

**Option B: Few-Shot Fine-Tuning**
- Fine-tune on 5-10 examples from target disease
- Quick adaptation (10-20 epochs)
- Improves performance by ~5-10%

**How Transfer Works (Example: Sjögren's)**

**Model Reasoning:**
```
1. Query: "Which drugs treat Sjögren's Syndrome?"

2. Model finds paths using learned patterns:
   
   Path 1 (from RA knowledge):
   Rituximab → targets → CD20 
            → depletes → B-cells 
            → reduces → autoantibodies
            → treats → [RA] ✓ (known)
   
   Transfer to Sjögren's:
   Rituximab → targets → CD20
            → depletes → B-cells
            → reduces → autoantibodies
            → treats → Sjögren's ✓ (predicted!)
   
   Why it works: Sjögren's ALSO has B-cell dysfunction
                 Model learned: CD20-targeting helps autoimmune

3. Score: 0.92 (high confidence)
```

**Input:**
- Trained model θ_autoimmune
- Query: (Candidate_Drug, Sjögren's_Syndrome)
- Enriched graph

**Process:**
1. For each drug in database (24,313 drugs):
   - Run model: score, paths, attention = model(drug, Sjögren's)
   - Store: (drug, score, top_paths)
2. Rank drugs by score
3. Return top-20 candidates

**Output:**
- Ranked drug list for each rare disease
- With mechanistic paths as explanations
- Confidence scores

**Code Concept:**
```python
# Zero-shot prediction for Sjögren's
model.eval()  # Inference mode
predictions = []

for drug in all_drugs:  # 24,313 drugs
    # Get prediction
    score, paths, attn = model(
        drug='Compound::DB12345',
        disease='Disease::DOID:12894',  # Sjögren's
        graph=enriched_graph,
        brg=autoimmune_brg
    )
    
    # Store result
    predictions.append({
        'drug': drug,
        'score': score.item(),
        'top_paths': paths[:5],  # Top 5 paths
        'attention': attn[:5]
    })

# Rank by score
predictions.sort(key=lambda x: x['score'], reverse=True)

# Top-20 candidates
top_20 = predictions[:20]
```

**Visualization in Architecture:**
```
┌──────────────────────────────────────┐
│    Transfer Learning                 │
│                                      │
│    Target Set (Data-Sparse):         │
│    - Sjögren's: 14 links             │
│    - Systemic Sclerosis: 173         │
│    - PBC: 207                        │
│    - Myasthenia Gravis: 109          │
│    - Autoimmune Hepatitis: 56        │
│                                      │
│    Strategy: Zero-Shot Inference     │
│    Using learned θ_autoimmune        │
│                                      │
│    Output: Top-20 drugs per disease  │
└──────────────────────────────────────┘
```

---

#### **STEP 10: Path Extraction & Ranking**
**What:** Extract the biological paths that support each prediction

**Why:**
- Predictions alone aren't enough (black box)
- Clinicians need to understand WHY
- Paths provide mechanistic explanations
- Enable biological validation

**For Each Prediction:**
1. Drug X → Disease Y with score 0.87
2. Model found 50 paths connecting them
3. Extract top-5 paths by attention weight
4. Format for human interpretation

**Example Output for Tocilizumab → Sjögren's:**

```
Prediction: Tocilizumab for Sjögren's Syndrome
Overall Score: 0.89 (High Confidence)

Top Paths:

Path 1 (Attention: 0.45) - Primary Mechanism
Tocilizumab → blocks → IL-6 
            → suppresses → B-cell_activation
            → reduces → Autoantibodies
            → ameliorates → Sjögren's_Syndrome
Literature Support: 15 papers

Path 2 (Attention: 0.32) - Secondary Mechanism  
Tocilizumab → inhibits → IL-6_signaling
            → downregulates → STAT3
            → reduces → Inflammation
            → treats → Sjögren's_Syndrome
Literature Support: 8 papers

Path 3 (Attention: 0.15)
Tocilizumab → modulates → Cytokine_cascade
            → affects → T-cell_function
            → involved_in → Sjögren's_Syndrome
Literature Support: 5 papers

Total Evidence: 28 papers support mechanisms
Recommendation: HIGH priority for experimental testing
```

**Input:**
- Model prediction: score, all_paths, attention_weights
- Top-K parameter (K=5 usually)

**Process:**
1. Sort paths by attention weight
2. Select top-K paths
3. Extract relation sequence
4. Map to human-readable format
5. Optionally: Query PubMed for path validation

**Output:**
- Top-K mechanistic paths per prediction
- Attention weights (importance)
- Human-readable explanations

**Code Concept:**
```python
def extract_top_paths(prediction, k=5):
    """Extract top-K paths from prediction"""
    
    # Get paths and attention
    paths = prediction['paths']
    attention = prediction['attention']
    
    # Sort by attention
    sorted_indices = np.argsort(attention)[::-1]
    top_k_indices = sorted_indices[:k]
    
    # Extract top paths
    top_paths = []
    for idx in top_k_indices:
        path = paths[idx]
        
        # Convert to readable format
        path_str = format_path(path)
        # "Tocilizumab → blocks → IL-6 → suppresses → ..."
        
        # Validate in literature
        lit_support = query_pubmed(path)
        
        top_paths.append({
            'path': path_str,
            'attention': attention[idx],
            'literature_support': lit_support
        })
    
    return top_paths
```

**Visualization in Architecture:**
```
┌──────────────────────────────────────┐
│    Path Extraction & Ranking         │
│                                      │
│    For each prediction:              │
│    1. Extract top-5 paths            │
│    2. Sort by attention weight       │
│    3. Format for interpretation      │
│    4. Validate with literature       │
│                                      │
│    Output: Mechanistic explanations  │
└──────────────────────────────────────┘
```

---

### **PHASE 5: VALIDATION & EVALUATION**

---

#### **STEP 11: Multi-Layer Validation**
**What:** Validate predictions through 5 independent validation layers

**Why Single Validation Isn't Enough:**
- Drug-disease testing doesn't exist for rare diseases
- Need multiple lines of evidence
- High confidence = passes multiple validation layers
- Mimics real scientific validation process

**The 5 Validation Layers:**

---

**Layer 1: RepoDB Benchmark (Computational)**

**What:** Test on known approved/failed drugs from RepoDB database

**Dataset:**
- 6,677 approved drug-disease pairs (positive)
- 4,123 failed clinical trials (negative)

**Metrics:**
- AUPRC (Area Under Precision-Recall Curve)
- AUROC (Area Under ROC Curve)
- Precision@K (top-10, top-20)
- Recall@K

**Baseline Comparisons:**
- Random prediction
- Degree-based ranking
- TransE (KGE baseline)
- GAT (GNN baseline)
- TxGNN (foundation model baseline)

**Success Criteria:**
- AUPRC ≥ 0.92
- AUROC ≥ 0.94
- ≥10% improvement over TxGNN

**Code Concept:**
```python
from sklearn.metrics import auc, precision_recall_curve

# Predict on RepoDB
y_true = repodb['label']  # 0 or 1
y_pred = []

for drug, disease in repodb[['drug', 'disease']]:
    score, _, _ = model(drug, disease, graph, brg)
    y_pred.append(score.item())

# Calculate metrics
precision, recall, _ = precision_recall_curve(y_true, y_pred)
auprc = auc(recall, precision)

print(f"AUPRC: {auprc:.3f}")  # Target: ≥0.92
```

---

**Layer 2: Zero-Shot Disease-Holdout Testing**

**What:** Train on 4 diseases, test on 5 held-out diseases (true transfer test)

**Protocol:**
- Training: RA, SLE, MS, Crohn's only
- Testing: Sjögren's, SSc, PBC, MG, AIH
- NO data from test diseases used in training

**Metrics:**
- Zero-shot AUPRC on held-out diseases
- Hit rate: % of known drugs in top-20

**Success Criteria:**
- Zero-shot AUPRC ≥ 0.80
- ≥60% of known drugs in top-20

**This Proves:** Model truly transfers knowledge, doesn't just memorize

---

**Layer 3: Temporal Validation**

**What:** Train on old data, test on future discoveries

**Protocol:**
- Training: Drug-disease associations ≤ 2015
- Validation: Associations from 2016-2020
- Testing: Associations from 2021-2025

**Why This Matters:**
- Prevents future-knowledge leakage
- Tests if model predicts "future" discoveries
- Validates real-world applicability

**Success Criteria:**
- Hit rate ≥ 40% in top-50 for post-2020 discoveries

**Example:**
```
Model trained on ≤2015 data predicts:
"Baricitinib may treat Rheumatoid Arthritis"

Actual: Baricitinib FDA approved for RA in 2018 ✓
Result: Correct prediction!
```

---

**Layer 4: Literature-Based Path Validation**

**What:** Validate mechanistic paths using PubMed literature

**Protocol:**
1. Extract top-10 predictions per disease
2. For each: Get top-5 paths
3. Query PubMed: "Drug AND Disease AND pathway_gene"
4. Classify evidence:
   - **Strong:** ≥3 papers support mechanism
   - **Moderate:** 1-2 papers support
   - **Weak:** 0 papers

**Success Criteria:**
- ≥70% of predictions have moderate+ evidence

**Example:**
```
Prediction: Rituximab → Sjögren's Syndrome

Path 1: Rituximab → CD20 → B-cells → Sjögren's

PubMed Search: "Rituximab AND Sjögren's AND CD20"
Results: 12 papers found ✓
Evidence: STRONG

Path 2: Rituximab → autoantibodies → Sjögren's

PubMed Search: "Rituximab AND Sjögren's AND autoantibodies"
Results: 8 papers found ✓
Evidence: STRONG

Overall: 2/2 paths validated → HIGH CONFIDENCE
```

**Code Concept:**
```python
from Bio import Entrez

def validate_path_literature(drug, disease, path_genes):
    """Query PubMed for path support"""
    
    # Build search query
    query = f"{drug} AND {disease} AND ({' OR '.join(path_genes)})"
    
    # Search PubMed
    handle = Entrez.esearch(db="pubmed", term=query)
    results = Entrez.read(handle)
    
    num_papers = int(results['Count'])
    
    # Classify evidence
    if num_papers >= 3:
        return "STRONG"
    elif num_papers >= 1:
        return "MODERATE"
    else:
        return "WEAK"
```

---

**Layer 5: Immune Pathway Analysis (Novel Validation)**

**What:** Verify paths contain known autoimmune mechanisms

**Protocol:**
1. Extract predicted paths
2. Check if they contain:
   - JAK-STAT pathway components
   - TNF-α or IL-6 interactions
   - B-cell or T-cell markers
   - Cytokine cascade elements
3. Cross-reference with:
   - ImmPort database (immune pathways)
   - Reactome (pathway database)
   - ImmunoGRID

**Success Criteria:**
- ≥80% of paths match known immune mechanisms

**Example:**
```
Prediction: Tofacitinib → Sjögren's

Path: Tofacitinib → JAK1 → STAT3 → IFN-γ → Sjögren's

Validation:
✓ Contains JAK-STAT components (JAK1, STAT3)
✓ Contains cytokine (IFN-γ)
✓ Matches Reactome pathway: R-HSA-449147 (JAK-STAT signaling)
✓ Confirmed in ImmPort as autoimmune mechanism

Result: VALIDATED ✓
```

---

**Validation Summary Table:**

| Layer | Type | Success Criteria | What It Proves |
|-------|------|------------------|----------------|
| 1. RepoDB | Computational | AUPRC ≥ 0.92 | Better than baselines |
| 2. Disease-Holdout | Transfer | ≥60% hit rate | True knowledge transfer |
| 3. Temporal | Future prediction | ≥40% hit rate | Real-world applicability |
| 4. Literature | Evidence-based | ≥70% support | Mechanistically sound |
| 5. Immune Pathway | Biological | ≥80% match | Autoimmune-specific |

**Overall Confidence:**
- Pass 3+ layers: **MEDIUM confidence**
- Pass 4+ layers: **HIGH confidence**
- Pass all 5 layers: **VERY HIGH confidence** → Experimental priority

**Visualization in Architecture:**
```
┌────────────────────────────────────────────┐
│       5-Layer Validation Framework         │
│                                            │
│  Layer 1: RepoDB Benchmark                 │
│           AUPRC, AUROC, Precision@K        │
│                                            │
│  Layer 2: Zero-Shot Disease-Holdout        │
│           Hit rate in top-20               │
│                                            │
│  Layer 3: Temporal Validation              │
│           Future discovery prediction      │
│                                            │
│  Layer 4: Literature Path Validation       │
│           PubMed evidence for paths        │
│                                            │
│  Layer 5: Immune Pathway Analysis          │
│           Match to known mechanisms        │
│                                            │
│  → Confidence Scoring: 3-5 layers pass     │
└────────────────────────────────────────────┘
```

---

### **PHASE 6: OUTPUT & DEPLOYMENT**

---

#### **STEP 12: Generate Final Outputs**
**What:** Compile all results into deliverable formats

**Deliverables:**

**1. Ranked Drug Candidates**
```
Sjögren's Syndrome - Top 20 Drug Candidates

Rank | Drug           | Score | Confidence | Validation Layers
-----|----------------|-------|------------|------------------
1    | Rituximab      | 0.94  | HIGH       | 5/5 ✓✓✓✓✓
2    | Tocilizumab    | 0.89  | HIGH       | 5/5 ✓✓✓✓✓
3    | Baricitinib    | 0.87  | HIGH       | 4/5 ✓✓✓✓
4    | Abatacept      | 0.84  | MEDIUM     | 3/5 ✓✓✓
...
20   | Drug_X         | 0.68  | LOW        | 2/5 ✓✓
```

**2. Mechanistic Path Reports**
- For each drug-disease pair
- Top-5 paths with attention weights
- Literature evidence
- Biological interpretation

**3. Visualization Files**
- Path diagrams (Drug → ... → Disease)
- Attention heatmaps
- Validation score plots

**4. Research Paper**
- Introduction
- Methods
- Results
- Discussion
- Supplementary materials

**5. Code Repository**
- Data processing scripts
- Model implementation
- Training code
- Evaluation scripts
- Documentation

**Output Locations:**
```
/outputs/
  ├── predictions/
  │   ├── sjogrens_top20.csv
  │   ├── systemic_sclerosis_top20.csv
  │   └── ...
  ├── paths/
  │   ├── sjogrens_mechanistic_paths.json
  │   └── ...
  ├── visualizations/
  │   ├── attention_heatmaps.png
  │   ├── path_diagrams.png
  │   └── validation_scores.png
  ├── validation/
  │   ├── repodb_results.csv
  │   ├── literature_validation.csv
  │   └── summary_statistics.txt
  └── paper/
      ├── manuscript.pdf
      ├── supplementary.pdf
      └── figures/
```

---

## **COMPLETE ARCHITECTURE SUMMARY**

### **Why Each Step Matters:**

| Step | What | Why Essential |
|------|------|---------------|
| 1 | Load DRKG | Base knowledge graph |
| 2 | Extract Autoimmune Subgraph | Focus on relevant domain, reduce computation |
| 3 | Enrich with Orphanet+HPO | +607% coverage for rare diseases |
| 4 | Generate Embeddings | Convert text to numbers for neural network |
| 5 | Build Immune BRG | Add autoimmune-specific regulatory knowledge |
| 6 | Define Metapaths | Constrain to biologically meaningful paths |
| 7 | Build BioPathNet++ | Path-based reasoning architecture |
| 8 | Multi-Disease Training | Learn shared autoimmune mechanisms |
| 9 | Transfer to Rare Diseases | Apply learned knowledge to sparse data |
| 10 | Extract Paths | Provide interpretable explanations |
| 11 | Multi-Layer Validation | Ensure predictions are trustworthy |
| 12 | Generate Outputs | Deliver actionable results |

---

## **NOVEL CONTRIBUTIONS**

What makes this different from existing work:

1. ✅ **First autoimmune-specific BioPathNet++**
   - Original: Generic biomedical KG
   - Ours: Tailored immune mechanisms

2. ✅ **Immune-specific metapaths**
   - Original: Generic path templates
   - Ours: TNF-α, JAK-STAT, IL-6 pathways

3. ✅ **Autoimmune Background Regulatory Graph**
   - Original: Generic regulatory network
   - Ours: Cytokine cascades, immune pathways

4. ✅ **Multi-disease transfer learning**
   - Original: Single-disease models
   - Ours: Learn from 4, transfer to 5 rare diseases

5. ✅ **Orphanet + HPO enrichment for autoimmune**
   - Original: No enrichment
   - Ours: +607% coverage for Sjögren's

6. ✅ **5-layer validation framework**
   - Original: RepoDB only
   - Ours: Computational + biological + literature

---

## **EXPECTED RESULTS**

| Metric | Baseline (GAT) | TxGNN | Our Model | Improvement |
|--------|---------------|-------|-----------|-------------|
| RepoDB AUPRC | 0.82 | 0.87 | **0.92-0.95** | +10-15% |
| Zero-shot AUPRC | 0.65 | 0.78 | **0.80-0.85** | +15-20% |
| Sjögren's Hit@20 | 30% | 55% | **≥60%** | +9% |
| Literature Support | N/A | 50% | **≥70%** | +20% |
| Immune Pathway Match | N/A | N/A | **≥80%** | Novel |

---

## **CONCLUSION**

This architecture solves the rare disease drug repurposing problem through:

1. **Path-based reasoning** (not just embeddings)
2. **Transfer learning** (learn from many → apply to few)
3. **Biological grounding** (immune-specific mechanisms)
4. **Interpretability** (mechanistic paths)
5. **Rigorous validation** (5 independent layers)

**The key insight:** 
> Rare diseases share mechanisms with common diseases. By learning shared autoimmune pathways from data-rich diseases, we can predict for data-sparse diseases even with only 14 known associations.

**Clinical Impact:**
> For diseases like Sjögren's Syndrome affecting 4 million Americans with no FDA-approved therapies, this approach identifies high-confidence drug candidates with mechanistic explanations, accelerating the path to clinical trials.

---

**Document Version:** 1.0  
**Last Updated:** January 30, 2026  
**For:** BioPathNet++ Autoimmune Drug Repurposing Capstone Project