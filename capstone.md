# Multi-Disease Drug Repurposing Project
## Issues Analysis & Solutions Document

**Project ID:** PH26_PH_03  
**Team:** TVNS Yatin (Leader), A Charan KR, Praveen Kumar Yadav, Srijan Jha  
**Guide:** Dr. Priyanka H  
**Document Date:** December 21, 2024

---

## 1. TEACHER FEEDBACK: Drug Composition Problem

### Issue Identified by Teachers
**"Composition of the drug: 2 same chemical drugs treating disease X and Y doesn't mean X and Y are the same. The composition and weights of certain chemicals in the drug are different."**

### Detailed Analysis
This is a **critical conceptual issue** that affects the validity of drug repurposing predictions:

1. **The Problem:**
   - The same drug molecule (e.g., Methotrexate) may be FDA-approved for multiple diseases
   - However, the **dosage, formulation, administration route, and treatment protocol** differ significantly
   - Example: 
     - Methotrexate for Rheumatoid Arthritis: 7.5-25 mg/week (oral)
     - Methotrexate for Cancer: 100-12,000 mg/m² (IV infusion)
   - Simply predicting "Drug X treats Disease Y" without considering dosage/formulation is **clinically misleading**

2. **Why This Matters for Your Project:**
   - DRKG contains drug-disease "treats" edges but **does not encode dosage information**
   - Your model might predict a drug-disease association that is technically correct but **practically irrelevant** if the required dose is toxic or infeasible
   - Two diseases sharing the same drug doesn't imply biological similarity unless the **mechanism of action** at that dose is shared

### Solutions & Mitigations

#### Solution 1: Add Drug Dosage Context (Recommended)
**Implementation:**
- Augment DRKG with dosage metadata from DrugBank/DailyMed
- Create dosage-aware embeddings:
  - Instead of: `Drug → treats → Disease`
  - Use: `Drug@Dosage_Range → treats → Disease`
- During inference, flag predictions where target disease requires significantly different dosing than training diseases

**Technical Approach:**
```python
# Pseudocode for dosage-aware drug nodes
drug_node_features = {
    'drug_id': 'DB00563',  # Methotrexate
    'base_embedding': KGE_embedding,
    'dosage_profile': {
        'RA': {'min': 7.5, 'max': 25, 'unit': 'mg/week'},
        'Cancer': {'min': 100, 'max': 12000, 'unit': 'mg/m²'}
    }
}
```

#### Solution 2: Mechanism-Based Filtering
**Implementation:**
- Add a post-prediction filter that validates:
  1. **Target protein overlap**: Does the drug target the same proteins in both diseases?
  2. **Pathway similarity**: Are the biological pathways affected similar?
- Use DRKG relations: `Drug → inhibits → Gene → participates → Pathway → associates → Disease`

**Validation Criteria:**
```
For a prediction Drug X → Disease Y to be valid:
✓ At least 2 shared protein targets with training diseases
✓ At least 1 shared inflammatory/immune pathway
✓ Similar mechanism of action (agonist/antagonist/inhibitor)
```

#### Solution 3: Literature-Based Dosage Verification
**Implementation:**
- For top-K predictions (K=10-20), perform automated PubMed search:
  - Query: "[Drug Name] AND [Disease Name] AND (dosage OR dose OR administration)"
- Flag predictions with **no literature evidence** of similar dosing

#### Solution 4: Multi-Modal Drug Representation
**Implementation:**
- Combine multiple drug representations:
  1. **KGE embedding** (relational context from DRKG)
  2. **Morgan fingerprints** (chemical structure)
  3. **Dosage vector** (therapeutic range encoding)
  4. **Target profile** (protein binding affinity scores)

**Why This Helps:**
- Diseases sharing similar drug **structures** and **dosages** will cluster together
- Prevents false transfer learning between mechanistically different uses

---

## 2. ADDITIONAL CRITICAL ISSUES IDENTIFIED

### Issue 2.1: Data Leakage in Validation

#### Problem
- Using RepoDB for evaluation risks **future-knowledge leakage**
- If your training data includes post-2015 drug approvals and you test on RepoDB, the model may have seen the answers

#### Solution
**Strict Temporal Split Protocol:**
```
Training Set: Drug-disease associations approved ≤ 2015
Validation Set: Associations approved 2016-2020
Test Set: Associations approved 2021-2024

Source: DrugCentral with FDA approval dates
```

**Implementation:**
1. Filter DRKG to remove all edges with timestamp > 2015
2. Train model on historical snapshot
3. Test predictions against "future" discoveries
4. Report **hit rate** (% of true discoveries in top-K predictions)

---

### Issue 2.2: Hub Gene Bias

#### Problem
- GAT attention will over-weight highly connected genes (e.g., TNF, IL-6)
- These "hub" genes appear in almost every autoimmune disease
- Model may predict drug-disease associations based on **degree centrality** rather than **causal mechanisms**

#### Solution A: Degree-Normalized Attention
**Modify GAT attention mechanism:**
```python
# Standard GAT
attention_score = softmax(LeakyReLU(a^T [Wh_i || Wh_j]))

# Degree-normalized GAT
degree_penalty = log(degree(node_j) + 1)
attention_score = softmax(LeakyReLU(a^T [Wh_i || Wh_j]) / degree_penalty)
```

#### Solution B: Degree-Preserving Baselines
**Create randomized control baselines:**
1. **Degree-preserving random graph**: Shuffle DRKG edges while preserving node degrees
2. Train model on randomized graph
3. True performance = (Your model AUC) - (Random baseline AUC)

This ensures your model learns **biological signal**, not just graph topology.

---

### Issue 2.3: Negative Sample Selection

#### Problem
- DRKG only contains **positive** drug-disease associations (approved treatments)
- How do you define "negative" examples (drug-disease pairs that **don't work**)?
- Random negatives are unrealistic (most unknown pairs are simply **untested**, not necessarily ineffective)

#### Solution: Curated Negative Sampling
**Use RepoDB's clinical trial failures:**
- RepoDB contains 4,123 **failed** drug-disease pairs (Phase II/III failures)
- These are **true negatives** with clinical evidence

**Sampling Strategy:**
```
Positive samples: DRKG "treats" edges (6,677 approved pairs)
Hard negatives: RepoDB failures (4,123 failed trials)
Easy negatives: Random drug-disease pairs from different therapeutic areas
  (e.g., cancer drug + autoimmune disease)

Training ratio: 1 positive : 1 hard negative : 2 easy negatives
```

---

### Issue 2.4: Interpretability vs. Accuracy Trade-off

#### Problem
- KGE (TransE, DistMult) provides interpretable embeddings but **lower accuracy**
- GAT provides high accuracy but **black-box predictions**
- Need to balance both

#### Solution: Hybrid Interpretable-Predictive Pipeline

**Architecture:**
```
Stage 1: KGE Pre-training
  ↓ (biologically meaningful embeddings)
Stage 2: GAT Fine-tuning (frozen KGE as node features)
  ↓ (attention-based prediction)
Stage 3: Path Extraction (post-hoc explainability)
  ↓
Output: Ranked predictions + Mechanistic paths
```

**Path Extraction Algorithm:**
1. For each top prediction `Drug X → Disease Y`:
2. Use BFS/DFS to find shortest paths in DRKG
3. Prioritize paths with:
   - High attention weights (from GAT layer)
   - Known immune/inflammatory relations
4. Output: `Drug → inhibits → Gene → regulates → Pathway → associates → Disease`

---

### Issue 2.5: Class Imbalance (Rare Disease Scarcity)

#### Problem
- Rare autoimmune diseases have **very few training examples**:
  - Sjögren's Syndrome: 14 total relationships (5 drugs, 9 genes)
  - Systemic Sclerosis: 173 relationships
- Standard supervised learning will **fail** on such sparse data

#### Solution A: Few-Shot Meta-Learning (MAML)
**Use Model-Agnostic Meta-Learning:**
```python
# MAML Training Loop
for episode in meta_training_episodes:
    # Sample a "support set" from data-rich disease
    support_RA = sample_k_shot(RA_data, k=10)
    
    # Fast adaptation (few gradient steps)
    theta_adapted = theta - alpha * grad(loss(support_RA, theta))
    
    # Evaluate on "query set" from rare disease
    query_Sjogrens = sample_k_shot(Sjogrens_data, k=5)
    meta_loss = loss(query_Sjogrens, theta_adapted)
    
    # Meta-update
    theta = theta - beta * grad(meta_loss, theta)
```

**Why This Works:**
- Model learns to **adapt quickly** from few examples
- Training on RA/SLE teaches "how to learn about autoimmune diseases"
- Fine-tuning on 5-10 Sjögren's examples achieves good performance

#### Solution B: Data Augmentation via Pathway Expansion
**Synthetic edge generation:**
1. Identify shared pathways between RA (data-rich) and Sjögren's (data-sparse)
2. If RA has: `Drug A → inhibits → TNF-α → regulates → Inflammation`
3. And Sjögren's has: `TNF-α → associates → Sjögren's`
4. Infer: `Drug A → (synthetic edge) → Sjögren's` with lower confidence weight

**Validation:** Only count synthetic edges if supported by ≥2 independent pathways

---

### Issue 2.6: Overfitting to Well-Studied Diseases

#### Problem
- 70% of DRKG drug-disease edges involve top 10 most-studied diseases
- Model may learn "disease-specific shortcuts" rather than generalizable patterns

#### Solution: Adversarial Disease Discriminator

**Add auxiliary task:**
```
Main task: Predict drug-disease associations
Adversarial task: Predict which disease a sample belongs to

Loss = L_prediction - λ * L_disease_discrimination
```

**Intuition:**
- Force the shared encoder to learn **disease-agnostic** features
- If the encoder embeddings can't distinguish RA from Sjögren's, it has learned **shared mechanisms**
- Similar to domain adaptation techniques in computer vision

---

## 3. RISK MANAGEMENT & MITIGATION STRATEGIES

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| **Insufficient rare disease data** | High | High | Few-shot meta-learning (MAML) + pathway augmentation |
| **Hub gene bias** | High | Medium | Degree-normalized GAT + randomized baselines |
| **Dosage composition issue** | Medium | High | Dosage-aware embeddings + mechanism filtering |
| **Data leakage** | Medium | Critical | Strict temporal splits (≤2015 train, >2020 test) |
| **Poor transfer learning** | Medium | High | Adversarial domain adaptation + multi-disease encoder |
| **Non-reproducible results** | Low | Medium | Fixed random seeds + ablation studies |
| **Computational bottleneck** | Low | Low | Start with GAT (efficient), scale to HGT only if needed |

### Contingency Plans

**If Transfer Learning Fails (<50% improvement over single-disease):**
- Fallback: Train separate models per disease cluster
- Use disease similarity matrix (pathway overlap) to guide clustering

**If GAT Underperforms KGE-only baselines:**
- Revert to pure KGE (DistMult/TransE) with post-hoc path extraction
- GAT may not be necessary if KGE achieves adequate AUC (>0.80)

**If Rare Disease Predictions Have Low Confidence:**
- Report as "hypothesis generation" rather than clinical recommendations
- Require ≥3 independent validation pathways for any rare disease prediction

---

## 4. VALIDATION STRATEGY (MULTI-LAYER)

### Layer 1: Benchmark Evaluation
- **Dataset:** RepoDB (6,677 approved + 4,123 failed)
- **Metrics:** AUC-ROC, AUC-PR, Precision@5, Precision@10, MRR
- **Baseline comparisons:**
  - Random prediction
  - Degree-based ranking
  - TransE/DistMult/ComplEx (KGE-only)
  - Single-disease GAT (no transfer learning)

### Layer 2: Disease-Holdout Transfer Test
- **Protocol:**
  - Train on {RA, SLE, MS, Crohn's}
  - Test on {Systemic Sclerosis, PBC, Sjögren's, Myasthenia Gravis}
- **Success metric:** ≥60% of known drugs for rare diseases appear in top-20 predictions

### Layer 3: Temporal Robustness
- **Protocol:**
  - Train on ≤2015 data
  - Test on 2016-2024 discoveries
- **Success metric:** Hit rate ≥40% in top-50 predictions

### Layer 4: Mechanistic Validation
- **For top-10 predictions per disease:**
  1. Extract top-3 shortest paths: Drug → Gene → Pathway → Disease
  2. Literature search: "[Drug] AND [Disease] AND (pathway OR mechanism)"
  3. Classify as:
     - **Strong support:** ≥3 peer-reviewed papers
     - **Moderate support:** 1-2 papers
     - **Unsupported:** 0 papers
- **Success metric:** ≥50% of predictions have moderate or strong literature support

### Layer 5: Transcriptomic Signature Reversal (Optional)
- **If LINCS L1000 data available:**
  - Compare drug perturbation signature vs. disease expression signature
  - Reversal score = -correlation(drug_signature, disease_signature)
  - Threshold: Reversal score > 0.3 indicates potential efficacy

---

## 5. SUMMARY OF SOLUTIONS

### Immediate Actions (Phase 1: Jan-Feb 2025)
1. ✅ Implement dosage-aware drug embeddings
2. ✅ Add degree-normalized GAT attention
3. ✅ Create RepoDB-based negative sampling
4. ✅ Set up temporal data splits (≤2015 vs. >2020)

### Medium-Term Actions (Phase 2: Mar-Apr 2025)
5. ✅ Implement few-shot meta-learning (MAML)
6. ✅ Add mechanistic pathway extraction module
7. ✅ Build adversarial domain discriminator for transfer learning
8. ✅ Create degree-preserving random baselines

### Long-Term Actions (Phase 3: May-Jun 2025)
9. ✅ Integrate LINCS L1000 transcriptomic validation
10. ✅ Scale from GAT to Heterogeneous GNN (HGT)
11. ✅ Perform ablation studies on all components
12. ✅ Write paper with comprehensive validation results

---

## 6. EXPECTED OUTCOMES

### Measurable Success Criteria
1. **Benchmarking:** AUC ≥ 0.85 on RepoDB test set (≥0.10 improvement over KGE baseline)
2. **Transfer Learning:** Rare disease hit rate ≥ 60% (known drugs in top-20)
3. **Temporal Validation:** Hit rate ≥ 40% for post-2020 discoveries
4. **Interpretability:** ≥70% of top-10 predictions have extractable mechanistic paths
5. **Novelty:** Identify ≥5 high-confidence repurposing candidates per rare disease with no prior clinical testing

### Deliverables
- ✅ Trained multi-disease GAT model
- ✅ Ranked drug predictions for 5 rare autoimmune diseases
- ✅ Mechanistic pathway visualizations
- ✅ Comprehensive validation report
- ✅ Research paper draft (target: bioinformatics conference/journal)

---

**Document Version:** 1.0  
**Last Updated:** December 21, 2024  
**Next Review:** January 15, 2025