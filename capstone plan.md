# Multi-Disease Drug Repurposing Project
## Detailed Timeline & Execution Plan
### December 23, 2024 - September 15, 2025

**Project ID:** PH26_PH_03  
**Team Lead:** TVNS Yatin  
**Team Members:** A Charan KR, Praveen Kumar Yadav, Srijan Jha  
**Guide:** Dr. Priyanka H

---

## TIMELINE OVERVIEW

```
Phase 1: Foundation & Data Pipeline    [Dec 23 - Feb 28]  →  10 weeks
Phase 2: Model Development             [Mar 01 - May 15]  →  11 weeks
Phase 3: Validation & Refinement       [May 16 - Jul 31]  →  11 weeks
Phase 4: Paper Writing & Finalization  [Aug 01 - Sep 15]  →  6.5 weeks
                                                             ───────────
                                                TOTAL:       38.5 weeks
```

---

## PHASE 1: FOUNDATION & DATA PIPELINE
### December 23, 2024 - February 28, 2025 (10 weeks)

---

### **WEEK 1-2: Setup & Data Acquisition** [Dec 23 - Jan 5]

#### Objectives
- Set up development environment
- Acquire and preprocess all datasets
- Establish version control and collaboration workflow

#### Tasks & Assignments

**Dec 23-24 (Mon-Tue): Environment Setup**
- **Yatin:** 
  - Create GitHub repository with proper structure
  - Set up Python environment (PyTorch 2.0+, PyG, DGL)
  - Configure GPU access (Colab Pro/Kaggle/University cluster)
- **Charan:** 
  - Install dependencies: NetworkX, pandas, scikit-learn, matplotlib
  - Set up Jupyter notebooks for exploratory analysis
- **Praveen & Srijan:** 
  - Literature review: Create annotated bibliography of 20 key papers
  - Extract evaluation metrics used in each paper

**Dec 25-29 (Wed-Sun): DRKG Data Loading**
- **Yatin & Charan:**
  - Download DRKG (tsv files from AWS)
  - Load into NetworkX graph structure
  - Verify entity counts (97,238 entities, 5.8M triples)
- **Praveen:**
  - Extract subgraph for target diseases:
    - Training: RA, SLE, MS, Crohn's, T1D
    - Testing: Systemic Sclerosis, PBC, Sjögren's, Myasthenia Gravis, Autoimmune Hepatitis
- **Srijan:**
  - Create disease-centric statistics table (update Tables 1-2 from proposal)
  - Verify DRKG IDs for all diseases

**Dec 30 - Jan 5: RepoDB & Metadata Integration**
- **Charan & Praveen:**
  - Download RepoDB (approved + failed drug-disease pairs)
  - Align RepoDB drug IDs with DRKG compound IDs
  - Create temporal splits: ≤2015 (train), 2016-2020 (val), 2021+ (test)
- **Yatin & Srijan:**
  - Scrape DrugBank for dosage information
  - Extract therapeutic dosage ranges for top 100 drugs in DRKG
  - Create dosage metadata dictionary

#### Deliverables
- ✅ GitHub repo with data loading scripts
- ✅ DRKG subgraph (autoimmune diseases only): ~150K triples
- ✅ RepoDB temporal splits (CSV files)
- ✅ Dosage metadata for 100 drugs
- ✅ EDA notebook with graph statistics

#### Risk Management
- **Risk:** DRKG download issues / missing files
  - **Mitigation:** Use official AWS S3 bucket; backup: contact DRKG authors
  - **Buffer:** +2 days
- **Risk:** RepoDB ID alignment fails (different drug naming)
  - **Mitigation:** Use PubChem/ChEMBL as intermediary mapping
  - **Buffer:** +3 days

---

### **WEEK 3-5: Baseline KGE Implementation** [Jan 6 - Jan 26]

#### Objectives
- Implement TransE/DistMult embeddings
- Train baseline KGE models
- Establish evaluation pipeline

#### Tasks & Assignments

**Jan 6-12 (Mon-Sun): KGE Training**
- **Yatin:**
  - Implement TransE using PyTorch Geometric
  - Embedding dimension: 128, 256 (compare both)
  - Training: 200 epochs, batch size 1024
- **Charan:**
  - Implement DistMult using DGL-KE
  - Same hyperparameters as TransE for fair comparison
- **Praveen & Srijan:**
  - Create train/val/test splits (80/10/10)
  - Implement negative sampling (1:4 ratio)
  - Use RepoDB failures as hard negatives

**Jan 13-19 (Mon-Sun): Evaluation Pipeline**
- **Yatin & Charan:**
  - Implement link prediction evaluation
  - Metrics: AUC-ROC, AUC-PR, MRR, Hits@10, Precision@5/10
- **Praveen:**
  - Create evaluation scripts for disease-holdout testing
  - Test generalization: train on RA → test on Sjögren's
- **Srijan:**
  - Visualize embeddings with t-SNE/UMAP
  - Create disease clustering analysis

**Jan 20-26 (Mon-Sun): Baseline Results & Analysis**
- **All team members:**
  - Run full evaluation on TransE and DistMult
  - Create results tables (compare with literature baselines)
  - Identify failure cases (false positives/negatives)
  - Experiment with different embedding dimensions (64, 128, 256)
  - **Team meeting (Jan 26):** Review results with Dr. Priyanka

#### Deliverables
- ✅ TransE model (saved checkpoint)
- ✅ DistMult model (saved checkpoint)
- ✅ Evaluation pipeline (reusable scripts)
- ✅ Baseline results table (AUC, MRR, P@K)
- ✅ Embedding visualization plots

#### Risk Management
- **Risk:** Poor baseline performance (AUC < 0.70)
  - **Mitigation:** Hyperparameter tuning (learning rate, negative ratio)
  - **Buffer:** +4 days for grid search
- **Risk:** Computational bottleneck (training too slow)
  - **Mitigation:** Use smaller embedding dimensions (64D), reduce epochs
  - **Buffer:** +3 days

---

### **WEEK 6-7: Addressing Dosage Composition Issue** [Jan 27 - Feb 9]

#### Objectives
- Implement dosage-aware drug embeddings
- Add mechanistic pathway filtering
- Solve teacher feedback issue #1

#### Tasks & Assignments

**Jan 27 - Feb 2 (Mon-Sun): Dosage-Aware Features**
- **Yatin & Charan:**
  - Augment drug nodes with dosage vectors
  - Encoding: [log(min_dose), log(max_dose), route_of_admin (one-hot)]
  - Concatenate with KGE embeddings: `[KGE (128D) | Dosage (5D)]`
- **Praveen:**
  - Extract mechanism-of-action (MOA) from DrugBank
  - Create MOA similarity matrix (Jaccard on target proteins)
- **Srijan:**
  - Implement pathway-based post-filter
  - Rule: Prediction valid only if ≥2 shared pathways between train/test disease

**Feb 3-9 (Mon-Sun): Validation of Dosage Module**
- **Yatin:**
  - Re-train TransE with dosage-augmented features
  - Compare with baseline TransE (AUC improvement expected: +0.03-0.05)
- **Charan & Praveen:**
  - Test MOA filtering on RepoDB test set
  - Measure precision improvement (expected: +5-10% P@10)
- **Srijan:**
  - Case study: Methotrexate (RA vs. Cancer)
  - Verify model distinguishes high-dose (cancer) vs. low-dose (RA) uses
- **All team members:**
  - **Team meeting (Feb 9):** Review dosage module with Dr. Priyanka

#### Deliverables
- ✅ Dosage-augmented drug embeddings
- ✅ MOA similarity matrix (CSV)
- ✅ Pathway filtering module (Python script)
- ✅ Case study report (Methotrexate analysis)
- ✅ Updated baseline table with dosage-aware results

#### Risk Management
- **Risk:** Dosage data incomplete (<50 drugs have full info)
  - **Mitigation:** Use median dosage per therapeutic class as fallback
  - **Buffer:** +3 days for data imputation
- **Risk:** Dosage features don't improve performance
  - **Mitigation:** Ablation study to quantify contribution; may remove if negligible
  - **Buffer:** +2 days

---

### **WEEK 8-10: GAT Architecture Design & Initial Training** [Feb 10 - Feb 28]

#### Objectives
- Design multi-disease GAT architecture
- Implement degree-normalized attention
- Conduct initial training and debugging

#### Tasks & Assignments

**Feb 10-16 (Mon-Sun): GAT Implementation**
- **Yatin:**
  - Implement GAT layer using PyTorch Geometric
  - Architecture: 2 GAT layers (128D → 128D → 128D)
  - Multi-head attention: 4 heads per layer
- **Charan:**
  - Implement degree-normalization in attention mechanism
  - Formula: `attention_score / log(degree + 1)`
- **Praveen:**
  - Design shared encoder + disease-specific heads architecture
  - Shared: 2 GAT layers
  - Disease-specific: 1 linear layer per disease (5 diseases → 5 heads)
- **Srijan:**
  - Create disease-centric subgraph extraction function
  - For each disease: extract 3-hop neighborhood from disease node

**Feb 17-23 (Mon-Sun): Unit Testing & Initial Training**
- **All team members:**
  - Unit tests for GAT forward pass
  - Verify attention weights sum to 1
  - Test degree normalization (high-degree nodes should have lower attention)
  - Integration test: KGE embeddings → GAT → prediction scores
  - Begin initial training runs on RA dataset

**Feb 24-28 (Mon-Fri): Debugging & Optimization**
- **Yatin & Charan:**
  - Debug any training issues (gradient flow, loss convergence)
  - Optimize memory usage (mini-batching, gradient checkpointing)
- **Praveen & Srijan:**
  - Monitor training metrics (loss curves, attention distributions)
  - Create visualization dashboard (TensorBoard)
- **Team meeting (Feb 28):** Phase 1 completion review with Dr. Priyanka

#### Deliverables
- ✅ GAT model class (PyTorch module)
- ✅ Degree-normalized attention mechanism
- ✅ Multi-disease architecture diagram
- ✅ Unit test suite (pytest)
- ✅ Initial training results (preliminary performance)
- ✅ Phase 1 summary report for Dr. Priyanka

#### Risk Management
- **Risk:** GAT implementation bugs (attention mechanism errors)
  - **Mitigation:** Use PyG's built-in GATConv as reference; extensive unit testing
  - **Buffer:** +3 days for debugging
- **Risk:** Memory issues with large graphs (5.8M edges)
  - **Mitigation:** Use mini-batch training; sample k-hop subgraphs per disease
  - **Buffer:** +2 days

---

## PHASE 2: MODEL DEVELOPMENT
### March 1 - May 15, 2025 (11 weeks)

---

### **WEEK 11-13: Single-Disease GAT Training** [Mar 1 - Mar 23]

#### Objectives
- Train GAT on individual diseases (no transfer learning yet)
- Establish single-disease performance baselines
- Compare GAT vs. KGE-only

#### Tasks & Assignments

**Mar 1-9 (Sat-Sun): Training Pipeline**
- **Yatin & Charan:**
  - Train single-disease GAT for RA (most data-rich)
  - Hyperparameters:
    - Learning rate: 0.001 (Adam)
    - Dropout: 0.3
    - Weight decay: 1e-5
    - Epochs: 100 (early stopping on validation AUC)
  - Log attention weights for interpretability analysis
- **Praveen:**
  - Train single-disease GAT for SLE, MS, Crohn's
  - Same hyperparameters as RA for fair comparison
- **Srijan:**
  - Create training monitoring dashboard (TensorBoard)
  - Plot loss curves, AUC curves, attention weight distributions

**Mar 10-16 (Mon-Sun): Hyperparameter Tuning**
- **All team members:**
  - Grid search over key hyperparameters:
    - Learning rate: [0.0001, 0.001, 0.01]
    - Dropout: [0.2, 0.3, 0.5]
    - Number of heads: [2, 4, 8]
    - Hidden dimensions: [64, 128, 256]
  - Use Optuna or Ray Tune for efficient search
  - Document best hyperparameters per disease

**Mar 17-23 (Mon-Sun): Evaluation & Comparison**
- **Yatin:**
  - Evaluate all single-disease GAT models
  - Compare with KGE baselines (TransE, DistMult)
  - Expected result: GAT > KGE by +0.05-0.10 AUC
- **Charan & Praveen:**
  - Analyze attention patterns
  - Identify top-10 most attended genes/pathways per disease
  - Validate with literature (PubMed search)
- **Srijan:**
  - Create results table: Single-disease GAT vs. KGE
  - Statistical significance testing (paired t-test on 5-fold CV)
- **Team meeting (Mar 23):** Review single-disease results

#### Deliverables
- ✅ 5 trained single-disease GAT models (RA, SLE, MS, Crohn's, T1D)
- ✅ Performance comparison table (GAT vs. KGE)
- ✅ Attention analysis report (top genes/pathways)
- ✅ Training logs (TensorBoard files)
- ✅ Hyperparameter tuning report

#### Risk Management
- **Risk:** GAT underperforms KGE (AUC difference < 0.02)
  - **Mitigation:** Hyperparameter search; try deeper GAT (3-4 layers); increase heads
  - **Contingency:** If GAT still underperforms, revert to KGE-only + post-hoc path extraction
  - **Buffer:** +5 days
- **Risk:** Overfitting (train AUC >> val AUC)
  - **Mitigation:** Increase dropout to 0.5; add L2 regularization; use early stopping
  - **Buffer:** +3 days

---

### **WEEK 14-17: Multi-Disease Transfer Learning** [Mar 24 - Apr 20]

#### Objectives
- Implement transfer learning architecture
- Train shared encoder on multiple diseases jointly
- Test knowledge transfer to rare diseases

#### Tasks & Assignments

**Mar 24 - Apr 6 (Mon-Sun): Shared Encoder Training**
- **Yatin:**
  - Implement multi-disease training loop
  - Batch sampling strategy: Equal samples per disease per batch
  - Loss: Weighted sum of per-disease losses
- **Charan:**
  - Implement disease-specific prediction heads
  - Architecture: Shared GAT encoder → 5 disease-specific linear layers
- **Praveen & Srijan:**
  - Create multi-disease training dataset
  - Ensure balanced sampling (RA has 1,695 edges, Sjögren's has 14 → upsample Sjögren's)
  - Experiment with different balancing strategies

**Apr 7-13 (Mon-Sun): Transfer to Rare Diseases**
- **Yatin & Charan:**
  - Fine-tune shared encoder on rare diseases
  - Protocol: Freeze first GAT layer, fine-tune second layer + disease head
  - Training: 20 epochs with small learning rate (1e-4)
  - Experiment with different fine-tuning strategies (full, partial freeze)
- **Praveen:**
  - Evaluate transfer performance
  - Metric: Hit rate (% of known drugs in top-20 predictions)
  - Target: ≥60% hit rate for rare diseases
- **Srijan:**
  - Compare transfer learning vs. training-from-scratch on rare diseases
  - Expected: Transfer learning AUC > from-scratch AUC by +0.15-0.25

**Apr 14-20 (Mon-Sun): Ablation Studies**
- **All team members:**
  - Ablation 1: Remove transfer learning (train on rare disease only) → measure AUC drop
  - Ablation 2: Remove degree normalization → measure AUC drop
  - Ablation 3: Remove dosage features → measure AUC drop
  - Ablation 4: Remove pathway filtering → measure precision drop
  - Ablation 5: Vary number of training diseases (2, 3, 4, 5) → measure transfer quality
  - Create comprehensive ablation table showing contribution of each component
- **Team meeting (Apr 20):** Mid-project review with Dr. Priyanka

#### Deliverables
- ✅ Multi-disease GAT model (shared encoder + 5 disease heads)
- ✅ Transfer learning evaluation (hit rate table for rare diseases)
- ✅ Comprehensive ablation study results
- ✅ Mid-project report (12-15 pages) for Dr. Priyanka

#### Risk Management
- **Risk:** Poor transfer to rare diseases (hit rate < 40%)
  - **Mitigation:** Implement few-shot meta-learning (MAML); augment rare disease data with synthetic pathways
  - **Buffer:** +6 days
- **Risk:** Negative transfer (multi-disease model worse than single-disease)
  - **Mitigation:** Add adversarial domain discriminator; increase disease-specific capacity
  - **Buffer:** +5 days

---

### **WEEK 18-21: Few-Shot Meta-Learning (MAML)** [Apr 21 - May 15]

#### Objectives
- Implement MAML for ultra-rare diseases
- Address data scarcity issue
- Improve rare disease hit rate to ≥70%

#### Tasks & Assignments

**Apr 21-27 (Mon-Sun): MAML Implementation**
- **Yatin:**
  - Implement MAML training loop
  - Inner loop: K-shot adaptation (K=5 examples from rare disease)
  - Outer loop: Meta-gradient update on query set
  - Use learn2learn library for stable implementation
- **Charan:**
  - Create episodic training sampler
  - Each episode: Sample support set (5 drug-disease pairs) + query set (10 pairs)
  - Sample from both data-rich and data-sparse diseases
- **Praveen & Srijan:**
  - Implement task distribution for meta-training
  - Tasks: {RA, SLE, MS, Crohn's, T1D, Systemic Sclerosis, PBC, Sjögren's}
  - Each task = predict drug-disease links for one disease

**Apr 28 - May 4 (Mon-Sun): Meta-Training**
- **Yatin & Charan:**
  - Meta-train MAML model
  - Hyperparameters:
    - Inner LR (alpha): 0.01
    - Outer LR (beta): 0.001
    - Inner steps: 5
    - Meta-epochs: 50
  - Experiment with first-order MAML (faster, similar performance)
- **Praveen:**
  - Monitor meta-training curves (task-specific losses)
  - Ensure all diseases benefit from meta-learning
  - Track generalization gap (train vs val performance per task)
- **Srijan:**
  - Implement meta-validation protocol
  - Sample new tasks (unseen diseases) for validation

**May 5-11 (Mon-Sun): MAML Evaluation**
- **All team members:**
  - Evaluate MAML on rare diseases with 5-shot, 10-shot, 20-shot adaptation
  - Compare: MAML vs. transfer learning vs. training-from-scratch
  - Expected: MAML > transfer by +0.05-0.10 AUC on rare diseases
  - Create results table with statistical significance tests
  - Analyze which diseases benefit most from MAML

**May 12-15 (Mon-Thu): Alternative Meta-Learning Approaches**
- **Yatin & Charan:**
  - If MAML underperforms, try Prototypical Networks (simpler)
  - Implement matching networks as backup
- **Praveen & Srijan:**
  - Compare all meta-learning approaches
  - Select best performing method for final model
- **Team meeting (May 15):** Phase 2 completion review

#### Deliverables
- ✅ MAML model (meta-trained checkpoint)
- ✅ Few-shot evaluation results (5-shot, 10-shot, 20-shot)
- ✅ Comparison table: MAML vs. Prototypical vs. transfer vs. from-scratch
- ✅ Task-specific learning curves
- ✅ Phase 2 completion report (18-20 pages)

#### Risk Management
- **Risk:** MAML implementation complexity (bugs in meta-gradients)
  - **Mitigation:** Use existing MAML library (learn2learn); extensive debugging
  - **Buffer:** +5 days
- **Risk:** MAML doesn't improve over transfer learning
  - **Mitigation:** Try Prototypical Networks (simpler alternative); increase inner steps
  - **Contingency:** Stick with standard transfer learning if MAML fails
  - **Buffer:** +4 days

---

## PHASE 3: VALIDATION & REFINEMENT
### May 16 - July 31, 2025 (11 weeks)

---

### **WEEK 22-24: Interpretability & Path Extraction** [May 16 - Jun 5]

#### Objectives
- Implement mechanistic path extraction
- Add attention-based explainability
- Generate interpretable predictions for top candidates

#### Tasks & Assignments

**May 16-22 (Fri-Thu): Path Extraction Module**
- **Yatin:**
  - Implement BFS-based path extraction
  - For each top-10 prediction, find 3-5 shortest paths: Drug → Disease
  - Path scoring: Combine edge weights + GAT attention scores
- **Charan:**
  - Implement metapath-guided search
  - Prioritize biologically meaningful paths:
    - Drug → inhibits → Gene → regulates → Pathway → associates → Disease
    - Drug → agonist → Gene → participates → BioProcess → upregulates → Disease
- **Praveen:**
  - Create path filtering rules
  - Discard paths with low-confidence edges (attention < 0.1)
  - Prefer paths with ≥2 known immune/inflammatory relations
- **Srijan:**
  - Visualize top paths using NetworkX + Graphviz
  - Create interactive visualizations (Plotly/Cytoscape.js)

**May 23-29 (Fri-Thu): Attention Analysis**
- **All team members:**
  - Extract and visualize attention weights for all predictions
  - Create attention heatmaps (drug-gene, gene-pathway, pathway-disease)
  - Identify most influential nodes/edges across diseases
  - Compare attention patterns between similar diseases (RA vs. PBC)

**May 30 - Jun 5 (Fri-Thu): Literature Validation**
- **Yatin & Charan:**
  - Automated PubMed search for top-10 predictions per disease
  - Query: "[Drug] AND [Disease] AND (mechanism OR pathway OR treatment)"
  - Count papers supporting each prediction
  - Extract relevant snippets from abstracts
- **Praveen:**
  - Manual curation of top-5 predictions for Sjögren's Syndrome
  - Classify support level: Strong (≥3 papers) / Moderate (1-2) / None (0)
  - Deep dive into mechanisms for high-confidence predictions
- **Srijan:**
  - Create literature validation table
  - Expected: ≥50% of top-10 predictions have moderate/strong support
  - Identify truly novel predictions (0 papers) for further discussion

#### Deliverables
- ✅ Path extraction module (Python scripts)
- ✅ Top-10 predictions with mechanistic paths (CSV + visualizations)
- ✅ Attention analysis report (heatmaps, influential nodes)
- ✅ Literature validation table (per-disease support levels)
- ✅ Interactive visualization dashboard

#### Risk Management
- **Risk:** Too many low-quality paths (noisy predictions)
  - **Mitigation:** Stricter filtering (require attention > 0.2); limit to top-3 paths
  - **Buffer:** +3 days
- **Risk:** Low literature support (<30% of predictions)
  - **Mitigation:** This is expected for novel predictions; highlight as "new hypotheses"
  - **Buffer:** +2 days

---

### **WEEK 25-27: RepoDB Benchmarking** [Jun 6 - Jun 26]

#### Objectives
- Comprehensive evaluation on RepoDB
- Compare with state-of-the-art baselines
- Temporal validation (≤2015 → >2020)

#### Tasks & Assignments

**Jun 6-12 (Fri-Thu): Full RepoDB Evaluation**
- **Yatin:**
  - Evaluate on RepoDB approved pairs (6,677 drugs)
  - Metrics: AUC-ROC, AUC-PR, MRR, Hits@1/5/10, Precision@5/10
  - Run evaluation with 5-fold cross-validation
- **Charan:**
  - Evaluate on RepoDB failed trials (4,123 pairs)
  - Ensure model ranks failures lower than approvals
  - Compute failure detection rate (% of failures ranked below median)
- **Praveen:**
  - Compare with published baselines:
    - DREAMwalk (Bang et al. 2023): AUC ~0.82
    - HGTDR (Gharizadeh et al. 2024): AUC ~0.87
    - AdaDR (Ye et al. 2024): AUC ~0.89
  - Target: Your model AUC ≥ 0.85 (competitive)
- **Srijan:**
  - Create comprehensive results table
  - Statistical significance testing (bootstrap 95% CI, paired t-tests)
  - Generate comparison plots (bar charts, ROC curves)

**Jun 13-19 (Fri-Thu): Temporal Validation**
- **Yatin & Charan:**
  - Train model on ≤2015 DRKG snapshot
  - Test on drug approvals 2016-2024 (from FDA database)
  - Metric: Hit rate in top-50 predictions per disease
  - Analyze performance degradation over time
- **Praveen:**
  - Analyze temporal generalization failures
  - Identify drug classes that transferred poorly
  - Hypothesis: Biologics (antibodies) may transfer worse than small molecules
  - Investigate covariate shift issues
- **Srijan:**
  - Create temporal validation plot (hit rate by year)
  - Compare: ≤2015→2016-2020 vs. ≤2015→2021-2024
  - Statistical analysis of temporal trends

**Jun 20-26 (Fri-Thu): Error Analysis**
- **All team members:**
  - Deep dive into false positives (predicted but failed in trials)
  - Deep dive into false negatives (missed known associations)
  - Identify systematic biases (drug class, disease type, graph structure)
  - Create error taxonomy and analysis report
- **Team meeting (Jun 26):** Benchmarking results review

#### Deliverables
- ✅ RepoDB evaluation results (full metrics table with 95% CI)
- ✅ Comparison with SOTA (bar chart + significance tests)
- ✅ Temporal validation results (hit rate table + trend plots)
- ✅ Failure analysis report (systematic error patterns)
- ✅ Error taxonomy (categorized false positives/negatives)

#### Risk Management
- **Risk:** Model underperforms SOTA (AUC < 0.80)
  - **Mitigation:** Hyperparameter re-tuning; ensemble multiple models; add more features
  - **Contingency:** Emphasize interpretability and rare disease transfer as novel contributions
  - **Buffer:** +6 days
- **Risk:** Poor temporal generalization (hit rate < 30%)
  - **Mitigation:** Analyze covariate shift; retrain with domain adaptation techniques
  - **Buffer:** +4 days

---

### **WEEK 28-30: Hub Gene Bias & Robustness Analysis** [Jun 27 - Jul 17]

#### Objectives
- Quantify hub gene bias in predictions
- Implement degree-preserving randomized baselines
- Ensure model learns biological signal, not topology
- Test robustness to various perturbations

#### Tasks & Assignments

**Jun 27 - Jul 3 (Fri-Thu): Degree-Preserving Baselines**
- **Yatin:**
  - Generate randomized DRKG with preserved degree distribution
  - Algorithm: Edge-swap Monte Carlo (10,000 swaps)
  - Train GAT on randomized graph
  - Measure prediction overlap with real model
- **Charan:**
  - Compute prediction overlap between real vs. randomized model
  - If overlap > 50% → model is learning topology, not biology
  - Calculate degree-bias score for top-K predictions
- **Praveen & Srijan:**
  - Identify top-20 hub genes (TNF, IL-6, IL-10, etc.)
  - Measure their contribution to predictions (attention weights)
  - Create hub gene contribution table
  - Analyze if hub genes are biologically justified

**Jul 4-10 (Fri-Thu): Bias Mitigation Validation**
- **Yatin:**
  - Compare degree-normalized GAT vs. standard GAT
  - Measure hub gene attention reduction (expected: 30-50% lower)
  - Evaluate impact on overall performance
- **Charan:**
  - Evaluate both models on RepoDB
  - Ensure degree normalization doesn't hurt overall AUC (<0.02 drop acceptable)
  - Test on rare diseases specifically
- **Praveen & Srijan:**
  - Case study: Identify predictions that changed after degree normalization
  - Manually validate if new predictions are biologically plausible
  - Literature search for top changed predictions

**Jul 11-17 (Fri-Thu): Robustness Testing**
- **All team members:**
  - Test 1: Edge dropout (randomly remove 10%, 20%, 30% of edges)
  - Test 2: Node feature noise (add Gaussian noise to embeddings)
  - Test 3: Adversarial perturbations (targeted edge additions/removals)
  - Test 4: Cross-database validation (train on DRKG, test on Hetionet overlaps)
  - Measure performance degradation under each perturbation
  - Create robustness analysis report
- **Team meeting (Jul 17):** Bias and robustness review

#### Deliverables
- ✅ Randomized baseline results (AUC on random graph)
- ✅ Hub gene contribution analysis (attention weight tables)
- ✅ Degree normalization comparison (standard vs. normalized GAT)
- ✅ Case study report (prediction differences)
- ✅ Comprehensive robustness analysis report
- ✅ Sensitivity analysis (performance vs. perturbation strength)

#### Risk Management
- **Risk:** High overlap with random baseline (>60%)
  - **Mitigation:** This indicates failure; implement stronger regularization or switch to causal GNN
  - **Contingency:** Reframe as "degree-aware" model; emphasize interpretability over novelty
  - **Buffer:** +5 days
- **Risk:** Degree normalization hurts performance significantly (AUC drop > 0.05)
  - **Mitigation:** Use softer normalization (log(degree + 10) instead of log(degree + 1))
  - **Buffer:** +3 days

---

### **WEEK 31-33: LINCS L1000 Transcriptomic Validation** [Jul 18 - Jul 31]

#### Objectives
- Validation using gene expression data
- Verify signature reversal for top predictions
- Add orthogonal validation layer

#### Tasks & Assignments

**Jul 18-24 (Fri-Thu): LINCS Data Acquisition & Processing**
- **Yatin & Charan:**
  - Download LINCS L1000 Level 3 data (normalized gene expression)
  - Focus on perturbagens: Small molecules that overlap with DRKG drugs
  - Process and normalize data for consistency
- **Praveen:**
  - Identify disease gene expression signatures
  - Sources: GEO, ArrayExpress (RA, SLE, MS expression studies)
  - Compute differential expression: Disease vs. healthy control
  - Filter for high-confidence differentially expressed genes (FDR < 0.05)
- **Srijan:**
  - Create disease signature profiles (top 100 up/down-regulated genes per disease)
  - Validate signatures against published disease transcriptomes

**Jul 25-31 (Fri-Thu): Signature Reversal Analysis**
- **Yatin:**
  - Compute reversal score for top-10 predictions per disease
  - Formula: -Pearson(drug_signature, disease_signature)
  - Threshold: Reversal score > 0.3 → strong evidence
- **Charan & Praveen:**
  - Rank predictions by reversal score
  - Compare with GAT-based ranking
  - Expected: Positive correlation between GAT score and reversal score
  - Measure Spearman correlation between rankings
- **Srijan:**
  - Create integrated ranking: 0.7 * GAT_score + 0.3 * Reversal_score
  - Evaluate if integrated ranking improves RepoDB AUC
  - Identify predictions with strong reversal + high GAT score (best candidates)
  - Identify predictions with low reversal + high GAT score (investigate why)
- **All team members:**
  - Create LINCS validation report
  - Case study: Rituximab for Sjögren's (known to reverse B-cell signatures)
  - Prepare visualizations (scatter plots, correlation heatmaps)
- **Team meeting (Jul 31):** Phase 3 completion review

#### Deliverables
- ✅ LINCS L1000 processed dataset (drug signatures for 100+ drugs)
- ✅ Disease gene expression signatures (RA, SLE, MS, Crohn's, T1D)
- ✅ Signature reversal scores (top-10 predictions per disease)
- ✅ Integrated ranking evaluation (GAT + reversal)
- ✅ LINCS validation report (8-10 pages)
- ✅ Phase 3 completion report (25-30 pages)

#### Risk Management
- **Risk:** LINCS data unavailable or incompatible
  - **Mitigation:** Use CMap (older version) or skip if data quality is poor
  - **Contingency:** This is optional; proceed without if necessary
  - **Buffer:** +4 days
- **Risk:** Low correlation between GAT and reversal scores
  - **Mitigation:** Expected for novel predictions; emphasize as orthogonal evidence
  - **Buffer:** +2 days

---

## PHASE 4: PAPER WRITING & FINALIZATION
### August 1 - September 15, 2025 (6.5 weeks)

---

### **WEEK 34-36: Paper Writing (Draft 1)** [Aug 1 - Aug 21]

#### Objectives
- Write first complete draft of research paper
- Target: Conference (ISMB, RECOMB) or Journal (Bioinformatics, BMC Bioinformatics)
- Approximately 10-12 pages (conference) or 15-20 pages (journal)

#### Tasks & Assignments

**Aug 1-7 (Fri-Thu): Core Sections I**
- **Yatin (Lead author):**
  - Abstract (250 words)
    - Problem, approach, key results, impact
  - Introduction (2 pages)
    - Drug repurposing motivation
    - Autoimmune disease challenges
    - Transfer learning rationale
    - Our contributions (3-4 key points)
- **Charan:**
  - Related Work (2-3 pages)
    - KGE methods for drug repurposing (TransE, DistMult, ComplEx)
    - GNN methods for biomedical graphs (GAT, GCN, HGT)
    - Transfer learning in drug discovery
    - Multi-disease learning approaches
    - Position our work relative to prior art

**Aug 8-14 (Fri-Thu): Core Sections II**
- **Charan:**
  - Methods (4-5 pages)
    - DRKG data preprocessing and disease subgraph extraction
    - KGE baseline (TransE/DistMult) with dosage augmentation
    - GAT architecture + degree normalization
    - Multi-disease training with shared encoder
    - Transfer learning protocol
    - MAML for few-shot adaptation
    - Path extraction and interpretability
- **Praveen:**
  - Experiments & Results (4-5 pages)
    - Experimental setup (datasets, splits, metrics)
    - RepoDB benchmarking results
    - Comparison with SOTA baselines
    - Disease-holdout transfer evaluation
    - Temporal validation results
    - Ablation studies
    - Hub gene bias analysis
    - LINCS transcriptomic validation
    - Statistical significance tests for all claims

**Aug 15-21 (Fri-Thu): Discussion & Case Studies**
- **Srijan:**
  - Case Studies (2 pages)
    - Sjögren's Syndrome: Top-5 predictions with mechanistic paths
    - Systemic Sclerosis: Top-5 predictions with literature support
    - Novel predictions with strong evidence
    - Validation through attention weights and pathways
  - Discussion (2 pages)
    - Key findings and their implications
    - Biological insights from attention patterns
    - Advantages of multi-disease transfer learning
    - Comparison with single-disease approaches
- **Yatin:**
  - Limitations (1 page)
    - Data sparsity for rare diseases
    - Validation constraints (computational, not clinical)
    - Scope limited to autoimmune diseases
    - Dosage information incompleteness
  - Conclusion & Future Work (1 page)
    - Summary of contributions
    - Clinical implications
    - Future directions (wet-lab validation, expansion to other disease families)

#### Deliverables
- ✅ Complete paper draft 1 (12-15 pages for journal)
- ✅ All sections written with clear structure
- ✅ Figures and tables prepared (8-10 figures, 4-6 tables)

#### Risk Management
- **Risk:** Writing takes longer than expected
  - **Mitigation:** Start with bullet-point outlines; expand later; use LaTeX templates
  - **Buffer:** +4 days built into next section

---

### **WEEK 36-37: Figures, Tables & Supplementary Materials** [Aug 22 - Sep 4]

#### Objectives
- Finalize all figures and tables
- Create comprehensive supplementary materials
- Internal peer review

#### Tasks & Assignments

**Aug 22-28 (Fri-Thu): Figures & Tables**
- **Yatin:**
  - Figure 1: Overall pipeline diagram (DRKG → KGE → GAT → Transfer → Predictions)
  - Figure 2: Multi-disease GAT architecture diagram
  - Table 1: Dataset statistics (DRKG entities, relations, disease coverage)
  - Table 2: Hyperparameters for all models
- **Charan:**
  - Figure 3: RepoDB benchmarking results (bar chart comparing SOTA)
  - Figure 4: Transfer learning performance (hit rate by rare disease)
  - Figure 5: Ablation study results (component contribution)
  - Table 3: Full RepoDB results (AUC, MRR, P@K with 95% CI)
- **Praveen:**
  - Figure 6: Temporal validation results (hit rate over time)
  - Figure 7: Attention heatmap (disease-specific attention patterns)
  - Figure 8: Case study - mechanistic path for top prediction
  - Table 4: Top-10 predictions per rare disease with confidence scores
- **Srijan:**
  - Figure 9: t-SNE visualization of disease embeddings
  - Figure 10: LINCS signature reversal analysis
  - Table 5: Literature validation summary
  - Table 6: Comparison with baselines (comprehensive metrics)
  - Ensure all figures have descriptive captions (2-3 sentences each)

**Aug 29 - Sep 4 (Fri-Thu): Supplementary Materials**
- **Yatin:**
  - Supplementary Table S1: Complete hyperparameter search results
  - Supplementary Table S2: Full dataset statistics (all diseases)
  - Supplementary Figure S1: Training curves (loss and AUC over epochs)
- **Charan:**
  - Supplementary Table S3: Disease-by-disease detailed results
  - Supplementary Table S4: Ablation study extended results
  - Supplementary Figure S2: Additional attention visualizations
- **Praveen:**
  - Supplementary Table S5: Top-20 predictions for all 5 rare diseases
  - Supplementary Table S6: Literature validation details (paper titles, DOIs)
  - Supplementary Figure S3: ROC and PR curves for all models
- **Srijan:**
  - Supplementary Table S7: LINCS reversal scores for all predictions
  - Supplementary Table S8: Hub gene contribution analysis
  - Supplementary Methods: Detailed algorithmic descriptions
  - Appendix: Extended case studies (additional 3-4 examples)
- **All team members:**
  - Internal peer review: Each member reads full draft
  - Check for consistency, clarity, completeness
  - Verify all citations are correct and properly formatted

#### Deliverables
- ✅ 10 main figures (publication-quality, high resolution)
- ✅ 6 main tables (properly formatted)
- ✅ Supplementary materials (10-15 pages)
- ✅ Complete figure captions and table descriptions
- ✅ Internal review completed with feedback document

#### Risk Management
- **Risk:** Figure quality issues (resolution, clarity)
  - **Mitigation:** Use vector graphics (SVG, PDF); professional plotting libraries
  - **Buffer:** +2 days for revisions

---

### **WEEK 38: Revision & Guide Feedback** [Sep 5 - Sep 11]

#### Objectives
- Incorporate Dr. Priyanka's feedback
- Revise and polish paper (Draft 2)
- Prepare for final submission

#### Tasks & Assignments

**Sep 5-6 (Fri-Sat): Guide Review**
- **All team members:**
  - Submit draft 1 to Dr. Priyanka (Sep 5 morning)
  - Schedule review meeting (Sep 6)
  - Take detailed notes on feedback
  - Prioritize revisions (critical vs. optional)

**Sep 7-9 (Sun-Tue): Major Revisions**
- **Yatin:**
  - Revise Introduction based on feedback
  - Strengthen motivation and clearly state contributions
  - Ensure smooth narrative flow
- **Charan:**
  - Revise Methods based on feedback
  - Add missing implementation details
  - Clarify technical descriptions
- **Praveen:**
  - Revise Results based on feedback
  - Add requested analyses or comparisons
  - Ensure statistical rigor in all claims
- **Srijan:**
  - Revise Discussion based on feedback
  - Address limitations and future work suggestions
  - Strengthen clinical relevance arguments

**Sep 10-11 (Wed-Thu): Polishing & Proofreading**
- **All team members:**
  - Proofread entire paper for grammar, typos, clarity
  - Ensure consistent terminology throughout
  - Verify all figure/table references are correct
  - Check reference formatting (consistent style)
  - Ensure paper meets target venue requirements (page limit, format)
  - Final formatting pass (LaTeX compilation, check for overflow, spacing)

#### Deliverables
- ✅ Revised paper draft 2 (incorporating all feedback)
- ✅ Response document addressing each feedback point
- ✅ Camera-ready draft for final submission
- ✅ Peer review within team (each member approves)

#### Risk Management
- **Risk:** Major revisions required (e.g., add new experiments)
  - **Mitigation:** Discuss feasibility with Dr. Priyanka; negotiate scope
  - **Contingency:** Some experiments may be moved to future work if too time-consuming
  - **Buffer:** +3 days

---

### **WEEK 39: Final Submission & Presentation Prep** [Sep 12 - Sep 15]

#### Objectives
- Submit final project report to university
- Prepare final presentation (40-50 slides)
- Prepare for project viva/defense

#### Tasks & Assignments

**Sep 12 (Fri): Final Report Compilation**
- **Yatin:**
  - Compile final project report for university submission
  - Structure: Abstract, Intro, Lit Survey, Methods, Results, Discussion, Conclusion, References
  - Format according to university template (40-50 pages)
  - Add acknowledgments, certificate pages
  - Final LaTeX compilation and PDF generation
- **Charan:**
  - Create project poster (A0 size, for potential poster session)
  - Highlight: Problem, Approach, Key Results, Novel Contributions, Impact
  - Use university branding and templates
- **Praveen & Srijan:**
  - Prepare code repository for submission
  - Add comprehensive README with:
    - Setup instructions
    - Dependencies and environment
    - Usage examples
    - Reproduction instructions
  - Clean up code, add docstrings, remove debug prints
  - Create requirements.txt and environment.yml
  - Add LICENSE file

**Sep 13-14 (Sat-Sun): Presentation Preparation**
- **All team members:**
  - Create final presentation (PowerPoint/Beamer, 40-50 slides)
  - Structure:
    - Title & Team (1 slide)
    - Problem & Motivation (5 slides)
    - Background & Related Work (5 slides)
    - Our Approach & Methods (12 slides)
    - Experimental Setup (3 slides)
    - Results & Validation (12 slides)
    - Case Studies (4 slides)
    - Discussion & Insights (3 slides)
    - Limitations & Future Work (2 slides)
    - Conclusion & Impact (2 slides)
    - Acknowledgments & Questions (1 slide)
  - Prepare backup slides (additional details if asked)
  - Practice presentation individually (each member presents their sections)
  - Record timing (target: 30-35 minutes + 15 min Q&A)
  - Prepare for potential questions:
    - Technical details (architecture, hyperparameters)
    - Validation strategy justification
    - Comparison with other methods
    - Clinical applicability
    - Future directions

**Sep 15 (Mon): Final Submission & Defense Prep**
- **Yatin:**
  - Submit final report to university portal (morning)
  - Submit paper to target conference/journal (if approved by guide)
  - Archive all materials (code, data, figures, results)
- **All team members:**
  - Final presentation rehearsal (with timer, Q&A practice)
  - Review potential questions and prepare answers
  - Ensure all team members can answer questions about any section
  - **Project defense (tentative: Sep 15 afternoon or later based on schedule)**

#### Deliverables
- ✅ Final project report (40-50 pages) submitted to university
- ✅ Research paper submitted to conference/journal (optional but encouraged)
- ✅ Final presentation (40-50 slides + backup slides)
- ✅ Code repository (GitHub with full documentation)
- ✅ Project poster (A0 size, publication-ready)
- ✅ Demo video (5-10 minutes, optional but recommended)
- ✅ Archived materials (all data, code, results, figures)
- ✅ Project defense completed successfully

#### Risk Management
- **Risk:** Presentation goes over time
  - **Mitigation:** Strict time management; practice with timer; prepare "quick version"
  - **Buffer:** +1 day for additional practice
- **Risk:** Technical questions we can't answer
  - **Mitigation:** Comprehensive Q&A preparation; honest "we'll investigate further" responses
  - **Buffer:** N/A (preparation is key)

---

## EXTENDED TIMELINE BENEFITS

### Additional Time Allocation (July → September extension)

The extended timeline (38.5 weeks vs. original 29 weeks) provides:

1. **More Comprehensive Validation** (+2 weeks)
   - LINCS L1000 transcriptomic validation (was optional, now standard)
   - Robustness testing under various perturbations
   - Cross-database validation

2. **Deeper Analysis** (+3 weeks)
   - Hub gene bias analysis with randomized baselines
   - Extended ablation studies (more configurations)
   - Detailed error analysis and failure case investigation

3. **Better Paper Quality** (+2 weeks)
   - More time for writing and revisions
   - Multiple internal review cycles
   - Comprehensive supplementary materials

4. **Increased Buffer Time** (+2.5 weeks)
   - Total buffer: ~70 days (10 weeks) across all phases
   - Handles unexpected issues without schedule slip
   - Allows for exploration of alternative approaches if main path fails

---

## WORKLOAD DISTRIBUTION

### Yatin (Team Leader)
- **Primary Responsibilities:**
  - Overall project coordination and timeline management
  - GAT/MAML implementation (core model development)
  - Paper writing (lead author, introduction, abstract, conclusion)
  - Weekly progress tracking and risk management
  - Liaison with Dr. Priyanka
- **Estimated Hours:** 25-30 hours/week
- **Peak Periods:** Weeks 11-21 (model development), Weeks 34-39 (paper writing)

### Charan
- **Primary Responsibilities:**
  - KGE implementation (DistMult, dosage features)
  - Evaluation pipeline and benchmarking
  - Methods section writing
  - Code documentation and repository management
- **Estimated Hours:** 20-25 hours/week
- **Peak Periods:** Weeks 3-7 (baselines), Weeks 25-27 (benchmarking), Weeks 36-37 (figures)

### Praveen
- **Primary Responsibilities:**
  - Data preprocessing (DRKG, RepoDB, temporal splits)
  - Transfer learning experiments and evaluation
  - Results analysis and statistical testing
  - Experiments section writing
- **Estimated Hours:** 20-25 hours/week
- **Peak Periods:** Weeks 14-17 (transfer learning), Weeks 25-27 (validation), Weeks 34-36 (results writing)

### Srijan
- **Primary Responsibilities:**
  - Visualization (t-SNE, attention plots, pathway diagrams)
  - Literature validation and PubMed search
  - Case studies and interpretability analysis
  - Supplementary materials preparation
- **Estimated Hours:** 20-25 hours/week
- **Peak Periods:** Weeks 22-24 (interpretability), Weeks 36-37 (supplementary materials)

---

## MEETING SCHEDULE

### Weekly Team Meetings
- **Day:** Every Sunday, 10:00 AM - 12:00 PM
- **Location:** In-person (university) or Zoom
- **Agenda:**
  - Progress updates (15 min/person)
  - Blockers and problem-solving (30-40 min)
  - Plan for next week (15 min)
  - Code review session (20-30 min)
- **Deliverable:** Weekly meeting notes (shared Google Doc)

### Biweekly Guide Meetings
- **Day:** Every alternate Friday, 4:00 PM - 5:00 PM
- **Location:** Dr. Priyanka's office or online
- **Agenda:**
  - Demo of recent progress (code, results, visualizations)
  - Discussion of technical challenges
  - Feedback on approach and direction
  - Approval for next phase activities
- **Deliverable:** Meeting minutes with action items

### Major Milestone Reviews
- **Jan 26:** Baseline KGE results review
- **Feb 28:** Phase 1 completion review
- **Mar 23:** Single-disease GAT results
- **Apr 20:** Mid-project review (formal presentation to guide)
- **May 15:** MAML results and Phase 2 completion review
- **Jun 26:** Benchmarking results review
- **Jul 17:** Bias analysis review
- **Jul 31:** Phase 3 completion review (comprehensive evaluation)
- **Sep 6:** Paper draft review (critical feedback session)
- **Sep 14:** Final rehearsal before defense

---

## BUFFER TIME ALLOCATION

### Built-in Buffers by Phase
- **Phase 1:** +12 days (for data issues, KGE bugs, GAT implementation)
- **Phase 2:** +23 days (for MAML complexity, transfer learning failures, hyperparameter tuning)
- **Phase 3:** +22 days (for validation issues, low performance, robustness testing)
- **Phase 4:** +10 days (for revisions, presentation prep, unexpected feedback)
- **Total Buffer:** ~70 days (~10 weeks)

### Buffer Usage Strategy
- **Minor delays (1-2 days):** Use daily buffers, don't escalate
- **Moderate delays (3-5 days):** Use weekly buffers, inform team in weekly meeting
- **Major delays (>5 days):** Use phase buffers, escalate to guide, potentially re-scope
- **Critical delays (>2 weeks):** Activate contingency plans (see below)

### Contingency Plans

#### If >2 weeks behind schedule:
- Skip optional tasks (LINCS validation, HGT implementation)
- Reduce number of rare diseases from 5 to 3 (focus on Sjögren's, Systemic Sclerosis, PBC)
- Simplify ablation studies (test only critical components)

#### If >4 weeks behind schedule:
- Reduce scope to single-disease high-performance model
- Skip MAML (use only standard transfer learning)
- Target workshop/short paper instead of full conference paper

#### If GAT consistently underperforms:
- Pivot to KGE-only approach with extensive path extraction
- Reframe as "interpretable drug repurposing" (emphasize explainability over accuracy)
- Focus on mechanistic validation rather than benchmark performance

#### If transfer learning fails:
- Pivot to multi-disease joint learning (no explicit transfer, just shared patterns)
- Use disease similarity-based predictions (if disease A and B are similar, drugs for A might work for B)
- Emphasize data analysis and biomedical insights over methodological novelty

---

## SUCCESS CRITERIA (FINAL EVALUATION)

### Minimum Viable Product (MVP)
To pass the capstone and graduate:
- ✅ Multi-disease GAT model trained on 4-5 autoimmune diseases
- ✅ AUC ≥ 0.80 on RepoDB benchmark
- ✅ Transfer learning demonstrated (hit rate ≥ 50% for rare diseases)
- ✅ Top-10 predictions per disease with at least one mechanistic path each
- ✅ Complete project report (40+ pages) and presentation (40 slides)
- ✅ Successful defense with satisfactory answers to panel questions

### Target Goals (Strong Performance)
To achieve distinction/high marks:
- ✅ AUC ≥ 0.85 on RepoDB (competitive with SOTA)
- ✅ Transfer learning hit rate ≥ 60% for rare diseases
- ✅ ≥50% of top-10 predictions have literature support (moderate or strong)
- ✅ Comprehensive ablation studies showing contribution of each component
- ✅ Research paper submitted to peer-reviewed conference or journal
- ✅ Evidence of novelty (new drug-disease candidates with biological plausibility)

### Stretch Goals (Publication-Quality)
To publish in top venue:
- ✅ AUC ≥ 0.87 on RepoDB (exceeds current SOTA)
- ✅ MAML implementation with significant improvement (hit rate ≥ 70%)
- ✅ LINCS L1000 transcriptomic validation with positive correlation
- ✅ ≥5 novel drug repurposing candidates per rare disease with strong evidence
- ✅ Paper accepted to top-tier venue (ISMB, RECOMB, Bioinformatics, Nature Communications)
- ✅ Wet-lab collaborators interested in experimental validation
- ✅ Code released as open-source tool with documentation

---

## CRITICAL PATH ANALYSIS

### Must-Complete Tasks (Cannot Skip)
These are on the critical path - any delay here delays final deadline:
1. DRKG data loading and disease subgraph extraction
2. TransE/DistMult baseline implementation and evaluation
3. GAT architecture implementation
4. Single-disease GAT training and validation
5. Multi-disease training loop with shared encoder
6. Transfer learning to rare diseases
7. RepoDB benchmarking
8. Mechanistic path extraction
9. Paper writing (all core sections)
10. Final presentation preparation

### Optional Tasks (Can Skip if Behind Schedule)
These add value but aren't critical:
- Few-shot meta-learning (MAML) - nice to have, not essential
- LINCS L1000 transcriptomic validation - adds validation layer but optional
- HGT implementation - potential improvement but GAT sufficient
- Degree-preserving randomized baselines - strengthens claims but not required
- Detailed hub gene bias analysis - interesting but not core contribution
- Cross-database validation - adds robustness but time-consuming

### Parallel Workstreams (To Maximize Efficiency)
- **Weeks 1-2:** Yatin & Charan (DRKG + env setup) || Praveen & Srijan (Literature review + RepoDB)
- **Weeks 3-5:** Yatin (TransE) || Charan (DistMult) || Praveen & Srijan (Evaluation pipeline + splits)
- **Weeks 6-7:** Yatin & Charan (Dosage module) || Praveen (MOA extraction) || Srijan (Pathway filtering)
- **Weeks 11-13:** Yatin & Charan (GAT training) || Praveen (Evaluation) || Srijan (Visualization)
- **Weeks 14-17:** Yatin (Multi-disease training) || Charan (Disease heads) || Praveen (Transfer eval) || Srijan (Comparison plots)
- **Weeks 34-36:** Yatin (Intro/Abstract/Conclusion) || Charan (Methods) || Praveen (Results) || Srijan (Case studies/Discussion)

---

## RISK REGISTER (COMPREHENSIVE)

| Risk ID | Risk Description | Probability | Impact | Mitigation Strategy | Owner | Status |
|---------|------------------|------------|--------|---------------------|-------|--------|
| R1 | DRKG download failure or data corruption | Low | High | Use official AWS S3; verify checksums; contact authors | Yatin | Monitor |
| R2 | RepoDB ID alignment issues | Medium | Medium | Use PubChem/ChEMBL mapping; manual curation if needed | Charan | Monitor |
| R3 | GAT underperforms KGE baselines | Medium | High | Hyperparameter tuning; deeper architecture; ensemble | Yatin | Mitigate |
| R4 | Poor transfer to rare diseases (<40% hit rate) | High | High | MAML; synthetic data augmentation; pathway-based transfer | All | Mitigate |
| R5 | Hub gene bias dominates predictions | High | Medium | Degree normalization; randomized baselines; causal filtering | Yatin | Mitigate |
| R6 | Low literature support (<30%) | Medium | Low | Expected for novel predictions; emphasize hypothesis generation | Srijan | Accept |
| R7 | MAML implementation bugs | Medium | Medium | Use learn2learn library; extensive testing; fallback to Prototypical Networks | Yatin | Mitigate |
| R8 | Computational bottlenecks | Low | Medium | Mini-batch training; GPU acceleration; use Colab Pro | Charan | Monitor |
| R9 | Data leakage in evaluation | Medium | Critical | Strict temporal splits; careful validation protocol | Praveen | Mitigate |
| R10 | Team member unavailability (illness, personal) | Low | High | Cross-training; documentation; flexible task reassignment | Yatin | Mitigate |
| R11 | Model doesn't beat SOTA (<0.80 AUC) | Medium | High | Emphasize interpretability + rare disease transfer as contributions | All | Accept |
| R12 | Paper rejected from first submission | Medium | Low | Target multiple venues; revise based on feedback; resubmit | Yatin | Accept |
| R13 | Guide requests major changes to approach | Low | Critical | Regular check-ins to align expectations; early validation of direction | Yatin | Mitigate |
| R14 | Hardware failure (lost work, data) | Low | High | Daily commits to GitHub; cloud storage; regular backups | All | Mitigate |
| R15 | Scope creep (trying to do too much) | Medium | High | Strict adherence to timeline; prioritize MVP; defer nice-to-haves | Yatin | Mitigate |

---

## TOOLS & INFRASTRUCTURE

### Development Environment
- **Language:** Python 3.9+
- **Deep Learning:** PyTorch 2.0+, PyTorch Geometric 2.3+, DGL 1.1+
- **Graph Processing:** NetworkX 3.0+, igraph, graph-tool
- **Data Processing:** pandas 2.0+, NumPy 1.24+, scikit-learn 1.3+
- **Visualization:** Matplotlib 3.7+, Seaborn 0.12+, Plotly 5.15+, Cytoscape.js
- **Meta-Learning:** learn2learn 0.2+
- **Optimization:** Optuna 3.0+ (hyperparameter tuning)
- **Version Control:** Git + GitHub (private repo initially, public after defense)
- **Collaboration:** Google Drive (shared docs), Slack/Discord (daily communication)
- **Paper Writing:** Overleaf (LaTeX, IEEE/ACM/Bioinformatics templates)
- **Presentation:** PowerPoint / Google Slides / Beamer (LaTeX)

### Computational Resources
- **Primary:** Google Colab Pro ($10/month per member, V100/A100 GPU, 25GB RAM)
- **Backup:** Kaggle Notebooks (P100 GPU, 16GB RAM, free)
- **Alternative:** University HPC cluster (if available, request access early)
- **Local Development:** Personal laptops (for code writing, debugging, visualization)
- **Storage:** 
  - Google Drive (100GB per member, shared folders)
  - GitHub (unlimited for public repos, code + small data)
  - AWS S3 (DRKG download only)

### Datasets & Resources
- **DRKG:** AWS S3 bucket (official, ~2GB compressed)
- **RepoDB:** GitHub repository (Brown & Patel 2017, ~10MB)
- **DrugBank:** Free academic license (register with university email)
- **LINCS L1000:** GEO repository (select cell lines, ~5GB)
- **PubMed:** NCBI E-utilities API (rate limit: 3 requests/second)
- **GEO/ArrayExpress:** Gene expression datasets for disease signatures

### Code Organization
```
project_root/
├── data/
│   ├── raw/              # Original DRKG, RepoDB
│   ├── processed/        # Cleaned, split datasets
│   └── metadata/         # Dosage, MOA, annotations
├── src/
│   ├── models/           # GAT, KGE, MAML implementations
│   ├── training/         # Training loops, optimization
│   ├── evaluation/       # Metrics, benchmarking scripts
│   ├── interpretability/ # Path extraction, attention analysis
│   └── utils/            # Data loading, preprocessing
├── experiments/
│   ├── baselines/        # TransE, DistMult results
│   ├── gat/              # Single-disease GAT results
│   ├── transfer/         # Multi-disease transfer results
│   └── maml/             # Few-shot learning results
├── notebooks/
│   ├── eda/              # Exploratory data analysis
│   ├── visualization/    # Plots, figures for paper
│   └── analysis/         # Error analysis, case studies
├── paper/
│   ├── main.tex          # Main paper
│   ├── supplementary.tex # Supplementary materials
│   ├── figures/          # Publication-quality figures
│   └── tables/           # LaTeX tables
├── presentation/
│   ├── slides.pptx       # Final presentation
│   └── demo/             # Demo materials
├── tests/
│   └── unit/             # Unit tests for all modules
├── README.md
├── requirements.txt
└── environment.yml
```

---

## COMMUNICATION PROTOCOLS

### Daily Communication
- **Platform:** Slack/Discord team channel
- **Expected:** Brief daily update from each member (what I did, what I'm doing, any blockers)
- **Time:** Before 9 PM each day (asynchronous, no meeting required)

### Code Review Process
- **When:** Before merging any feature branch to main
- **Who:** At least one other team member must review
- **What:** Check for bugs, readability, documentation, test coverage
- **Tool:** GitHub Pull Requests with review comments

### Documentation Standards
- **Code:** Docstrings for all functions (Google style)
- **Experiments:** Jupyter notebooks with markdown explanations
- **Results:** CSV files with metadata (date, hyperparameters, metrics)
- **Meetings:** Shared Google Doc with agenda, notes, action items

### Emergency Escalation
- **Minor blocker (<1 day):** Post in Slack, team helps debug
- **Major blocker (>1 day):** Schedule emergency team meeting
- **Critical blocker (>3 days, affects timeline):** Escalate to Dr. Priyanka immediately

---

## QUALITY ASSURANCE

### Code Quality Checklist
- [ ] All functions have docstrings
- [ ] Critical functions have unit tests
- [ ] Code follows PEP 8 style guide
- [ ] No hardcoded paths (use config files)
- [ ] Random seeds fixed for reproducibility
- [ ] Memory usage optimized (no unnecessary copies)
- [ ] GPU memory freed after use

### Experiment Quality Checklist
- [ ] Hyperparameters logged (config file or experiment tracker)
- [ ] Random seeds logged for reproducibility
- [ ] Results saved with timestamp and git commit hash
- [ ] Visualizations have clear labels, legends, titles
- [ ] Statistical tests performed where appropriate
- [ ] Results compared against baselines

### Paper Quality Checklist
- [ ] All claims supported by experiments
- [ ] All figures referenced in text
- [ ] All tables have descriptive captions
- [ ] All equations numbered and explained
- [ ] No undefined acronyms (check first use)
- [ ] References formatted consistently
- [ ] Grammar and spelling checked (Grammarly)
- [ ] Meets venue formatting requirements
- [ ] Supplementary materials complete

---

## EXPECTED OUTCOMES & DELIVERABLES

### Academic Deliverables
1. **Project Report** (40-50 pages)
   - Complete technical documentation
   - All experiments, results, analysis
   - Formatted per university template
   - Submitted to university portal

2. **Research Paper** (12-15 pages)
   - Manuscript for conference/journal submission
   - Follows venue guidelines (ISMB, RECOMB, Bioinformatics)
   - Includes supplementary materials
   - Peer-reviewed quality

3. **Presentation** (40-50 slides)
   - Comprehensive project overview
   - Clear visualization of results
   - 30-35 minute presentation + Q&A
   - Backup slides for technical details

4. **Poster** (A0 size, optional)
   - Visual summary of project
   - For poster sessions or conferences
   - High-quality graphics

### Technical Deliverables
1. **Trained Models**
   - TransE/DistMult baselines (checkpoints)
   - Single-disease GAT models (5 diseases)
   - Multi-disease transfer learning model
   - MAML meta-learned model (if successful)
   - All with saved hyperparameters and training logs

2. **Code Repository** (GitHub)
   - Clean, documented, reproducible code
   - Installation instructions
   - Usage examples and tutorials
   - Trained model weights (via releases)
   - MIT or Apache 2.0 license

3. **Results & Data**
   - All experimental results (CSV files)
   - Top-K predictions for all rare diseases
   - Mechanistic paths (graph files)
   - Attention weights and visualizations
   - Literature validation data

### Intellectual Contributions
1. **Novel Methods**
   - Dosage-aware drug embeddings
   - Degree-normalized GAT for biomedical graphs
   - Multi-disease transfer learning for drug repurposing
   - Mechanistic path extraction with attention

2. **Novel Findings**
   - 5-10 high-confidence drug repurposing candidates per rare disease
   - Shared biological pathways across autoimmune diseases
   - Attention pattern insights (which genes/pathways are most predictive)
   - Comparison of meta-learning strategies for rare diseases

3. **Datasets**
   - Curated autoimmune disease subgraph from DRKG
   - Dosage metadata for 100+ drugs
   - Disease gene expression signatures
   - Temporal splits for retrospective validation

---

## POST-PROJECT OPPORTUNITIES

### Short-Term (After Defense)
1. **Paper Submission & Revision**
   - Submit to target conference/journal (Aug-Sep 2025)
   - Address reviewer comments (Oct-Nov 2025)
   - Revised submission (Dec 2025)

2. **Open-Source Release**
   - Make GitHub repo public
   - Add comprehensive tutorials
   - Promote on Twitter/LinkedIn/Reddit (r/MachineLearning, r/bioinformatics)

3. **Conference Presentation**
   - If paper accepted, present at conference (2026)
   - Network with researchers in drug repurposing
   - Explore collaboration opportunities

### Long-Term (Career Development)
1. **MS/PhD Applications**
   - Strong portfolio project for graduate school apps
   - Demonstrates research ability, technical skills
   - Potential for recommendation letters from Dr. Priyanka

2. **Industry Positions**
   - Showcases expertise in AI + biology
   - Relevant for biotech, pharma, health-tech companies
   - Drug discovery startups (Recursion, Insitro, etc.)

3. **Wet-Lab Collaboration**
   - Partner with experimental biology labs
   - Validate top predictions in vitro/in vivo
   - Potential for high-impact publication (Nature/Science)

4. **Startup Opportunity**
   - Drug repurposing platform based on this work
   - Expand to other disease families (cancer, cardiovascular)
   - Seek funding (grants, angel investors, accelerators)

---

## FINAL NOTES & ADVICE

### For Team Leader (Yatin)
- **Your role:** Coordinator, not dictator. Facilitate, don't micromanage.
- **Weekly check-ins:** Ensure everyone is on track; identify blockers early.
- **Conflict resolution:** Address issues immediately; don't let them fester.
- **Risk management:** Monitor risks weekly; activate mitigation plans proactively.
- **Guide relationship:** Regular updates to Dr. Priyanka; seek advice early, not late.

### For All Team Members
- **Communication is key:** Over-communicate rather than under-communicate.
- **Document everything:** Future you (and your teammates) will thank you.
- **Ask for help early:** If stuck >4 hours, reach out. Don't suffer in silence.
- **Celebrate small wins:** Each milestone is progress. Acknowledge it.
- **Stay flexible:** If something isn't working, pivot quickly. Don't be stubborn.
- **Focus on impact:** Every task should contribute to final paper/presentation quality.

### Time Management Tips
- **Deep work blocks:** 3-4 hour focused sessions (no distractions)
- **Pomodoro technique:** 25 min work + 5 min break for sustained productivity
- **Friday review:** Every Friday, review week's progress and plan next week
- **Avoid burnout:** Take one full day off per week (Sunday recommended)
- **Sleep well:** 7-8 hours/night. Tired brain = slow progress.

### Technical Best Practices
- **Commit often:** Daily git commits, even if code isn't perfect
- **Experiment tracking:** Use WandB or TensorBoard for all training runs
- **Reproducibility:** Fix random seeds everywhere (Python, NumPy, PyTorch)
- **Sanity checks:** Always validate results with simple checks (e.g., train AUC should be high)
- **Iterative development:** Start simple, then add complexity gradually

### Writing Tips
- **Start early:** Don't wait until August to start writing
- **Write as you go:** Document methods, results immediately after experiments
- **Storytelling:** Paper should tell a clear story: Problem → Approach → Results → Impact
- **Figures first:** Create all figures before writing text (easier to write around visuals)
- **Read examples:** Study 5-10 papers from target venues to understand style

---

## CONCLUSION

This timeline provides a **systematic, risk-aware, and achievable** plan for completing your Multi-Disease Drug Repurposing project from December 23, 2024, to September 15, 2025.

### Key Success Factors
1. **Extended timeline** (38.5 weeks) with ample buffer time (~10 weeks)
2. **Balanced workload** across 4 team members with clear role definitions
3. **Built-in risk management** with contingency plans for all major risks
4. **Regular checkpoints** with guide (biweekly) and team (weekly)
5. **Flexible scope** (MVP vs. target vs. stretch goals) allowing adaptation
6. **Comprehensive validation** (RepoDB, transfer, temporal, LINCS) for strong paper
7. **Adequate writing time** (6.5 weeks) for high-quality paper and defense

### Final Reminder
- **Communication is critical:** Weekly team meetings, biweekly guide meetings, daily Slack updates
- **Document everything:** Code comments, experiment logs, decision rationales, meeting notes
- **Stay flexible:** If something isn't working after 1 week, pivot quickly (don't be stubborn)
- **Focus on core contributions:** Multi-disease transfer learning + interpretability + rare disease focus
- **Manage scope:** Better to do fewer things excellently than many things poorly
- **Take care of yourselves:** Burnout helps no one. Sustainable pace wins the marathon.

### You Have Everything You Need
- **Strong foundation:** Phase 1 proposal approved, clear problem definition
- **Robust dataset:** DRKG + RepoDB provide ample data
- **Solid methodology:** KGE + GAT + Transfer Learning is well-motivated
- **Clear validation:** Multi-layer evaluation strategy is rigorous
- **Adequate time:** 38.5 weeks with 10 weeks buffer is generous
- **Capable team:** 4 motivated students with complementary skills
- **Supportive guide:** Dr. Priyanka is invested in your success

**You will succeed.** Stick to the plan, communicate regularly, adapt when necessary, and you will deliver an excellent capstone project and high-quality research paper that you'll be proud of.

**Good luck, team! Now let's build something amazing.**

---

**Document Version:** 2.0 (Extended Timeline)  
**Last Updated:** December 21, 2024  
**Next Review:** January 5, 2025 (after Week 2 completion)  
**Final Review:** September 1, 2025 (pre-defense check)