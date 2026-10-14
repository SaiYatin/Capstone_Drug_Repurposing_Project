"""
BioPathNet++ v3
===============

Path-based (NBFNet / BioPathNet-style) zero-shot drug repurposing for rare
autoimmune diseases, rebuilt after the October 2026 audit.

Modules
  kg        : compact knowledge-graph format, per-batch graph construction (leakage control)
  build_*   : converters from PrimeKG (kg.csv) and DRKG (enriched_drkg.tsv) to the compact format
  model     : BioPathNet++ (NBFNet core with relation-wise sparse propagation)
  metrics   : full-ranking metrics, negative tiers, bootstrap CIs
  baselines : popularity, autoimmune-frequency, disease-similarity kNN
  train     : training / evaluation CLI (CPU or CUDA, resumable for Kaggle)
"""

__version__ = "3.0.0"
