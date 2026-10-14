# =============================================================================
# BioPathNet++ Retraining on Kaggle
# =============================================================================
# 
# SETUP INSTRUCTIONS:
# 
# Step 1: Create a Kaggle Dataset
#   - Go to kaggle.com/datasets → New Dataset
#   - Name it "biopathnet-data"
#   - Upload these files/folders from your laptop:
#     c:\D_Games\capstone\implementations\datasets\processed\pyg_graph.pt         (~292 MB)
#     c:\D_Games\capstone\implementations\datasets\processed\entity_embeddings.npy (~248 MB)
#     c:\D_Games\capstone\implementations\datasets\processed\enriched_entity2id.json
#     c:\D_Games\capstone\implementations\datasets\processed\enriched_relation2id.json
#     c:\D_Games\capstone\implementations\datasets\processed\train_pairs.csv
#     c:\D_Games\capstone\implementations\datasets\processed\val_pairs.csv
#     c:\D_Games\capstone\implementations\datasets\processed\test_pairs.csv
#     c:\D_Games\capstone\implementations\checkpoints\best_model.pt              (~53 MB)
#     c:\D_Games\capstone\implementations\checkpoints\config.json
#   Total upload: ~600 MB
#
# Step 2: Create a Kaggle Notebook  
#   - Go to kaggle.com/code → New Notebook
#   - Settings → Accelerator → GPU T4 x2 (or P100)
#   - Add your dataset: + Add Data → search "biopathnet-data"
#   - Paste this entire script into one cell and run
#
# Step 3: Download results
#   - After training, outputs will be in /kaggle/working/
#   - Download best_model_v2.pt and results/
# =============================================================================

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import json
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm import tqdm
import time

# ============================
# PATHS (Kaggle layout)
# ============================
DATA_DIR = Path("/kaggle/input/biopathnet-data")  # Your uploaded dataset
OUT_DIR = Path("/kaggle/working")                  # Output directory
OUT_DIR.mkdir(exist_ok=True)

# Check GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# ============================
# 1. NBFNet (copied from models/nbfnet.py — paste your version here)
# ============================
# NOTE: You need to paste the NBFNet class here since Kaggle can't import from your local modules.
# If your nbfnet.py is small, paste the whole thing. Otherwise, upload it as part of the dataset.

# For now, we'll try importing it:
sys.path.insert(0, str(DATA_DIR))

# If you uploaded nbfnet.py to the dataset:
try:
    from nbfnet import NBFNet
    print("✓ Imported NBFNet from dataset")
except ImportError:
    print("⚠ NBFNet not found in dataset. Paste NBFNet class below.")
    # ---- PASTE YOUR NBFNet CLASS HERE IF IMPORT FAILS ----
    raise ImportError("Please upload nbfnet.py to your Kaggle dataset")


# ============================
# 2. Fixed BioPathNet++ Model (v2)
# ============================
class MetapathExtractor(nn.Module):
    def __init__(self, hidden_dim, num_metapaths=7):
        super().__init__()
        self.num_metapaths = num_metapaths
        self.hidden_dim = hidden_dim
        self.path_encoders = nn.ModuleList([
            nn.GRU(hidden_dim, hidden_dim, batch_first=True) for _ in range(num_metapaths)
        ])
        self.metapath_embeddings = nn.Embedding(num_metapaths, hidden_dim)

    def forward(self, node_features, paths_per_metapath):
        metapath_reprs = []
        for idx, paths in enumerate(paths_per_metapath):
            if len(paths) == 0:
                metapath_reprs.append(torch.zeros(self.hidden_dim, device=node_features.device))
                continue
            path_features = torch.stack([node_features[p] for p in paths])
            _, h_n = self.path_encoders[idx](path_features)
            mr = h_n.squeeze(0).mean(dim=0)
            mr = mr + self.metapath_embeddings(torch.tensor([idx], device=node_features.device)).squeeze(0)
            metapath_reprs.append(mr)
        return torch.stack(metapath_reprs)


class MetapathAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, metapath_reprs, query_repr):
        mp = metapath_reprs.unsqueeze(0)
        q = query_repr.unsqueeze(0).unsqueeze(0)
        attn_out, attn_w = self.attention(q, mp, mp)
        out = self.layer_norm(attn_out.squeeze(0).squeeze(0) + query_repr)
        return out, attn_w.squeeze(0).squeeze(0)


class BRGEncoder(nn.Module):
    def __init__(self, hidden_dim, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True,
                          dropout=0.1 if num_layers > 1 else 0)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        if len(x) == 0:
            return torch.zeros(self.gru.hidden_size, device=x.device)
        _, h_n = self.gru(x)
        return self.layer_norm(h_n[-1].mean(dim=0))


class BioPathNetPlusPlusV2(nn.Module):
    """Fixed BioPathNet++ with per-candidate metapaths, active BRG, per-candidate fusion."""

    def __init__(self, num_nodes, num_relations, hidden_dim=256, num_nbf_layers=3,
                 num_metapaths=7, num_attention_heads=4, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.num_relations = num_relations
        self.hidden_dim = hidden_dim

        self.nbfnet = NBFNet(num_nodes, num_relations, hidden_dim, num_nbf_layers, dropout)
        self.metapath_extractor = MetapathExtractor(hidden_dim, num_metapaths)
        self.metapath_attention = MetapathAttention(hidden_dim, num_attention_heads)
        self.brg_encoder = BRGEncoder(hidden_dim, 2)

        # FIX 1: Learned per-metapath projections
        self.metapath_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
            for _ in range(num_metapaths)
        ])
        # FIX 2: BRG neighborhood projection
        self.brg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim)
        )

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.LayerNorm(hidden_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, node_features, edge_index, edge_type, query_drug,
                candidate_diseases, metapaths=None, brg_paths=None):
        nbfnet_features = self.nbfnet(node_features, edge_index, edge_type, query_drug)
        drug_repr = nbfnet_features[query_drug]
        disease_reprs = nbfnet_features[candidate_diseases]

        scores = []
        for i, disease_repr in enumerate(disease_reprs):
            disease_idx = candidate_diseases[i]

            # Per-candidate metapaths
            mp_reprs = self._metapaths_for_pair(nbfnet_features, query_drug, disease_idx)
            weighted_mp, _ = self.metapath_attention(mp_reprs, drug_repr)

            # Active BRG context
            brg_ctx = self._brg_context(nbfnet_features, edge_index, disease_idx)

            # Per-candidate fusion
            fused = self.fusion(torch.cat([drug_repr, weighted_mp, brg_ctx], dim=0))
            score = self.predictor(torch.cat([fused, disease_repr], dim=0))
            scores.append(score)

        return torch.cat(scores), None

    def _metapaths_for_pair(self, node_features, drug_idx, disease_idx):
        pair = torch.cat([node_features[drug_idx], node_features[disease_idx]])
        return torch.stack([proj(pair) for proj in self.metapath_projections])

    def _brg_context(self, node_features, edge_index, disease_idx):
        mask = (edge_index[0] == disease_idx) | (edge_index[1] == disease_idx)
        nbrs = torch.cat([edge_index[0][mask], edge_index[1][mask]]).unique()
        if len(nbrs) == 0:
            return torch.zeros(self.hidden_dim, device=node_features.device)
        return self.brg_projection(node_features[nbrs].mean(dim=0))


# ============================
# 3. Data Loading
# ============================
class DrugDiseaseDataset(Dataset):
    def __init__(self, pairs_file, entity2id):
        self.pairs = pd.read_csv(pairs_file)
        self.entity2id = entity2id
        print(f"  Loaded {len(self.pairs)} pairs from {Path(pairs_file).name}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs.iloc[idx]
        drug_entity = row['head'] if 'Compound' in str(row['head']) else row['tail']
        disease_entity = row['tail'] if 'Disease' in str(row['tail']) else row['head']
        drug_idx = self.entity2id.get(drug_entity, 0)
        disease_idx = self.entity2id.get(disease_entity, 0)
        label = int(row['label']) if 'label' in row.index else 1
        return drug_idx, disease_idx, label


def load_data():
    """Load all data from Kaggle dataset."""
    print("Loading data...")

    # Graph
    graph = torch.load(DATA_DIR / "pyg_graph.pt", weights_only=False, map_location=device)
    print(f"  Graph: {graph.num_nodes:,} nodes, {graph.edge_index.size(1):,} edges")

    # Embeddings
    emb = np.load(DATA_DIR / "entity_embeddings.npy")
    node_features = torch.tensor(emb, dtype=torch.float32).to(device)
    print(f"  Embeddings: {node_features.shape}")

    # Entity mapping
    with open(DATA_DIR / "enriched_entity2id.json") as f:
        entity2id = json.load(f)

    return graph, node_features, entity2id


# ============================
# 4. Training Loop
# ============================
def train_one_epoch(model, train_loader, node_features, graph, optimizer, criterion):
    model.train()
    total_loss = 0
    n_batches = 0

    for drugs, diseases, labels in tqdm(train_loader, desc="  Train"):
        drugs, diseases = drugs.to(device), diseases.to(device)
        labels = labels.float().to(device)

        optimizer.zero_grad()
        batch_loss = 0

        for i in range(len(drugs)):
            scores, _ = model(
                node_features, graph.edge_index.to(device),
                graph.edge_type.to(device), drugs[i],
                diseases[i].unsqueeze(0)
            )
            batch_loss += criterion(scores, labels[i].unsqueeze(0))

        batch_loss = batch_loss / len(drugs)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += batch_loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, val_loader, node_features, graph):
    model.eval()
    all_scores, all_labels = [], []

    for drugs, diseases, labels in tqdm(val_loader, desc="  Val"):
        drugs, diseases = drugs.to(device), diseases.to(device)
        for i in range(len(drugs)):
            scores, _ = model(
                node_features, graph.edge_index.to(device),
                graph.edge_type.to(device), drugs[i],
                diseases[i].unsqueeze(0)
            )
            all_scores.append(scores.cpu().numpy())
            all_labels.append(labels[i].numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    try:
        auprc = average_precision_score(all_labels, all_scores)
        auroc = roc_auc_score(all_labels, all_scores)
    except:
        auprc, auroc = 0.0, 0.0

    return auprc, auroc


# ============================
# 5. Main
# ============================
def main():
    # Load data
    graph, node_features, entity2id = load_data()

    num_nodes = graph.num_nodes
    num_relations = graph.edge_type.max().item() + 1
    print(f"  Relations: {num_relations}")

    # Datasets
    train_ds = DrugDiseaseDataset(DATA_DIR / "train_pairs.csv", entity2id)
    val_ds = DrugDiseaseDataset(DATA_DIR / "val_pairs.csv", entity2id)
    test_ds = DrugDiseaseDataset(DATA_DIR / "test_pairs.csv", entity2id)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    # Create v2 model
    print("\nCreating BioPathNet++ v2 (fixed)...")
    model = BioPathNetPlusPlusV2(
        num_nodes=num_nodes, num_relations=num_relations,
        hidden_dim=256, num_nbf_layers=3,
        num_metapaths=7, num_attention_heads=4, dropout=0.1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Load pretrained weights (partial - skip new layers)
    pretrained_path = DATA_DIR / "best_model.pt"
    if pretrained_path.exists():
        print("\n  Loading pretrained weights (partial transfer)...")
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        pretrained_dict = checkpoint.get('model_state_dict', checkpoint)

        model_dict = model.state_dict()
        # Only load weights that match in name AND shape
        matched = {k: v for k, v in pretrained_dict.items()
                   if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(matched)}/{len(model_dict)} layers from pretrained")
    else:
        print("  No pretrained weights found, training from scratch")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda')  # Mixed precision

    # Train
    print("\n" + "=" * 60)
    print("TRAINING BioPathNet++ v2")
    print("=" * 60)

    best_auprc = 0.0
    NUM_EPOCHS = 100

    for epoch in range(NUM_EPOCHS):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, node_features, graph, optimizer, criterion)
        val_auprc, val_auroc = validate(model, val_loader, node_features, graph)

        scheduler.step(val_auprc)
        elapsed = time.time() - t0

        print(f"\nEpoch {epoch}: loss={train_loss:.4f}  val_AUPRC={val_auprc:.4f}  "
              f"val_AUROC={val_auroc:.4f}  time={elapsed:.0f}s")

        # Save best
        is_best = val_auprc > best_auprc
        if is_best:
            best_auprc = val_auprc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auprc': best_auprc
            }, OUT_DIR / "best_model_v2.pt")
            print(f"  ★ New best model saved (AUPRC={best_auprc:.4f})")

        # Save every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), OUT_DIR / f"checkpoint_epoch_{epoch}.pt")

    # Final test evaluation
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    # Load best
    best_ckpt = torch.load(OUT_DIR / "best_model_v2.pt", map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'])

    test_auprc, test_auroc = validate(model, test_loader, node_features, graph)
    print(f"Test AUPRC: {test_auprc:.4f}")
    print(f"Test AUROC: {test_auroc:.4f}")

    # Save results
    results = {
        "best_val_auprc": best_auprc,
        "best_epoch": best_ckpt['epoch'],
        "test_auprc": test_auprc,
        "test_auroc": test_auroc,
        "num_epochs": NUM_EPOCHS,
        "model_params": total_params
    }
    with open(OUT_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Done! Results saved to {OUT_DIR}")
    print(f"  Download: best_model_v2.pt ({(OUT_DIR / 'best_model_v2.pt').stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
