"""
Evaluate Trained BioPathNet++ Model
====================================

Tests the best_model.pt checkpoint to verify performance.

Author: Yatin
Date: March 17, 2026
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
from tqdm import tqdm

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader


def evaluate_model(checkpoint_path, data_loader, split='val'):
    """
    Evaluate model on validation or test set
    
    Args:
        checkpoint_path: Path to checkpoint file
        data_loader: GraphDataLoader instance
        split: 'val' or 'test'
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Get config
    config = checkpoint['config']
    print(f"\nModel Config:")
    print(f"  Hidden Dim: {config['hidden_dim']}")
    print(f"  NBF Layers: {config['num_nbf_layers']}")
    print(f"  Epochs Trained: {checkpoint['epoch'] + 1}")
    print(f"  Best Val AUPRC: {checkpoint['best_val_auprc']:.4f}")
    
    # Create model
    model = BioPathNetPlusPlus(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        hidden_dim=config['hidden_dim'],
        num_nbf_layers=config['num_nbf_layers'],
        num_metapaths=config['num_metapaths'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout']
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Model loaded successfully!")
    print(f"✓ Evaluating on {split} set...")
    
    # Get data loader
    if split == 'val':
        loader = data_loader.get_val_loader()
    elif split == 'test':
        loader = data_loader.get_test_loader()
    else:
        raise ValueError(f"Unknown split: {split}")
    
    # Evaluate
    all_scores = []
    all_labels = []
    all_drugs = []
    all_diseases = []
    
    with torch.no_grad():
        for drugs, diseases, labels in tqdm(loader, desc=f"Evaluating {split}"):
            drugs = drugs.to(device)
            diseases = diseases.to(device)
            
            for i in range(len(drugs)):
                drug_idx = drugs[i]
                disease_idx = diseases[i].unsqueeze(0)
                
                # Forward pass
                scores, _ = model(
                    data_loader.node_features,
                    data_loader.graph_data.edge_index,
                    data_loader.graph_data.edge_type,
                    drug_idx,
                    disease_idx
                )
                
                # Store results
                all_scores.append(scores.cpu().item())
                all_labels.append(labels[i].cpu().item())
                all_drugs.append(drug_idx.cpu().item())
                all_diseases.append(disease_idx.cpu().item())
    
    # Convert to numpy
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    print(f"\n{'='*80}")
    print(f"EVALUATION RESULTS - {split.upper()} SET")
    print(f"{'='*80}")
    print(f"Total Samples: {len(all_labels)}")
    print(f"Positive Samples: {all_labels.sum()}")
    print(f"Negative Samples: {len(all_labels) - all_labels.sum()}")
    print(f"\nScore Statistics:")
    print(f"  Min:  {all_scores.min():.4f}")
    print(f"  Max:  {all_scores.max():.4f}")
    print(f"  Mean: {all_scores.mean():.4f}")
    print(f"  Std:  {all_scores.std():.4f}")
    
    # Check if all samples are positive (data leakage!)
    if all_labels.sum() == len(all_labels):
        print(f"\n⚠️  WARNING: ALL SAMPLES ARE POSITIVE!")
        print(f"    This indicates a problem with the dataset!")
        print(f"    AUPRC = 1.0 is meaningless in this case.")
        return
    
    # Compute AUPRC
    try:
        auprc = average_precision_score(all_labels, all_scores)
        print(f"\n✓ AUPRC: {auprc:.4f}")
    except Exception as e:
        print(f"\n✗ Could not compute AUPRC: {e}")
        auprc = None
    
    # Compute AUROC
    try:
        auroc = roc_auc_score(all_labels, all_scores)
        print(f"✓ AUROC: {auroc:.4f}")
    except Exception as e:
        print(f"✗ Could not compute AUROC: {e}")
        auroc = None
    
    # Compute precision at different thresholds
    if auprc is not None:
        precision, recall, thresholds = precision_recall_curve(all_labels, all_scores)
        
        print(f"\nPrecision at different recalls:")
        for r in [0.1, 0.3, 0.5, 0.7, 0.9]:
            idx = np.argmin(np.abs(recall - r))
            print(f"  Recall={r:.1f}: Precision={precision[idx]:.4f}")
    
    # Show top predictions
    print(f"\nTop 10 Predictions:")
    top_indices = np.argsort(all_scores)[::-1][:10]
    for rank, idx in enumerate(top_indices, 1):
        print(f"  {rank}. Score={all_scores[idx]:.4f}, Label={int(all_labels[idx])}, "
              f"Drug={all_drugs[idx]}, Disease={all_diseases[idx]}")
    
    print(f"\n{'='*80}")
    
    return {
        'auprc': auprc,
        'auroc': auroc,
        'scores': all_scores,
        'labels': all_labels
    }


def main():
    """Main evaluation function"""
    
    print("="*80)
    print("BIOPATHNET++ MODEL EVALUATION")
    print("="*80)
    
    # Load data
    print("\nLoading data...")
    data_loader = GraphDataLoader(
        processed_dir='datasets/processed',
        batch_size=4
    )
    
    # Evaluate best model on validation set
    results_val = evaluate_model(
        checkpoint_path='checkpoints/best_model.pt',
        data_loader=data_loader,
        split='val'
    )
    
    # Evaluate on test set
    print("\n" + "="*80)
    input("\nPress Enter to evaluate on TEST set...")
    
    results_test = evaluate_model(
        checkpoint_path='checkpoints/best_model.pt',
        data_loader=data_loader,
        split='test'
    )
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    if results_val and results_test:
        print(f"Validation AUPRC: {results_val.get('auprc', 'N/A'):.4f}")
        print(f"Test AUPRC:       {results_test.get('auprc', 'N/A'):.4f}")
        
        if results_val.get('auprc') and results_test.get('auprc'):
            gap = results_val['auprc'] - results_test['auprc']
            print(f"Generalization Gap: {gap:.4f}")
            
            if gap > 0.1:
                print(f"\n⚠️  Large gap suggests overfitting!")
            else:
                print(f"\n✓ Good generalization!")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
