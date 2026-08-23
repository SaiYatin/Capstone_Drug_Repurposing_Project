import sys
import io
import json
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# Setup Windows encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader
from models.disease_mappings import load_entity_mappings
from validation.metrics import evaluate_from_csv

def main():
    print("=" * 80)
    print("BIO-PATHNET++ ABLATION STUDY")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 1. Load Data
    print("\nLoading data...")
    data_loader = GraphDataLoader(
        processed_dir=str(project_root / 'datasets' / 'processed'),
        batch_size=4
    )
    entity2id, _ = load_entity_mappings(str(project_root / 'datasets' / 'processed'))

    # 2. Load Model
    print("Loading model...")
    checkpoint_path = project_root / 'checkpoints' / 'best_model.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    model = BioPathNetPlusPlus(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        hidden_dim=config['hidden_dim'],
        num_nbf_layers=config['num_nbf_layers'],
        num_metapaths=config['num_metapaths'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_csv = project_root / 'datasets' / 'processed' / 'test_pairs.csv'

    ablations = [
        "Full Model",
        "No Metapaths",
        "No BRG",
        "No Fusion",
        "Single NBF Layer",
        "No BioBERT Features"
    ]

    results = {}
    baseline_auprc = None
    baseline_auroc = None

    for ablation in ablations:
        print(f"\n{'-' * 40}")
        print(f"Running Ablation: {ablation}")
        print(f"{'-' * 40}")

        handles = []
        cleanup_fn = lambda: None

        if ablation == "Full Model":
            pass
        elif ablation == "No Metapaths":
            def metapath_hook(module, input, output):
                return (torch.zeros_like(output[0]), output[1])
            h = model.metapath_attention.register_forward_hook(metapath_hook)
            handles.append(h)
        elif ablation == "No BRG":
            # Documenting: BRG context is already a zero tensor placeholder in forward pass when brg_paths is None
            pass
        elif ablation == "No Fusion":
            def fusion_hook(module, input, output):
                # Return the original drug_repr (first hidden_dim elements)
                return input[0][:model.hidden_dim]
            h = model.fusion.register_forward_hook(fusion_hook)
            handles.append(h)
        elif ablation == "Single NBF Layer":
            original_num_layers = model.nbfnet.num_layers
            model.nbfnet.num_layers = 1
            def cleanup_nbf():
                model.nbfnet.num_layers = original_num_layers
            cleanup_fn = cleanup_nbf
        elif ablation == "No BioBERT Features":
            original_features = data_loader.node_features.clone()
            data_loader.node_features = torch.randn_like(data_loader.node_features)
            def cleanup_features():
                data_loader.node_features = original_features
            cleanup_fn = cleanup_features

        # Evaluate
        overall_metrics, _ = evaluate_from_csv(
            model, data_loader, test_csv, entity2id, device, split_name='test'
        )
        
        auprc = overall_metrics['auprc']
        auroc = overall_metrics['auroc']

        # Cleanup
        for h in handles:
            h.remove()
        cleanup_fn()

        if ablation == "Full Model":
            baseline_auprc = auprc
            baseline_auroc = auroc
            delta_auprc = 0.0
            delta_auroc = 0.0
        else:
            delta_auprc = auprc - baseline_auprc
            delta_auroc = auroc - baseline_auroc

        print(f"Result for {ablation}:")
        print(f"  AUPRC: {auprc:.4f} (Delta: {delta_auprc:+.4f})")
        print(f"  AUROC: {auroc:.4f} (Delta: {delta_auroc:+.4f})")

        results[ablation] = {
            "AUPRC": auprc,
            "AUROC": auroc,
            "Delta_AUPRC": delta_auprc,
            "Delta_AUROC": delta_auroc
        }

    # Save results to JSON
    outputs_dir = project_root / 'outputs' / 'validation_v2'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    results_path = outputs_dir / 'ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved numerical results to {results_path}")

    # Generate Bar Chart
    vis_dir = project_root / 'outputs' / 'visualizations'
    vis_dir.mkdir(parents=True, exist_ok=True)
    chart_path = vis_dir / 'ablation_chart.png'

    labels = list(results.keys())
    auprc_vals = [results[l]["AUPRC"] for l in labels]
    auroc_vals = [results[l]["AUROC"] for l in labels]

    x = range(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar([p - width/2 for p in x], auprc_vals, width, label='AUPRC')
    rects2 = ax.bar([p + width/2 for p in x], auroc_vals, width, label='AUROC')

    ax.set_ylabel('Scores')
    ax.set_title('BioPathNet++ Ablation Study Results')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()

    ax.set_ylim([0.0, 1.0])

    fig.tight_layout()
    plt.savefig(chart_path)
    print(f"Saved visualization to {chart_path}")
    print("\nAblation study complete.")

if __name__ == '__main__':
    main()
