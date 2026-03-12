"""
Optimized Training Configuration for Powerful GPU
==================================================

For V100 (16-32 GB), A100 (40-80 GB), or RTX 4090 (24 GB)

This configuration maximizes model capacity and training quality.

Author: Yatin
Date: February 10, 2026
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

from models.biopathnet import BioPathNetPlusPlus
from models.data_loader import GraphDataLoader


# ============================================================================
# OPTIMIZED CONFIGURATIONS FOR DIFFERENT GPUs
# ============================================================================

CONFIGS = {
    # RTX 3050 (4 GB) - Your current GPU
    'rtx_3050': {
        'hidden_dim': 256,
        'num_nbf_layers': 3,
        'num_metapaths': 7,
        'num_attention_heads': 4,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 16,
        'num_epochs': 50,
        'gradient_accumulation_steps': 2,  # Simulate larger batch
        'mixed_precision': False
    },
    
    # RTX 4090 (24 GB) - High-end consumer GPU
    'rtx_4090': {
        'hidden_dim': 512,          # 2x larger!
        'num_nbf_layers': 4,        # Deeper!
        'num_metapaths': 7,
        'num_attention_heads': 8,   # More attention heads
        'dropout': 0.2,             # Higher dropout for larger model
        'learning_rate': 5e-5,      # Lower LR for stability
        'weight_decay': 1e-4,
        'batch_size': 64,           # 4x larger batch!
        'num_epochs': 100,          # Train longer
        'gradient_accumulation_steps': 1,
        'mixed_precision': True     # FP16 for speed
    },
    
    # V100 (16-32 GB) - Common in academic clusters
    'v100': {
        'hidden_dim': 512,
        'num_nbf_layers': 5,        # Even deeper!
        'num_metapaths': 7,
        'num_attention_heads': 8,
        'dropout': 0.2,
        'learning_rate': 5e-5,
        'weight_decay': 1e-4,
        'batch_size': 48,
        'num_epochs': 100,
        'gradient_accumulation_steps': 1,
        'mixed_precision': True
    },
    
    # A100 (40-80 GB) - State-of-the-art research GPU
    'a100': {
        'hidden_dim': 768,          # 3x larger than baseline!
        'num_nbf_layers': 6,        # Very deep!
        'num_metapaths': 7,
        'num_attention_heads': 12,  # Many attention heads
        'dropout': 0.3,
        'learning_rate': 3e-5,
        'weight_decay': 1e-4,
        'batch_size': 128,          # Very large batch!
        'num_epochs': 150,          # Train much longer
        'gradient_accumulation_steps': 1,
        'mixed_precision': True,
        'use_gradient_checkpointing': True  # Save memory
    }
}


class OptimizedTrainer:
    """
    Enhanced trainer with advanced optimizations
    """
    
    def __init__(
        self,
        model,
        data_loader,
        config,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        self.config = config
        
        # Mixed precision training (FP16)
        self.use_amp = config.get('mixed_precision', False)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✓ Using mixed precision training (FP16)")
        
        # Optimizer with optimized settings
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999),  # Default
            eps=1e-8
        )
        
        # Advanced learning rate scheduler
        self.warmup_epochs = 5
        self.scheduler = self.get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_epochs,
            num_training_steps=config['num_epochs']
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        
        # Metrics
        self.best_val_auprc = 0.0
        self.train_losses = []
        self.val_auprcs = []
        
        # Early stopping
        self.patience = 20
        self.patience_counter = 0
    
    def get_cosine_schedule_with_warmup(self, optimizer, num_warmup_steps, num_training_steps):
        """
        Cosine learning rate schedule with warmup
        
        Better than ReduceLROnPlateau for large models
        """
        def lr_lambda(current_step):
            if current_step < num_warmup_steps:
                # Linear warmup
                return float(current_step) / float(max(1, num_warmup_steps))
            # Cosine decay
            progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    def train_epoch(self, epoch):
        """
        Optimized training epoch with gradient accumulation and mixed precision
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.data_loader.get_train_loader()
        
        self.optimizer.zero_grad()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (drugs, diseases, labels) in enumerate(pbar):
            drugs = drugs.to(self.device)
            diseases = diseases.to(self.device)
            labels = labels.float().to(self.device)
            
            batch_loss = 0.0
            
            # Mixed precision training
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                for i in range(len(drugs)):
                    drug_idx = drugs[i]
                    disease_idx = diseases[i].unsqueeze(0)
                    label = labels[i].unsqueeze(0)
                    
                    scores, _ = self.model(
                        self.data_loader.node_features,
                        self.data_loader.graph_data.edge_index,
                        self.data_loader.graph_data.edge_type,
                        drug_idx,
                        disease_idx
                    )
                    
                    loss = self.criterion(scores, label)
                    batch_loss += loss
                
                batch_loss = batch_loss / len(drugs)
                # Scale loss for gradient accumulation
                batch_loss = batch_loss / self.gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            if self.use_amp:
                self.scaler.scale(batch_loss).backward()
            else:
                batch_loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            total_loss += batch_loss.item() * self.gradient_accumulation_steps
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{batch_loss.item() * self.gradient_accumulation_steps:.4f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """Validation with mixed precision"""
        self.model.eval()
        all_scores = []
        all_labels = []
        
        val_loader = self.data_loader.get_val_loader()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for drugs, diseases, labels in pbar:
                drugs = drugs.to(self.device)
                diseases = diseases.to(self.device)
                
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    for i in range(len(drugs)):
                        drug_idx = drugs[i]
                        disease_idx = diseases[i].unsqueeze(0)
                        
                        scores, _ = self.model(
                            self.data_loader.node_features,
                            self.data_loader.graph_data.edge_index,
                            self.data_loader.graph_data.edge_type,
                            drug_idx,
                            disease_idx
                        )
                        
                        all_scores.append(scores.cpu().numpy())
                        all_labels.append(labels[i].cpu().numpy())
        
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        auprc = self.compute_auprc(all_scores, all_labels)
        return auprc
    
    def compute_auprc(self, scores, labels):
        """Compute AUPRC"""
        from sklearn.metrics import average_precision_score
        try:
            auprc = average_precision_score(labels, scores)
        except:
            auprc = 0.0
        return auprc
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auprc': self.best_val_auprc,
            'config': self.config
        }
        
        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        checkpoint_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_file)
        
        if is_best:
            best_file = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_file)
            print(f"✓ Saved best model (AUPRC: {self.best_val_auprc:.4f})")
    
    def train(self, num_epochs):
        """Complete training loop"""
        print("="*80)
        print("OPTIMIZED TRAINING - BIOPATHNET++")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
        print(f"Mixed Precision: {self.use_amp}")
        print(f"Hidden Dim: {self.config['hidden_dim']}")
        print(f"NBF Layers: {self.config['num_nbf_layers']}")
        print(f"Batch Size: {self.config['batch_size']}")
        print(f"Epochs: {num_epochs}")
        print(f"Model Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_auprc = self.validate(epoch)
            self.val_auprcs.append(val_auprc)
            
            # Step scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUPRC: {val_auprc:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Time: {epoch_time:.1f}s")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('AUPRC/val', val_auprc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Save checkpoint
            is_best = val_auprc > self.best_val_auprc
            if is_best:
                self.best_val_auprc = val_auprc
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\nEarly stopping at epoch {epoch} (patience: {self.patience})")
                break
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation AUPRC: {self.best_val_auprc:.4f}")
        
        self.writer.close()


def main():
    """
    Main training function with auto-GPU detection
    """
    
    # Detect GPU
    if not torch.cuda.is_available():
        print("⚠️  No GPU detected! Using CPU (will be very slow)")
        gpu_type = 'rtx_3050'  # Use smallest config
        device = 'cpu'
    else:
        device = 'cuda'
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        print(f"✓ GPU detected: {gpu_name}")
        print(f"  Memory: {gpu_memory:.1f} GB")
        
        # Auto-select config based on GPU
        if 'A100' in gpu_name or 'A6000' in gpu_name:
            gpu_type = 'a100'
            print("  → Using A100 config (MAXIMUM POWER!)")
        elif 'V100' in gpu_name:
            gpu_type = 'v100'
            print("  → Using V100 config (High Performance)")
        elif '4090' in gpu_name or '3090' in gpu_name:
            gpu_type = 'rtx_4090'
            print("  → Using RTX 4090 config (High Performance)")
        elif gpu_memory < 8:
            gpu_type = 'rtx_3050'
            print("  → Using RTX 3050 config (Standard)")
        else:
            gpu_type = 'rtx_4090'  # Default to high-end for unknown GPUs with enough memory
            print("  → Using RTX 4090 config (assuming high-end GPU)")
    
    # Get config
    config = CONFIGS[gpu_type]
    
    print(f"\nConfiguration: {gpu_type.upper()}")
    print(json.dumps(config, indent=2))
    
    # Save config
    config_file = Path('checkpoints/config.json')
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load data
    print("\nLoading data...")
    data_loader = GraphDataLoader(
        processed_dir='datasets/processed',
        batch_size=config['batch_size']
    )
    
    # Create model
    print("\nCreating model...")
    model = BioPathNetPlusPlus(
        num_nodes=data_loader.num_nodes,
        num_relations=data_loader.num_relations,
        hidden_dim=config['hidden_dim'],
        num_nbf_layers=config['num_nbf_layers'],
        num_metapaths=config['num_metapaths'],
        num_attention_heads=config['num_attention_heads'],
        dropout=config['dropout']
    )
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model size: {num_params:,} parameters ({num_params/1e6:.1f}M)")
    
    # Create trainer
    trainer = OptimizedTrainer(
        model=model,
        data_loader=data_loader,
        config=config,
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/training'
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()
