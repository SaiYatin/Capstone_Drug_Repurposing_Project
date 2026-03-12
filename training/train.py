"""
Training Script for BioPathNet++
=================================

Trains the model on data-rich diseases (RA, SLE, MS) for transfer learning.

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


class Trainer:
    """
    Trainer for BioPathNet++
    """
    
    def __init__(
        self,
        model,
        data_loader,
        learning_rate=1e-4,
        weight_decay=1e-5,
        device='cuda',
        checkpoint_dir='checkpoints',
        log_dir='logs'
    ):
        self.model = model.to(device)
        self.data_loader = data_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function (Binary Cross Entropy)
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir)
        
        # Metrics tracking
        self.best_val_auprc = 0.0
        self.train_losses = []
        self.val_auprcs = []
    
    def train_epoch(self, epoch):
        """
        Train for one epoch
        
        Returns:
            Average loss
        """
        
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        train_loader = self.data_loader.get_train_loader()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (drugs, diseases, labels) in enumerate(pbar):
            # Move to device
            drugs = drugs.to(self.device)
            diseases = diseases.to(self.device)
            labels = labels.float().to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass (one drug at a time for simplicity)
            batch_loss = 0.0
            for i in range(len(drugs)):
                drug_idx = drugs[i]
                disease_idx = diseases[i].unsqueeze(0)  # Make it a single-element tensor
                label = labels[i].unsqueeze(0)
                
                # Get scores
                scores, _ = self.model(
                    self.data_loader.node_features,
                    self.data_loader.graph_data.edge_index,
                    self.data_loader.graph_data.edge_type,
                    drug_idx,
                    disease_idx
                )
                
                # Compute loss
                loss = self.criterion(scores, label)
                batch_loss += loss
            
            # Average loss over batch
            batch_loss = batch_loss / len(drugs)
            
            # Backward pass
            batch_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            
            # Track loss
            total_loss += batch_loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{batch_loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, epoch):
        """
        Validate model
        
        Returns:
            AUPRC score
        """
        
        self.model.eval()
        all_scores = []
        all_labels = []
        
        val_loader = self.data_loader.get_val_loader()
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
            for drugs, diseases, labels in pbar:
                drugs = drugs.to(self.device)
                diseases = diseases.to(self.device)
                
                # Forward pass
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
        
        # Compute AUPRC
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)
        
        auprc = self.compute_auprc(all_scores, all_labels)
        
        return auprc
    
    def compute_auprc(self, scores, labels):
        """
        Compute Area Under Precision-Recall Curve
        """
        from sklearn.metrics import average_precision_score
        
        try:
            auprc = average_precision_score(labels, scores)
        except:
            auprc = 0.0
        
        return auprc
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Save model checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_auprc': self.best_val_auprc,
            'train_losses': self.train_losses,
            'val_auprcs': self.val_auprcs
        }
        
        # Save regular checkpoint
        checkpoint_file = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_file)
        
        # Save best model
        if is_best:
            best_file = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_file)
            print(f"✓ Saved best model (AUPRC: {self.best_val_auprc:.4f})")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_auprc = checkpoint['best_val_auprc']
        self.train_losses = checkpoint['train_losses']
        self.val_auprcs = checkpoint['val_auprcs']
        
        print(f"✓ Loaded checkpoint from {checkpoint_path}")
        return checkpoint['epoch']
    
    def train(self, num_epochs, start_epoch=0):
        """
        Complete training loop
        """
        print("="*80)
        print("TRAINING BIOPATHNET++")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print()
        
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            self.train_losses.append(train_loss)
            
            # Validate
            val_auprc = self.validate(epoch)
            self.val_auprcs.append(val_auprc)
            
            epoch_time = time.time() - start_time
            
            # Log
            print(f"\nEpoch {epoch}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val AUPRC: {val_auprc:.4f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('AUPRC/val', val_auprc, epoch)
            self.writer.add_scalar('LR', self.optimizer.param_groups[0]['lr'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_auprc)
            
            # Save checkpoint
            is_best = val_auprc > self.best_val_auprc
            if is_best:
                self.best_val_auprc = val_auprc
            
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
        
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation AUPRC: {self.best_val_auprc:.4f}")
        print(f"Model saved to: {self.checkpoint_dir / 'best_model.pt'}")
        
        self.writer.close()


def main():
    """
    Main training function
    """
    
    # Configuration
    config = {
        'hidden_dim': 256,
        'num_nbf_layers': 3,
        'num_metapaths': 7,
        'num_attention_heads': 4,
        'dropout': 0.1,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'batch_size': 32,
        'num_epochs': 50
    }
    
    # Save config
    config_file = Path('checkpoints/config.json')
    config_file.parent.mkdir(exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
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
    
    # Create trainer
    trainer = Trainer(
        model=model,
        data_loader=data_loader,
        learning_rate=config['learning_rate'],
        weight_decay=config['weight_decay'],
        device=device,
        checkpoint_dir='checkpoints',
        log_dir='logs/training'
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])


if __name__ == '__main__':
    main()
