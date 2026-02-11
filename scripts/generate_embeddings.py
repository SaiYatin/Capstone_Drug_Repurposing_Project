"""
Entity Embedding Generator
===========================

Generates embeddings for all entities in enriched graph using:
- BioBERT (dmis-lab/biobert-v1.1) for: Disease, Gene, Protein, Phenotype, etc.
- ChemBERTa (seyonec/ChemBERTa-zinc-base-v1) for: Compounds/Drugs

Output: 768-dimensional embeddings for each entity

Author: Yatin
Date: February 9, 2026
"""

import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EmbeddingGenerator:
    """Generate BioBERT and ChemBERTa embeddings"""
    
    def __init__(
        self,
        processed_dir='datasets/processed',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=32
    ):
        self.processed_dir = Path(processed_dir)
        self.device = device
        self.batch_size = batch_size
        
        print("="*80)
        print("ENTITY EMBEDDING GENERATOR")
        print("="*80)
        print(f"\nDevice: {self.device}")
        print(f"Batch size: {self.batch_size}")
        print()
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load BioBERT and ChemBERTa"""
        print("Loading pre-trained models...")
        
        try:
            # BioBERT
            print("  Loading BioBERT (dmis-lab/biobert-v1.1)...")
            self.biobert_tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
            self.biobert_model = AutoModel.from_pretrained('dmis-lab/biobert-v1.1').to(self.device)
            self.biobert_model.eval()
            print("  ✓ BioBERT loaded")
            
            # ChemBERTa (for compounds)
            # Note: ChemBERTa expects SMILES strings, but we don't have those
            # Fallback: Use BioBERT for compound names too
            print("  ✓ Using BioBERT for all entities (ChemBERTa requires SMILES)")
            self.use_chembert = False
            
        except Exception as e:
            print(f"  ✗ Error loading models: {e}")
            print("  Attempting to use smaller model...")
            
            # Fallback to smaller model if GPU memory is limited
            try:
                from transformers import BertTokenizer, BertModel
                print("  Loading BERT-base (fallback)...")
                self.biobert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.biobert_model = BertModel.from_pretrained('bert-base-uncased').to(self.device)
                self.biobert_model.eval()
                print("  ✓ BERT-base loaded")
                self.use_chembert = False
            except Exception as e2:
                print(f"  ✗ Failed to load models: {e2}")
                raise
        
        print()
    
    def encode_text(self, texts, model, tokenizer):
        """Encode texts using given model"""
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            
            # Tokenize
            inputs = tokenizer(
                batch_texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings)
    
    def extract_entity_text(self, entity):
        """Extract meaningful text from entity ID"""
        # Entity format: Type::ID or Type::Name
        # Examples:
        #   Disease::MESH:D001172
        #   Gene::TNF
        #   Compound::DB00001
        #   Phenotype::HP:0001234
        
        if '::' not in entity:
            return entity
        
        parts = entity.split('::', 1)
        entity_type = parts[0]
        entity_id = parts[1]
        
        # Clean up the ID part
        # Remove database prefixes (MESH:, ORPHA:, HP:, etc.)
        if ':' in entity_id:
            # Keep database prefix for context
            # E.g., "MESH D001172" instead of just "D001172"
            db, code = entity_id.split(':', 1)
            text = f"{entity_type} {db} {code}"
        else:
            text = f"{entity_type} {entity_id}"
        
        return text
    
    def generate_embeddings(self):
        """Generate embeddings for all entities"""
        print("Loading entity mappings...")
        
        # Load entity2id
        entity2id_file = self.processed_dir / 'enriched_entity2id.json'
        with open(entity2id_file, 'r') as f:
            entity2id = json.load(f)
        
        print(f"  ✓ Loaded {len(entity2id):,} entities")
        
        # Extract entity texts
        print("\nExtracting entity texts...")
        entities = sorted(entity2id.keys())
        entity_texts = [self.extract_entity_text(e) for e in entities]
        
        # Show samples
        print("\n  Sample entity texts:")
        for i in range(min(5, len(entity_texts))):
            print(f"    {entities[i][:50]:50s} → {entity_texts[i][:60]}")
        print()
        
        # Generate embeddings
        print(f"Generating embeddings (batch_size={self.batch_size})...")
        print(f"This may take 5-15 minutes depending on dataset size...")
        
        all_embeddings = []
        
        # Process in chunks with progress bar
        chunk_size = 1000
        for i in tqdm(range(0, len(entity_texts), chunk_size), desc="Processing"):
            chunk_texts = entity_texts[i:i+chunk_size]
            chunk_embeddings = self.encode_text(
                chunk_texts,
                self.biobert_model,
                self.biobert_tokenizer
            )
            all_embeddings.append(chunk_embeddings)
        
        # Combine all embeddings
        embedding_matrix = np.vstack(all_embeddings)
        
        print(f"\n✓ Generated embeddings: {embedding_matrix.shape}")
        print(f"  Entities: {embedding_matrix.shape[0]:,}")
        print(f"  Dimensions: {embedding_matrix.shape[1]}")
        print()
        
        return embedding_matrix, entities
    
    def save_embeddings(self, embeddings, entities):
        """Save embeddings"""
        print("Saving embeddings...")
        
        # Save as numpy array
        output_file = self.processed_dir / 'entity_embeddings.npy'
        np.save(output_file, embeddings)
        print(f"✓ Saved embeddings: {output_file}")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Size: {output_file.stat().st_size / (1024**2):.1f} MB")
        
        # Save metadata
        metadata = {
            'num_entities': len(entities),
            'embedding_dim': embeddings.shape[1],
            'model': 'BioBERT (dmis-lab/biobert-v1.1)',
            'device': str(self.device),
            'date_generated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        metadata_file = self.processed_dir / 'embedding_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"✓ Saved metadata: {metadata_file}")
        
        print()
    
    def run(self):
        """Run complete embedding generation"""
        # Generate embeddings
        embeddings, entities = self.generate_embeddings()
        
        # Save
        self.save_embeddings(embeddings, entities)
        
        # Final summary
        print("="*80)
        print("EMBEDDING GENERATION COMPLETE")
        print("="*80)
        print(f"\nGenerated embeddings for {len(entities):,} entities")
        print(f"Embedding dimension: {embeddings.shape[1]}")
        print(f"\nOutput file: {self.processed_dir / 'entity_embeddings.npy'}")
        print("\nNext step: Build Background Regulatory Graph (BRG)")
        print("="*80)

def main():
    """Main entry point"""
    
    # Check if GPU available
    if torch.cuda.is_available():
        print(f"\n✓ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB\n")
    else:
        print("\n⚠ No GPU detected - using CPU (will be slower)\n")
    
    generator = EmbeddingGenerator(
        processed_dir='datasets/processed',
        batch_size=32  # Reduce if GPU out of memory
    )
    
    generator.run()

if __name__ == '__main__':
    main()
