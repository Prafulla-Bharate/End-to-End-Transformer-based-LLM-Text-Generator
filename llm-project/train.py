import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import time
from tqdm import tqdm

# Import our custom modules
from config import config
from model import GPTModel
from dataset import create_data_loader
from tokenizer import SimpleTokenizer

tokenizer = SimpleTokenizer()

class Trainer:

    def __init__(self):
        
        self.device = torch.device(config.device)
        print(f"Using device: {self.device}")

        #Initializing Model
        self.model = GPTModel(vocab_size=tokenizer.get_vocab_size())
        self.model = self.model.to(self.device)

        #count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total pameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")

        # Initialize optimizer (AdamW is best for transformers)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,  # L2 regularization
            betas=(0.9, 0.95)   # Momentum parameters optimized for transformers
        )

         # Learning rate scheduler (cosine annealing)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.1  # Minimum learning rate
        )

        # Create data loader
        self.data_loader = create_data_loader(config.data_path)
        print(f"Training batches per epoch: {len(self.data_loader)}")

        # Training tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.training_losses = []

        # Create directory for saved models
        os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)


    def train_epoch(self):
        """
        Train the model for one epoch.
        
        Returns:
            float: Average loss for the epoch
        """
        self.model.train()  # Set model to training mode
        epoch_losses = []
        
        # Progress bar for this epoch
        pbar = tqdm(self.data_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = batch.to(self.device)  # Shape: [batch_size, seq_len]
            
            # Forward pass
            logits, loss = self.model(batch, targets=batch)
            
            # Backward pass
            self.optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            
            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Track loss
            epoch_losses.append(loss.item())
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Generate sample text periodically
            if batch_idx % 50 == 0 and batch_idx > 0:
                self.generate_sample()
        
        # Calculate average loss for epoch
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        self.training_losses.append(avg_loss)
        
        return avg_loss


    def generate_sample(self):
        """Generate sample text to monitor training progress"""
        self.model.eval()  # Set to evaluation mode
        
        # Sample prompt
        prompt = "The future of artificial intelligence"
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids]).to(self.device)
        
        # Generate text
        with torch.no_grad():
            generated = self.model.generate(
                input_tensor,
                max_length=50,
                temperature=0.8,
                top_k=50
            )
        
        # Decode generated text
        generated_text = tokenizer.decode(generated[0].cpu().tolist())
        print(f"\nSample generation: {generated_text}\n")
        
        self.model.train()  # Back to training mode
    

    def save_checkpoint(self, is_best=False):
        """
        Save model checkpoint.
        
        Args:
            is_best (bool): Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': self.training_losses[-1] if self.training_losses else float('inf'),
            'global_step': self.global_step
        }
        
        # Save regular checkpoint
        checkpoint_path = config.model_save_path.replace('.pt', f'_epoch_{self.current_epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, config.model_save_path)
            print(f"Saved new best model with loss: {self.training_losses[-1]:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model checkpoint to resume training.
        
        Args:
            checkpoint_path (str): Path to checkpoint file
        """
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            # Load model state
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Load training state
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            
            print(f"Resumed from epoch {self.current_epoch}")
        else:
            print("No checkpoint found, starting fresh training")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        print(f"Training for {config.num_epochs} epochs")
        print(f"Batch size: {config.batch_size}")
        print(f"Learning rate: {config.learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, config.num_epochs):
            self.current_epoch = epoch
            
            # Train one epoch
            avg_loss = self.train_epoch()
            
            # Update learning rate
            self.scheduler.step()
            
            # Check if this is the best model
            is_best = avg_loss < self.best_loss
            if is_best:
                self.best_loss = avg_loss
            
            # Save checkpoint
            self.save_checkpoint(is_best=is_best)
            
            # Print epoch summary
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch + 1}/{config.num_epochs} completed")
            print(f"Average loss: {avg_loss:.4f}")
            print(f"Best loss: {self.best_loss:.4f}")
            print(f"Time elapsed: {elapsed_time:.2f}s")
            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print("-" * 50)
        
        print("Training completed!")
        print(f"Total training time: {time.time() - start_time:.2f}s")
        print(f"Best loss achieved: {self.best_loss:.4f}")
        
        # Final sample generation
        print("\nFinal sample generation:")
        self.generate_sample()

def main():
    """Main function to start training"""
    # Create trainer and start training
    trainer = Trainer()
    
    # Optionally load existing checkpoint
    # trainer.load_checkpoint(config.model_save_path)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()