"""
H100 GPU-Optimized Training Script for AgentB

Features:
- Mixed precision (FP16) for 2x speed on H100
- Automatic checkpointing
- TensorBoard logging
- Time-limited training (stops before SLURM kills it)
- Resume from checkpoint
"""

import sys
import os
import time
import yaml
import json
import argparse
from pathlib import Path
from datetime import datetime, timedelta

# Add paths to import agent modules
# From training/ we need to go up to AgentPro/ to import evaluator
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)  # AgentPro/

sys.path.insert(0, agentpro_dir)  # For 'from evaluator import PositionEvaluator'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler

import numpy as np

from evaluator import PositionEvaluator


class ChickenDataset(Dataset):
    """Dataset for chicken game positions"""

    def __init__(self, data_file, augment=False, noise_level=0.02):
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.augment = augment
        self.noise_level = noise_level
        print(f"Loaded {len(self.data)} positions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        score = torch.FloatTensor([item['score']])

        # Data augmentation
        if self.augment and np.random.random() < 0.3:
            noise = torch.randn_like(features) * self.noise_level
            features = features + noise

        return features, score


class ImprovedPositionEvaluator(nn.Module):
    """
    Enhanced evaluator with residual connections
    """

    def __init__(self, input_size=128, hidden_size=256, num_blocks=2, dropout=0.2):
        super().__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_size) for _ in range(num_blocks)
        ])

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.input_layer(x)
        for block in self.residual_blocks:
            x = block(x)
        x = self.output_layers(x)
        return x


class ResidualBlock(nn.Module):
    """Residual block with skip connection"""

    def __init__(self, size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(size, size),
            nn.BatchNorm1d(size),
            nn.ReLU(),
            nn.Linear(size, size),
            nn.BatchNorm1d(size)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.layers(x) + x)


class Trainer:
    """
    H100-optimized trainer with mixed precision
    """

    def __init__(self, config, resume_from=None):
        self.config = config
        self.device = torch.device(config['hardware']['device'])

        print(f"Using device: {self.device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Load data
        self.train_loader, self.val_loader, self.test_loader = self._load_data()

        # Create model
        self.model = ImprovedPositionEvaluator(
            input_size=config['model']['input_size'],
            hidden_size=config['model']['hidden_size'],
            num_blocks=config['model']['num_residual_blocks'],
            dropout=config['model']['dropout']
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=config['training']['betas']
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=config['training']['scheduler']['factor'],
            patience=config['training']['scheduler']['patience'],
            min_lr=config['training']['scheduler']['min_lr']
        )

        # Loss function
        self.criterion = nn.SmoothL1Loss()

        # Mixed precision scaler for H100
        self.use_amp = config['training']['use_amp']
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("Using Automatic Mixed Precision (FP16)")

        # TensorBoard
        if config['logging']['use_tensorboard']:
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(config['logging']['tensorboard_dir']) / datetime.now().strftime('%Y%m%d_%H%M%S')
                self.writer = SummaryWriter(log_dir)
                print(f"TensorBoard logging to: {log_dir}")
            except ImportError:
                print("Warning: tensorboard not installed. Install with: pip install tensorboard")
                print("Disabling TensorBoard logging...")
                self.writer = None
        else:
            self.writer = None

        # Training state
        self.start_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.global_step = 0
        self.start_time = time.time()
        self.last_checkpoint_time = time.time()

        # Resume from checkpoint
        if resume_from:
            self._load_checkpoint(resume_from)

    def _load_data(self):
        """Load and split dataset"""
        config = self.config

        dataset = ChickenDataset(
            config['data']['train_file'],
            augment=config['data']['augment'],
            noise_level=config['data']['noise_level']
        )

        # Split
        total_size = len(dataset)
        test_size = int(total_size * config['data']['test_split'])
        val_size = int(total_size * config['data']['val_split'])
        train_size = total_size - val_size - test_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"\nDataset splits:")
        print(f"  Train: {len(train_dataset)}")
        print(f"  Val:   {len(val_dataset)}")
        print(f"  Test:  {len(test_dataset)}")

        # Data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['hardware']['num_workers'],
            pin_memory=config['hardware']['pin_memory']
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (features, scores) in enumerate(self.train_loader):
            features = features.to(self.device)
            scores = scores.to(self.device)

            self.optimizer.zero_grad()

            # Mixed precision forward pass
            if self.use_amp:
                with autocast():
                    predictions = self.model(features)
                    loss = self.criterion(predictions, scores)

                self.scaler.scale(loss).backward()

                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features)
                loss = self.criterion(predictions, scores)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

                self.optimizer.step()

            total_loss += loss.item()
            self.global_step += 1

            # Logging
            if batch_idx % self.config['logging']['log_every_n_batches'] == 0:
                if self.writer:
                    self.writer.add_scalar('train/batch_loss', loss.item(), self.global_step)

        return total_loss / num_batches

    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, scores in self.val_loader:
                features = features.to(self.device)
                scores = scores.to(self.device)

                if self.use_amp:
                    with autocast():
                        predictions = self.model(features)
                        loss = self.criterion(predictions, scores)
                else:
                    predictions = self.model(features)
                    loss = self.criterion(predictions, scores)

                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch, val_loss, filename='checkpoint.pth'):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'global_step': self.global_step,
            'val_loss': val_loss,
            'config': self.config
        }

        if self.use_amp:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()

        torch.save(checkpoint, filename)
        print(f"  Saved checkpoint: {filename}")

    def _load_checkpoint(self, filename):
        """Load training checkpoint"""
        print(f"Resuming from checkpoint: {filename}")
        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        self.patience_counter = checkpoint['patience_counter']
        self.global_step = checkpoint['global_step']

        if self.use_amp and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

        print(f"  Resumed from epoch {self.start_epoch}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")

    def train(self, max_hours=None):
        """
        Full training loop

        Args:
            max_hours: Maximum training time in hours (for SLURM time limits)
        """
        config = self.config
        epochs = config['training']['epochs']

        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        if max_hours:
            print(f"Time limit: {max_hours} hours")
        print(f"{'='*60}\n")

        for epoch in range(self.start_epoch, epochs):
            epoch_start = time.time()

            # Check time limit
            if max_hours:
                elapsed_hours = (time.time() - self.start_time) / 3600
                if elapsed_hours > max_hours - 0.5:  # Stop 30min before limit
                    print(f"\nApproaching time limit ({elapsed_hours:.1f}/{max_hours}h)")
                    print("Saving final checkpoint...")
                    self.save_checkpoint(epoch, self.best_val_loss, 'final_checkpoint.pth')
                    break

            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss = self.validate()

            # Update scheduler
            self.scheduler.step(val_loss)

            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']

            # Print progress
            elapsed = time.time() - epoch_start
            print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar('train/epoch_loss', train_loss, epoch)
                self.writer.add_scalar('val/loss', val_loss, epoch)
                self.writer.add_scalar('train/learning_rate', current_lr, epoch)

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, val_loss, 'best_evaluator.pth')
                print(f"  ✓ New best model! (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Periodic checkpoint (every N epochs or N minutes)
            if (epoch + 1) % config['checkpoint']['save_every_n_epochs'] == 0:
                self.save_checkpoint(epoch, val_loss, f'checkpoint_epoch_{epoch+1}.pth')

            time_since_checkpoint = time.time() - self.last_checkpoint_time
            if time_since_checkpoint > config['checkpoint']['save_every_n_minutes'] * 60:
                self.save_checkpoint(epoch, val_loss, 'latest_checkpoint.pth')
                self.last_checkpoint_time = time.time()

            # Early stopping
            if self.patience_counter >= config['training']['early_stopping']['patience']:
                print(f"\nEarly stopping after {epoch+1} epochs")
                print(f"No improvement for {self.patience_counter} epochs")
                break

        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print(f"Total time: {(time.time() - self.start_time) / 3600:.2f} hours")
        print(f"{'='*60}\n")

        # Test on held-out set
        print("Evaluating on test set...")
        test_loss = self._test()
        print(f"Test loss: {test_loss:.4f}")

        if self.writer:
            self.writer.close()

        return self.best_val_loss

    def _test(self):
        """Test on held-out set"""
        self.model.eval()
        total_loss = 0.0

        # Load best model
        checkpoint = torch.load('best_evaluator.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        with torch.no_grad():
            for features, scores in self.test_loader:
                features = features.to(self.device)
                scores = scores.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, scores)

                total_loss += loss.item()

        return total_loss / len(self.test_loader)


def main():
    parser = argparse.ArgumentParser(description='Train AgentB on H100 GPU')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--max-hours', type=float, default=None,
                        help='Maximum training time in hours')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    print(f"\n{'='*60}")
    print("AgentB Training on H100 GPU")
    print(f"{'='*60}\n")

    # Train
    trainer = Trainer(config, resume_from=args.resume)
    trainer.train(max_hours=args.max_hours)


if __name__ == '__main__':
    main()
