"""
Neural Network Training for AgentB

Trains the position evaluator to predict deep search results.
Incorporates techniques from Stockfish/AlphaZero for robustness.
"""

import sys
import os
import json
import numpy as np
from typing import List, Tuple

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, random_split
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch not available! Install with: pip install torch")
    sys.exit(1)

from evaluator import PositionEvaluator


class ChickenPositionDataset(Dataset):
    """
    Dataset of chicken game positions with evaluations.
    """

    def __init__(self, data_file: str, augment: bool = True):
        with open(data_file, 'r') as f:
            self.data = json.load(f)

        self.augment = augment
        print(f"Loaded {len(self.data)} positions from {data_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        score = torch.FloatTensor([item['score']])

        # Data augmentation: add small noise to make model robust
        if self.augment and np.random.random() < 0.3:
            # Add small Gaussian noise to features
            noise = torch.randn_like(features) * 0.02
            features = features + noise

        return features, score


class ImprovedPositionEvaluator(nn.Module):
    """
    Improved neural network with:
    - Deeper architecture
    - Residual connections (like AlphaZero)
    - Batch normalization
    - Dropout for robustness
    """

    def __init__(self, input_size=128, hidden_size=256):
        super().__init__()

        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Residual blocks (like AlphaZero)
        self.residual_block1 = self._make_residual_block(hidden_size)
        self.residual_block2 = self._make_residual_block(hidden_size)

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def _make_residual_block(self, size):
        """Create a residual block (skip connection)"""
        return ResidualBlock(size)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_block1(x)
        x = self.residual_block2(x)
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
        residual = x
        out = self.layers(x)
        out = out + residual  # Skip connection
        out = self.relu(out)
        return out


class Trainer:
    """
    Trainer class with Stockfish-inspired techniques:
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Model checkpointing
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = 0.001,
        device: str = 'cpu'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimizer with weight decay (L2 regularization)
        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

        # Learning rate scheduler (reduce on plateau)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )

        # Loss function: MSE + L1 (Huber-like)
        self.criterion = nn.SmoothL1Loss()  # Robust to outliers

        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = 15

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for features, scores in self.train_loader:
            features = features.to(self.device)
            scores = scores.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(features)
            loss = self.criterion(predictions, scores)

            # Backward pass
            loss.backward()

            # Gradient clipping (prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches

    def validate(self) -> float:
        """Validate model"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for features, scores in self.val_loader:
                features = features.to(self.device)
                scores = scores.to(self.device)

                predictions = self.model(features)
                loss = self.criterion(predictions, scores)

                total_loss += loss.item()
                num_batches += 1

        return total_loss / num_batches

    def train(self, epochs: int, save_path: str = 'best_evaluator.pth'):
        """
        Full training loop with early stopping.
        """
        print(f"\nTraining for {epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print progress
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"LR: {current_lr:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ Saved new best model (val_loss={val_loss:.4f})")
            else:
                self.patience_counter += 1

            # Early stopping
            if self.patience_counter >= self.max_patience:
                print(f"\nEarly stopping after {epoch+1} epochs (no improvement for {self.max_patience} epochs)")
                break

        print(f"\n✓ Training complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        return self.model


def train_robust_model(
    data_file: str = 'training_data.json',
    epochs: int = 200,
    batch_size: int = 128,
    lr: float = 0.001,
    val_split: float = 0.15,
    test_split: float = 0.10
):
    """
    Train robust position evaluator using Stockfish-inspired techniques.

    Args:
        data_file: Path to training data JSON
        epochs: Maximum training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        val_split: Fraction of data for validation
        test_split: Fraction of data for testing
    """
    # Load dataset
    dataset = ChickenPositionDataset(data_file, augment=True)

    # Split into train/val/test
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} ({100*train_size/total_size:.1f}%)")
    print(f"  Val: {len(val_dataset)} ({100*val_size/total_size:.1f}%)")
    print(f"  Test: {len(test_dataset)} ({100*test_size/total_size:.1f}%)")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Windows compatibility
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    # Initialize model
    model = ImprovedPositionEvaluator(input_size=128, hidden_size=256)

    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")

    # Train
    trainer = Trainer(model, train_loader, val_loader, lr=lr, device=device)
    trained_model = trainer.train(epochs=epochs)

    # Test final model
    print("\n" + "="*60)
    print("Testing final model...")
    trained_model.eval()
    test_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for features, scores in test_loader:
            features = features.to(device)
            scores = scores.to(device)

            predictions = trained_model(features)
            loss = nn.MSELoss()(predictions, scores)
            test_loss += loss.item()
            num_batches += 1

    test_loss /= num_batches
    print(f"Test Loss: {test_loss:.4f}")
    print("="*60)

    return trained_model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train AgentB position evaluator')
    parser.add_argument('--data', type=str, default='training_data.json',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')

    args = parser.parse_args()

    print("="*60)
    print("AgentB Neural Network Training")
    print("="*60)

    model = train_robust_model(
        data_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print("\n✓ Training complete! Use 'best_evaluator.pth' in your agent.")
