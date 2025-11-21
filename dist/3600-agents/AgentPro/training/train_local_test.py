"""
Quick Local Test - Training Script

Tests training on the 100 positions we just generated.
Runs only 5 epochs to verify everything works.
"""

import sys
import os
import time
import yaml
import json
from pathlib import Path

# Add paths to import agent modules
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)  # AgentPro/

sys.path.insert(0, agentpro_dir)  # For 'from evaluator import PositionEvaluator'

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import numpy as np

from evaluator import PositionEvaluator


class ChickenDataset(Dataset):
    """Dataset for chicken game positions"""

    def __init__(self, data_file):
        print(f"Loading data from {data_file}...")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        print(f"Loaded {len(self.data)} positions")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.FloatTensor(item['features'])
        score = torch.FloatTensor([item['score']])
        return features, score


def main():
    print(f"\n{'='*60}")
    print("Quick Training Test")
    print(f"{'='*60}\n")

    # Check if data exists
    if not os.path.exists('training_data_local.json'):
        print("❌ Error: training_data_local.json not found!")
        print("Run: python3.10 generate_data_local.py first")
        return

    # Load dataset
    dataset = ChickenDataset('training_data_local.json')

    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * 0.15)
    val_size = int(total_size * 0.15)
    train_size = total_size - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Dataset splits:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    print(f"  Test:  {len(test_dataset)}\n")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Create model
    device = torch.device('cpu')
    model = PositionEvaluator(input_size=128, hidden_size=128).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Device: {device}\n")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.SmoothL1Loss()

    # Training loop (just 5 epochs for quick test)
    print(f"{'='*60}")
    print("Training for 5 epochs (quick test)")
    print(f"{'='*60}\n")

    best_val_loss = float('inf')

    for epoch in range(5):
        # Train
        model.train()
        train_loss = 0.0
        for features, scores in train_loader:
            features = features.to(device)
            scores = scores.to(device)

            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, scores)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, scores in val_loader:
                features = features.to(device)
                scores = scores.to(device)

                predictions = model(features)
                loss = criterion(predictions, scores)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        # Print progress
        print(f"Epoch {epoch+1}/5 | Train: {train_loss:.4f} | Val: {val_loss:.4f}", end="")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'epoch': epoch
            }, 'test_evaluator.pth')
            print(" ✓ Best!")
        else:
            print()

    print(f"\n{'='*60}")
    print("Training test complete!")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: test_evaluator.pth")
    print(f"\n✅ Training pipeline works!")
    print(f"\nReady for PACE deployment with full settings:")
    print(f"  - 50,000 positions")
    print(f"  - 200 epochs")
    print(f"  - H100 GPU")
    print(f"  - Mixed precision (FP16)")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
