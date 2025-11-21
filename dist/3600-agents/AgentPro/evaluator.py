"""
Neural Network Position Evaluator for AgentB

This module contains the position evaluation neural network.
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    import numpy as np


if TORCH_AVAILABLE:
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


    class PositionEvaluator(nn.Module):
        """
        Enhanced evaluator with residual connections.
        Trained on H100 GPU with 50k positions.
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

        def load_weights(self, path='best_evaluator.pth'):
            """Load pre-trained weights"""
            try:
                import os
                # Look for the model in the same directory as this file
                model_dir = os.path.dirname(os.path.abspath(__file__))
                full_path = os.path.join(model_dir, path)

                # Load checkpoint
                checkpoint = torch.load(full_path, map_location='cpu')

                # Extract model weights from checkpoint
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    # Full checkpoint with training state
                    self.load_state_dict(checkpoint['model_state_dict'])
                else:
                    # Just model weights
                    self.load_state_dict(checkpoint)

                return True
            except (FileNotFoundError, Exception):
                return False
else:
    class PositionEvaluator:
        """
        Simple linear evaluator when PyTorch is not available.
        Falls back to basic linear combination.
        """

        def __init__(self, input_size=64, hidden_size=128):
            # Fallback: simple weights
            self.weights = np.random.randn(input_size) * 0.01

        def forward(self, x):
            # Simple linear evaluation
            return np.dot(x, self.weights)

        def load_weights(self, path='evaluator_weights.npy'):
            """Load numpy weights if available"""
            try:
                self.weights = np.load(path)
                return True
            except:
                return False
