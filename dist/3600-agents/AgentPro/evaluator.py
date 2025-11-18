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
    class PositionEvaluator(nn.Module):
        """
        Neural network for evaluating board positions.
        Takes board features and outputs a score.
        """

        def __init__(self, input_size=64, hidden_size=128):
            super().__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
            self.fc3 = nn.Linear(hidden_size // 2, 1)
            self.relu = nn.ReLU()
            self.tanh = nn.Tanh()

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            x = self.tanh(self.fc3(x))
            return x

        def load_weights(self, path='evaluator_weights.pth'):
            """Load pre-trained weights"""
            try:
                self.load_state_dict(torch.load(path))
                print(f"✓ Loaded trained evaluator weights from {path}")
                return True
            except FileNotFoundError:
                print(f"⚠ No trained weights found at {path}, using random initialization")
                return False
            except Exception as e:
                print(f"⚠ Error loading weights: {e}")
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
                print(f"✓ Loaded weights from {path}")
                return True
            except:
                print(f"⚠ No weights found, using random initialization")
                return False
