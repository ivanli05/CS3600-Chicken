# AgentB Training Guide: Building a Robust Model Like Stockfish

## Table of Contents
1. [Overview](#overview)
2. [Handling Changing Factors (Trapdoors)](#handling-trapdoors)
3. [Robustness Techniques from Stockfish](#stockfish-techniques)
4. [Complete Training Pipeline](#training-pipeline)
5. [Integrating Trained Model](#integration)

---

## Overview

**Goal**: Train a neural network that can evaluate positions accurately, even with:
- Unknown trapdoor locations (hidden information)
- Varying game states
- Noisy sensor data

**Approach**: Use deep search as "ground truth" to train the NN to predict search results.

---

## Handling Changing Factors (Trapdoors)

### The Challenge

Trapdoors are **hidden information** that changes:
- **Location is random** at game start
- **Sensor readings are probabilistic** (50% hear, 30% feel, etc.)
- **Beliefs update** throughout the game

### Solution 1: Feature Engineering

**Encode trapdoor information as features, not positions:**

```python
# BAD: Hard-code trapdoor position
features[0] = trapdoor_x  # This changes every game!

# GOOD: Encode probability distribution
features[50] = trapdoor_tracker.get_danger_score(my_location)  # Normalized probability
features[51] = trapdoor_tracker.get_danger_score(enemy_location)
features[52] = average_danger_in_neighborhood(my_location)
```

**Why this works:**
- The NN learns "how to handle danger", not "where trapdoors are"
- Generalizes to any trapdoor configuration
- Works even with incomplete information

### Solution 2: Data Augmentation

**Generate training positions with varying trapdoor configurations:**

```python
def generate_diverse_trapdoor_positions():
    """
    For each game position, create multiple training examples
    with different trapdoor beliefs.
    """
    base_board = generate_random_position()

    # Generate 5 different trapdoor scenarios
    for _ in range(5):
        # Simulate different sensor histories
        sensor_history = generate_random_sensors()

        # Update trapdoor beliefs
        trapdoor_tracker.reset()
        for location, sensors in sensor_history:
            trapdoor_tracker.update_beliefs(location, sensors)

        # Extract features WITH these beliefs
        features = extract_features(base_board, trapdoor_tracker)

        # Same position, different trapdoor beliefs!
        yield (features, ground_truth_score)
```

**Result**: NN sees same position with different trapdoor configurations → learns to be robust.

### Solution 3: Bayesian Features

**Encode uncertainty directly in features:**

```python
# Instead of "trapdoor is at (3,4)"
# Use "probability distribution over squares"

def encode_trapdoor_beliefs(tracker):
    features = []

    # Top 3 most likely trapdoor positions
    likely_positions = tracker.get_most_likely_trapdoors(3)

    for (x, y), prob in likely_positions:
        features.append(x / 8.0)  # Normalized x
        features.append(y / 8.0)  # Normalized y
        features.append(prob)      # Probability

    # Entropy of distribution (how uncertain are we?)
    entropy = calculate_entropy(tracker.prob_white + tracker.prob_black)
    features.append(entropy)

    return features
```

**Benefit**: NN learns "when I'm uncertain about trapdoors, be more cautious"

---

## Stockfish-Inspired Robustness Techniques

### 1. **Deep Search as Ground Truth**

**Stockfish approach:**
- Generate positions
- Evaluate with search depth 20-40
- Train NN to predict search results

**Our implementation:**
```python
def generate_labels(position):
    # Use deep search (depth 6-8) as "truth"
    score = minimax_search(position, depth=8, time_limit=30.0)

    # Normalize to [-1, 1]
    normalized_score = tanh(score / 1000.0)

    return normalized_score
```

**Why depth 8?**
- Depth 4: Used during actual games (fast)
- Depth 8: Used for training labels (accurate)
- NN learns to "think ahead" like depth 8, but runs as fast as depth 0!

### 2. **Self-Play Data Generation**

**Stockfish generates billions of positions from self-play.**

**Our version:**
```python
def self_play_data_generation(num_games=1000):
    for game in range(num_games):
        board = Board()

        # Play a full game
        while not board.is_game_over():
            # Use current best agent
            move = agent.play(board)
            board = board.forecast_move(move)

            # Save position for training
            if random.random() < 0.1:  # Sample 10% of positions
                save_position_for_training(board)
```

**Diversity tricks:**
1. **Random openings** (first 5 moves random)
2. **Temperature sampling** (add randomness to move selection)
3. **Varied opponents** (play against different agents)

### 3. **Residual Networks (Like AlphaZero)**

**Architecture:**
```
Input (128 features)
    ↓
Dense + BatchNorm + ReLU
    ↓
[Residual Block 1] ←┐ (skip connection)
    ↓                │
Dense + BN          │
    ↓                │
Dense + BN          │
    ↓                │
Add skip ──────────┘
    ↓
ReLU
    ↓
[Residual Block 2] (same structure)
    ↓
Output (1 value)
```

**Why residual connections?**
- Prevents vanishing gradients
- Allows deeper networks
- Better feature learning

### 4. **Regularization Stack**

**Prevent overfitting (like Stockfish does):**

```python
# 1. Dropout (random neuron deactivation)
nn.Dropout(0.2)

# 2. Weight Decay (L2 regularization)
optimizer = AdamW(params, weight_decay=1e-4)

# 3. Batch Normalization (stabilizes training)
nn.BatchNorm1d(hidden_size)

# 4. Data Augmentation (add noise)
features = features + torch.randn_like(features) * 0.02

# 5. Early Stopping (stop when validation plateaus)
if val_loss not improving for 15 epochs:
    stop_training()
```

### 5. **Learning Rate Scheduling**

**Stockfish reduces learning rate as training progresses:**

```python
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,      # Halve LR when plateauing
    patience=5,      # Wait 5 epochs before reducing
    verbose=True
)

# Start: lr = 0.001
# After plateau: lr = 0.0005
# After another: lr = 0.00025
# etc.
```

**Why?**
- Early training: Large steps to find general patterns
- Late training: Small steps to fine-tune

### 6. **Robust Loss Function**

**Stockfish uses loss functions resistant to outliers:**

```python
# BAD: Mean Squared Error (sensitive to outliers)
loss = (prediction - target)^2

# GOOD: Smooth L1 / Huber Loss
if abs(prediction - target) < 1:
    loss = 0.5 * (prediction - target)^2
else:
    loss = abs(prediction - target) - 0.5
```

**Benefit**: A few bad labels don't ruin training.

---

## Complete Training Pipeline

### Step 1: Generate Data

```bash
cd AgentB/training
python generate_training_data.py
```

**Output**: `training_data.json` with ~5,000-10,000 positions

**What it contains:**
- Position features (128-dimensional vectors)
- Deep search evaluations (ground truth)
- Metadata (turn number, egg difference, etc.)

### Step 2: Train Model

```bash
python train_network.py --epochs 200 --batch-size 128 --lr 0.001
```

**Timeline:**
- **5,000 positions**: ~30 minutes training
- **50,000 positions**: ~3 hours training
- **500,000 positions**: ~24 hours training (Stockfish-level data)

**Monitor training:**
```
Epoch 1/200 | Train Loss: 0.1523 | Val Loss: 0.1489 | LR: 0.001000
Epoch 2/200 | Train Loss: 0.1201 | Val Loss: 0.1156 | LR: 0.001000
...
Epoch 45/200 | Train Loss: 0.0342 | Val Loss: 0.0389 | LR: 0.000500
  ✓ Saved new best model (val_loss=0.0389)
```

### Step 3: Validate Performance

**Test on held-out positions:**

```python
# Compare NN predictions vs. deep search
test_positions = load_test_set()

for pos in test_positions:
    nn_eval = model(extract_features(pos))
    deep_search_eval = minimax(pos, depth=8)

    error = abs(nn_eval - deep_search_eval)
    print(f"NN: {nn_eval:.3f} | Search: {deep_search_eval:.3f} | Error: {error:.3f}")
```

**Good performance:**
- Average error < 0.1
- Correlation > 0.85 with deep search

### Step 4: Integrate into Agent

**Add to agent.py:**

```python
if TORCH_AVAILABLE:
    self.nn_evaluator = PositionEvaluator(input_size=128, hidden_size=256)
    self.nn_evaluator.load_weights('best_evaluator.pth')
    self.nn_evaluator.eval()
```

**Use in evaluation:**

```python
def evaluate_position(self, board):
    # Extract features
    features = extract_features(board, self.trapdoor_tracker)

    # Get NN prediction
    with torch.no_grad():
        nn_score = self.nn_evaluator(features).item() * 1000.0

    # Combine with heuristics
    heuristic_score = evaluate_heuristics(board)

    # Weighted combination
    final_score = 0.7 * nn_score + 0.3 * heuristic_score

    return final_score
```

---

## Advanced: Handling Trapdoors Robustly

### Problem: Overfitting to Trapdoor Patterns

**Bad scenario:**
- NN learns "trapdoors are usually at (3,3) and (5,5)"
- Fails when trapdoors are at (1,7) and (6,2)

### Solution: Trapdoor-Agnostic Features

**Instead of encoding absolute trapdoor information:**

```python
# BAD
features[10] = prob_white[3, 3]  # Specific square
features[11] = prob_black[5, 5]

# GOOD
features[10] = max_danger_in_quadrant_1
features[11] = max_danger_in_quadrant_2
features[12] = danger_at_my_location
features[13] = danger_at_enemy_location
features[14] = entropy_of_belief  # How uncertain?
```

**Result**: NN learns "avoid high-danger areas" not "avoid (3,3)"

### Advanced: Multi-Task Learning

**Train NN to predict multiple things simultaneously:**

```python
class MultiTaskEvaluator(nn.Module):
    def forward(self, x):
        shared_features = self.backbone(x)

        # Task 1: Predict position value
        value = self.value_head(shared_features)

        # Task 2: Predict trapdoor locations (auxiliary task)
        trapdoor_probs = self.trapdoor_head(shared_features)

        return value, trapdoor_probs

# Loss function
loss = value_loss + 0.1 * trapdoor_loss
```

**Benefit**: Learning to predict trapdoors improves value prediction!

---

## Checklist: Is My Model Robust?

✅ **Data Diversity**
- [ ] Positions from all game phases (early, mid, late)
- [ ] Various trapdoor configurations
- [ ] Different egg count differences
- [ ] Multiple trapdoor belief states

✅ **Architecture**
- [ ] Residual connections for depth
- [ ] Batch normalization for stability
- [ ] Dropout for regularization
- [ ] Appropriate output activation (tanh for [-1,1])

✅ **Training**
- [ ] Train/val/test split (75%/15%/10%)
- [ ] Early stopping to prevent overfitting
- [ ] Learning rate scheduling
- [ ] Data augmentation (noise injection)

✅ **Evaluation**
- [ ] Test on unseen positions
- [ ] Compare against deep search
- [ ] Test with different trapdoor scenarios
- [ ] Measure inference speed (should be <1ms)

✅ **Integration**
- [ ] Graceful fallback if model fails
- [ ] Combine NN with heuristics (hybrid approach)
- [ ] Profile performance (CPU/memory usage)

---

## Quick Start Commands

```bash
# 1. Generate training data (takes ~2 hours for 10k positions)
cd training
python generate_training_data.py

# 2. Train model (takes ~1 hour for 10k positions, 100 epochs)
python train_network.py --epochs 100 --batch-size 128

# 3. Copy trained weights to agent directory
cp best_evaluator.pth ../evaluator_weights.pth

# 4. Test agent
cd ../../../
python engine/run_local_agents.py AgentB AgentB

# 5. Verify NN is loaded
# Look for: "✓ Loaded trained evaluator weights"
```

---

## Comparison: Stockfish vs. Our Approach

| Aspect | Stockfish | AgentB |
|--------|-----------|--------|
| **Data Size** | Billions of positions | Thousands to millions |
| **Search Depth** | 20-40 ply (training) | 6-8 ply (training) |
| **Architecture** | NNUE (efficiently updatable) | Standard feedforward + residual |
| **Training Time** | Days to weeks | Hours to days |
| **Features** | 40k+ (incremental updates) | 128 (full evaluation) |
| **Robustness** | Trained on varied positions | Data augmentation + regularization |

**Key Takeaway**: We use the same *principles* but scale to our game's complexity.

---

## Troubleshooting

### "Model overfits training data"
- **Solution**: More data, stronger regularization, earlier stopping

### "NN makes worse moves than heuristics"
- **Solution**: Need more training data, or use hybrid (0.5 NN + 0.5 heuristic)

### "Training is too slow"
- **Solution**: Reduce model size, use GPU, parallelize data generation

### "Model doesn't handle trapdoors well"
- **Solution**: Generate more training examples with varied trapdoor beliefs

---

## Next Steps

1. **Generate initial dataset** (5k positions)
2. **Train baseline model** (100 epochs)
3. **Test against heuristic-only agent**
4. **If NN wins**: Generate more data (50k positions), train bigger model
5. **If heuristics win**: Improve features, try hybrid approach
6. **Iterate** until competitive!

**Remember**: Stockfish took *years* to develop. Start simple, iterate, improve!
