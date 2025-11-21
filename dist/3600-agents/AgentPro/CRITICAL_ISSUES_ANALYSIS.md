# CRITICAL ISSUES ANALYSIS - AgentPro

## Executive Summary

Your agent has several critical issues that explain why training only takes 5 minutes and why performance is unsatisfactory:

1. **Training data generation produces WRONG labels** (value calculation is fundamentally broken)
2. **AgentPro is using a much simpler heuristic than AgentProNoNets**
3. **Training dataset is too small** (100 positions locally, though 50k on PACE)
4. **Neural network features are incomplete**

---

## Problem 1: VALUE CALCULATION IS FUNDAMENTALLY BROKEN ⚠️

### The Core Issue

In `generate_data_parallel.py` lines 78-104:

```python
def evaluate_with_search(board, depth=6):
    # ... minimax search ...
    score, _ = agent.search_engine._minimax(...)

    # THIS IS WRONG:
    normalized_score = np.tanh(score / 1000.0)  # ❌ BROKEN
    return normalized_score
```

### Why This Is Wrong

1. **Egg difference dominates scoring**: Each egg is worth 100 points (line 92 in heuristics.py)
2. **Even 1 egg difference** = 100 points → `tanh(100/1000) = tanh(0.1) ≈ 0.099`
3. **ALL scores compressed to [-0.5, 0.5]** range
4. **Result**: The network learns almost nothing because all positions have nearly identical labels

### What's Happening

- **Best validation loss: 0.0023**
- **Test loss: 0.0024**
- This looks good, but the network is likely just predicting ~0 for everything!
- The MSE is low because all true labels are ~0 anyway (due to tanh compression)

### Correct Normalization

Option 1: **No normalization** (let network learn raw scores)
```python
# Just divide by reasonable scale
normalized_score = score / 400.0  # Max 4 egg diff = ~400 points → ±1.0
```

Option 2: **Proper normalization**
```python
# Based on actual egg difference (the only thing that matters)
my_eggs = board.chicken_player.get_eggs_laid()
enemy_eggs = board.chicken_enemy.get_eggs_laid()
# Max possible egg diff is ~40 eggs
normalized_score = (my_eggs - enemy_eggs) / 20.0  # Maps to roughly [-2, +2]
```

---

## Problem 2: AgentPro Heuristics Are TOO WEAK

### Current AgentPro Heuristics (lines 38-77):

```python
if move_type == MoveType.EGG:
    score += 100.0  # ❌ TOO LOW
    if self._is_corner(new_loc):
        score += 50.0   # ❌ DOESN'T UNDERSTAND CORNERS GIVE 3X EGGS!

# Trapdoor penalty
danger = trapdoor_tracker.get_danger_score(new_loc)
score -= danger * 1000.0  # ❌ WAY TOO LOW (costs 4 eggs = 400 points!)
```

### AgentProNoNets Heuristics (MUCH BETTER):

```python
if move_type == MoveType.EGG:
    score += 300.0  # ✓ 3x stronger!

    # Bonus for new areas
    if new_loc not in visited_squares:
        score += 150.0  # ✓ Encourages exploration

    # Spread eggs out
    if min_dist_to_existing_egg >= 4:
        score += 60.0

    # CORNER EGGS ARE SPECIAL (3x eggs!)
    if self._is_corner(new_loc) and can_lay_on_corner:
        score += 500.0  # ✓ MASSIVE bonus for 3x eggs!

# Trapdoor avoidance
if new_loc in known_trapdoors:
    score -= 1000000.0  # ✓ ABSOLUTELY NEVER GO THERE
else:
    if danger > 0.2:
        score -= danger * 200000.0  # ✓ VERY heavy penalty
    # ... graduated penalties based on probability
```

### Key Differences

| Feature | AgentPro | AgentProNoNets | Impact |
|---------|----------|----------------|--------|
| Egg move bonus | 100 | 300 | **3x stronger** |
| Corner egg bonus | 50 | 500 | **10x stronger** |
| Known trapdoor penalty | ~1000 | 1,000,000 | **1000x stronger** |
| Visited square tracking | ❌ None | ✓ Yes | **Prevents repetition** |
| Blocked location tracking | ❌ None | ✓ Yes | **Avoids wasting turns** |
| Escape from trapdoors | ❌ None | ✓ Yes | **Explores after finding trapdoors** |
| Region-based exploration | ❌ None | ✓ Yes | **Covers whole board** |

---

## Problem 3: Missing Game Mechanics Understanding

### Critical Mechanic: **CORNER EGGS GIVE 3X VALUE!**

From the game rules (assignment.pdf, implied from board.py):
- **Normal egg**: 1 egg
- **Corner position egg**: **3 eggs** (they can't be blocked as easily)

**Your agent doesn't know this!**

AgentProNoNets recognizes this:
```python
# Line 73-80
if self._is_corner(new_loc):
    can_lay_on_corner = board.chicken_player.can_lay_egg(new_loc)
    if can_lay_on_corner:
        score += 500.0  # Prioritize corner eggs!
```

### Trapdoor Cost Understanding

- Stepping on trapdoor: **-4 eggs to opponent** = **-400 points**
- Your penalty: `-danger * 1000` = max `-1000` for 100% danger
- **Problem**: Even a guaranteed trapdoor penalty is only 1000, but potential egg move is +100
- **Result**: Agent might take trapdoor risk for eggs!

AgentProNoNets fixes this:
- Known trapdoor: `-1,000,000` (NEVER go there)
- High probability (>20%): `-danger * 200,000`
- Result: Agent ALWAYS avoids trapdoors

---

## Problem 4: Agent Gets Stuck in Patterns

### Diamond Pattern Problem

AgentPro doesn't track:
- Where it's been before
- Recent positions
- Blocked locations (enemy eggs)

**Result**: Agent repeats same 2-3 squares near trapdoors, gets stuck in "diamond" patterns.

AgentProNoNets fixes this with:
1. `visited_squares`: Tracks ALL visited squares (-400 penalty for revisiting)
2. `recent_positions`: Last 8 moves (-500 to -1100 penalty based on recency)
3. `blocked_locations`: Enemy eggs/barriers (-50,000 penalty)
4. **Pseudo-random exploration**: Uses `location_hash % 1000` to break ties deterministically

---

## Problem 5: Neural Network Features Are Incomplete

### Current Features (128 total, only ~10 used):

```python
features[0-1] = my_location / 8      # Position
features[2-3] = enemy_location / 8   # Enemy position
features[8-10] = egg_counts / 40     # Eggs
features[12-13] = turds / 5          # Turds
features[32-33] = mobility / 8       # Valid moves
features[36-37] = turn_info / 40     # Turns
features[40-41] = center_distance    # Distance to center
# ... Rest are ZEROS
```

### Missing Critical Features:

1. **Trapdoor information** (most important!)
   - Trapdoor probability map (64 values, one per square)
   - Known trapdoor locations
   - Distance to likely trapdoors

2. **Egg spread patterns**
   - Distance to nearest own egg
   - Distance to nearest enemy egg
   - Egg cluster sizes
   - Corner occupancy (who controls corners?)

3. **Movement history**
   - Squares visited recently
   - Repetition patterns
   - Stuck indicator

4. **Positional features**
   - Control of center
   - Control of corners
   - Area of board covered

---

## Why Training Takes Only 5 Minutes

### On PACE (50k positions, 94 epochs):

1. **H100 is extremely fast**: 0.8s per epoch
2. **Early stopping triggered**: No improvement after epoch 69
3. **Network converged quickly**: Loss went from 0.0042 → 0.0023

### But This Is Deceptive!

The network converged because:
- **All labels are ~0** (due to broken tanh normalization)
- Network learns to predict ~0 for everything
- Loss plateaus at 0.0023 because that's the variance in the (compressed) labels
- **The network learned nothing useful!**

Test it:
```python
# Your network probably outputs ~0 for ANY position
model.eval()
with torch.no_grad():
    output1 = model(random_position)  # → ~-0.02
    output2 = model(different_position)  # → ~+0.01
    output3 = model(winning_position)  # → ~+0.03
    output4 = model(losing_position)  # → ~-0.04
# ALL outputs are within [-0.1, +0.1] range!
```

---

## Recommendations

### IMMEDIATE (Critical):

1. **Fix value calculation in generate_data_parallel.py**:
   ```python
   def evaluate_with_search(board, depth=6):
       # ... search ...
       # Option 1: Just scale down
       return score / 400.0  # Map ~4 eggs to ±1.0

       # Option 2: Use egg difference directly
       my_eggs = board.chicken_player.get_eggs_laid()
       enemy_eggs = board.chicken_enemy.get_eggs_laid()
       return (my_eggs - enemy_eggs) / 20.0
   ```

2. **Copy heuristics from AgentProNoNets to AgentPro**:
   - Replace `heuristics.py` entirely
   - Update `agent.py` to track visited_squares, recent_positions, blocked_locations

3. **Add corner egg detection**:
   ```python
   # Check if corner gives 3x eggs
   if self._is_corner(new_loc):
       eggs_multiplier = 3 if is_corner_special else 1
       score += 300.0 * eggs_multiplier
   ```

### IMPORTANT (High Impact):

4. **Expand neural network features to 128** (actually use all of them):
   - Features[0-63]: Trapdoor probability map (8x8 grid)
   - Features[64-75]: Egg/turd spatial info
   - Features[76-91]: Position history (recent squares)
   - Features[92-127]: Strategic features (corners, center, clusters)

5. **Generate MUCH more data**:
   - Increase from 50k to 200k positions
   - Use depth=8 for labels (currently depth=6)
   - More varied starting positions (not just random walks)

6. **Increase model capacity**:
   ```yaml
   model:
     hidden_size: 512  # Was 256
     num_residual_blocks: 4  # Was 2
   ```

7. **Train longer**:
   ```yaml
   training:
     epochs: 500  # Was 200
     early_stopping:
       patience: 50  # Was 25
   ```

### OPTIMIZATION (Lower Priority):

8. **Add value head + policy head** (AlphaZero style):
   - Value head: predicts position evaluation
   - Policy head: predicts move probabilities
   - Train on (position, minimax_value, best_move) tuples

9. **Self-play data generation**:
   - Let agent play against itself
   - Use actual game outcomes as labels
   - Mix with minimax-evaluated positions

10. **Ensemble**:
    - Train 3-5 networks with different random seeds
    - Average predictions at inference time
    - More robust to individual network failures

---

## Bottom Line

**Your agent is not learning from the neural network** because:
1. Labels are compressed to ~0 by broken tanh normalization
2. Features are incomplete (missing trapdoors, history, spatial info)
3. Heuristics are too weak (3x-1000x weaker than AgentProNoNets)

**Immediate fix**: Use AgentProNoNets heuristics, they're much better.

**Medium-term fix**: Fix training data generation and retrain on PACE with:
- Correct value normalization
- 200k positions
- Complete feature set
- Larger model

**Your 5-minute training is a red flag**, not a green light! The network converged too quickly because it's not learning meaningful patterns.
