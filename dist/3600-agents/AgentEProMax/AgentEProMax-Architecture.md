# AgentEProMax - Architecture & Strategy

## Overview
**AgentEProMax** is an aggressive territory-control agent for the Chicken game that combines Patrick's "Harvester" philosophy with advanced trapdoor tracking and strategic planning. The agent focuses on maximizing egg-laying opportunities while maintaining calculated risk management.

## Core Philosophy
- **Eggs First**: Prioritize laying eggs over territory expansion
- **Calculated Aggression**: Balance between scoring and safety (not paranoid, not reckless)
- **Endgame Urgency**: Double scoring weight in final 10 turns
- **Natural Turd Usage**: Let minimax decide turd placement based on strategic value

---

## Architecture Components

### 1. **Main Agent (agent.py)**
#### Hyperparameters
```python
MAXDEPTH = 7        # Iterative deepening search depth
TIME_LIMIT = 0.8    # Use 80% of allocated time per move
```

#### Decision Flow
1. **Trapdoor Belief Update**
   - Update probabilistic beliefs about trapdoor locations
   - Detect teleportation events (falling into trapdoors)
   - Mark confirmed trapdoors

2. **Safety Filtering**
   - Base risk threshold: 0.35 (increases to 0.45 after turn 60)
   - Filter out moves with unacceptable trapdoor risk
   - Fallback: Choose best risky move if no safe options exist

3. **Search Execution**
   - Run iterative deepening minimax search on safe moves only
   - Return best move found within time limit

---

### 2. **Heuristic Evaluation (heuristics.py)**

#### Position Evaluation Components

**1. Safety Assessment** (Highest Priority)
```python
Known trapdoor: -1,000,000 (instant loss)
Danger > 0.25:  -(danger * 5,000)
```

**2. Material Score** (Egg Count)
```python
Base weight: 150 points per egg
Endgame multiplier: 2x when turns_left < 10
Uses get_eggs_laid() which accounts for corner 3x multiplier
```

**3. Reachability Analysis** (BFS depth 20)
- **Potential**: Empty squares with correct parity (actual egg opportunities)
  - Weight: 25 per square
  - Corner squares: 3x value
- **Territory**: Total reachable squares (mobility/safety)
  - Weight: 5 per square

**4. Corner Positioning Bonus**
```python
+50 if standing in corner and can lay egg
```

**5. Distance Bonus** (Anti-idling)
```python
(max_dist - dist_to_nearest_egg_spot) * 3.0
Prevents wandering aimlessly in controlled territory
```

**6. Turd Penalty** (Removed from evaluation)
- No penalty in position evaluation
- Minimax decides turd value naturally

#### Move Ordering Heuristic
Fast evaluation for alpha-beta pruning:

```python
Egg moves:     +1000 (corners: +3000)
Plain moves:   0
Turd moves:    -50 (close range) to -200 (far range)
Known trapdoor: -50,000
Danger > 0.25: -(danger * 5,000)
```

---

### 3. **Search Engine (search_engine.py)**

#### Key Features

**Iterative Deepening**
- Start at depth 1, progressively search deeper
- Stop when time runs out
- Use best result from deepest completed iteration

**Transposition Table**
- Cache evaluated positions (max 100,000 entries)
- Avoid re-evaluating same board state
- Track cache hits for performance monitoring

**Equal Time Allocation**
```python
Estimated total moves: 60
Time per move = (total_time / moves_remaining) * 0.95
Cap at 30 seconds to avoid timeout
```

**Alpha-Beta Pruning**
- Minimax with alpha-beta cutoffs
- Move ordering improves pruning efficiency
- Perspective flipping for enemy turns

---

### 4. **Trapdoor Tracker (trapdoor_tracker.py)**

#### Bayesian Belief System
Maintains separate probability distributions for:
- **White trapdoor** (even parity squares)
- **Black trapdoor** (odd parity squares)

#### Sensor Model
```python
Adjacent:  P(hear)=0.50, P(feel)=0.30
Diagonal:  P(hear)=0.25, P(feel)=0.15
Zone 3:    P(hear)=0.10, P(feel)=0.00
Far:       P(hear)=0.00, P(feel)=0.00
```

#### Prior Distribution
- Higher probability for interior squares
- Zero probability at edges (known safe)
- Distance-from-edge weighting

#### Belief Updates
- Bayesian update on each sensor reading
- Set probability to 0 for visited squares
- Mark confirmed trapdoors at 100% probability

---

## Key Strategic Improvements

### 1. **Egg Potential vs Territory**
**Before**: Counted all reachable squares as valuable
**Now**: Only counts EMPTY squares with matching parity
- Uses `chicken.even_chicken` for parity
- Distinguishes "I control 50 squares" vs "I can lay 30 eggs"

### 2. **Distance-Based Movement**
**Added**: BFS to find nearest valid egg spot
- Rewards moving closer to scoring opportunities
- Prevents "plain move" idling in controlled territory

### 3. **Lethal Zone Avoidance**
**Added**: 3x3 danger zones around enemy turds
- More conservative pathfinding
- Avoids getting stuck near enemy blocks

### 4. **Aggressive Egg Priority**
**Material Scoring**: Uses `get_eggs_laid()` directly
- Automatically accounts for corner 3x multiplier
- Simpler and more accurate than manual counting

**Endgame Desperation**: 2x egg weight when turns_left < 10
- Pushes for final eggs aggressively
- Matches Patrick's urgency strategy

### 5. **Natural Turd Placement**
**Removed**: Pre-search turd evaluation logic
**Removed**: Turd penalty from position evaluation
**Kept**: Turd penalty in move ordering only (-50 to -200)
- Minimax explores turds naturally
- Only chosen when strategic value is clear

---

## Tunable Parameters

### Search Configuration
```python
MAXDEPTH = 7         # Adjust for performance vs depth trade-off
TIME_LIMIT = 0.8     # Fraction of time to use (0.8 = 80%)
```

### Risk Tolerance
```python
base_threshold = 0.35           # Early game risk acceptance
late_threshold = 0.45           # Late game (turn 60+)
danger_penalty = 5000           # Penalty multiplier for risky squares
```

### Evaluation Weights
```python
BASE_W_EGG = 150.0              # Egg value (2x in endgame)
BASE_W_POTENTIAL = 25.0         # Future egg opportunities
BASE_W_SPACE = 5.0              # Mobility/territory
corner_bonus = 50.0             # Standing in corner
distance_weight = 3.0           # Distance to nearest egg
```

### Move Ordering
```python
egg_bonus = 1000                # Egg move priority
corner_egg_bonus = 2000         # Corner egg extra priority
turd_penalty_close = -50        # Turd when dist ≤ 3
turd_penalty_far = -200         # Turd when dist > 3
```

---

## Performance Characteristics

### Strengths
- **Aggressive egg-laying**: Prioritizes scoring over pure territory
- **Smart trapdoor avoidance**: Bayesian tracking with 5000x danger penalty
- **Efficient search**: Transposition table + move ordering
- **Anti-idling**: Distance bonus keeps agent moving toward eggs
- **Endgame urgency**: Doubles egg value in final turns

### Weaknesses
- **Moderate depth**: Depth 7 may miss deep tactical opportunities
- **Conservative around turds**: 3x3 lethal zones might be overly cautious
- **No pre-search turd logic**: Relies entirely on minimax for turd placement

### Time Complexity
- **Per move**: O(b^d) where b ≈ 8 (branching), d = 7 (depth)
- **Transposition table**: Reduces effective branching significantly
- **Alpha-beta pruning**: Can cut search tree by ~50% with good move ordering

---

## Comparison with Patrick's Agent

| Feature | Patrick | AgentEProMax |
|---------|---------|--------------|
| **Search Depth** | 10 | 7 |
| **Egg Weight** | 150 (2x endgame) | 150 (2x endgame) |
| **Potential Weight** | 25 | 25 |
| **Territory Weight** | 5 | 5 |
| **Risk Penalty** | 2000 | 5000 |
| **Turd Logic** | Move ordering only | Move ordering only |
| **Corner Bonus** | None | +50 position bonus |
| **Distance Bonus** | (14 - dist) * 2.0 | (16 - dist) * 3.0 |
| **Transposition Table** | No | Yes (100k entries) |
| **Trapdoor System** | Numpy-based Bayesian | Same approach |

---

## Future Improvements

1. **Adaptive Depth**: Increase depth in critical positions (few remaining squares)
2. **Opening Book**: Pre-computed strong opening moves
3. **Turd Clustering**: Detect patterns where multiple turds create strong blocks
4. **Quiescence Search**: Search deeper when position is unstable (near enemy)
5. **Time Banking**: Use less time in simple positions, more in complex ones

---

## Code Structure

```
AgentEProMax/
├── agent.py              # Main agent logic & decision flow
├── heuristics.py         # Position evaluation & move ordering
├── search_engine.py      # Iterative deepening minimax + transposition table
└── trapdoor_tracker.py   # Bayesian trapdoor probability tracking
```

---

## Summary

**AgentEProMax** is designed to be an **aggressive harvester** that:
1. Lays eggs whenever safe and valuable
2. Avoids trapdoors with calculated risk tolerance
3. Maximizes actual scoring opportunities (not just territory)
4. Searches efficiently with transposition tables
5. Adapts strategy in endgame with 2x egg urgency

The agent balances Patrick's aggressive scoring philosophy with intelligent safety mechanisms and efficient search techniques.