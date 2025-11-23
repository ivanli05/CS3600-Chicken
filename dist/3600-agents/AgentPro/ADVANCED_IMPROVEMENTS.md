# Advanced Improvements for AgentPro

## ✅ Implemented (Just Added)

### 1. **Transposition Table**
**Impact:** 2-5× speedup by caching evaluated positions

**How it works:**
- Hash each board state (positions, eggs, turds)
- Store: (depth, score, best_move)
- Reuse if we encounter same position at equal/greater depth
- Cleared at start of each game

**Benefit:** Avoids re-computing identical positions from different move orders.

---

## 🚀 More High-Impact Improvements

### 2. **Null Move Pruning** (Medium difficulty, 20-30% speedup)

Skip opponent's turn to detect if position is already winning/losing.

**Add to `search_engine.py` in `_minimax()`:**

```python
# After transposition table lookup, before move ordering:
# Null move pruning (if not in check and depth >= 3)
if not maximizing and depth >= 3 and not self._is_critical_position(board):
    # Try "passing" - what if opponent doesn't move?
    null_score, _ = self._minimax(
        board, depth - 3, alpha, beta, True,  # Reduce depth significantly
        time_left - 0.01, trapdoor_tracker
    )

    if null_score >= beta:
        # Position is so good even giving opponent a free move wins
        return beta, None  # Beta cutoff
```

**When to use:** Safe positions where opponent can't trap us immediately.

---

### 3. **Late Move Reduction (LMR)** (Medium, 15-25% speedup)

Search later moves (likely worse) at shallower depth.

**Add to `_maximize()` and `_minimize()`:**

```python
for i, move in enumerate(moves_to_search):
    # ... existing code ...

    # Reduce depth for moves after the first few
    search_depth = depth - 1
    if i >= 4 and depth >= 3:  # After first 4 moves, reduce depth
        search_depth = depth - 2  # Search 1 ply shallower

    score, _ = self._minimax(
        forecast, search_depth, alpha, beta, False,
        time_left - 0.01, trapdoor_tracker
    )
```

---

### 4. **Iterative Deepening** (Easy, better time management)

Search depth 1, then 2, then 3... until time runs out. Always have a move ready.

**Replace `search()` method:**

```python
def search(self, board, time_left, trapdoor_tracker=None, ...):
    available_time = time_left()
    start_time = time.time()
    search_time = min(available_time * self.time_limit, 2.0)

    best_move = None
    best_score = float('-inf')

    # Iterative deepening: 1, 2, 3, 4, 5...
    for depth in range(1, self.max_depth + 3):
        elapsed = time.time() - start_time
        if elapsed > search_time * 0.8:  # Stop before time limit
            break

        score, move = self._minimax(
            board, depth, float('-inf'), float('inf'), True,
            search_time - elapsed, trapdoor_tracker, ...
        )

        if move:
            best_move = move
            best_score = score
            print(f"  Depth {depth}: score={score:.1f}, move={move}")

    return best_score, best_move
```

**Benefit:** Never timeout, always have best move from previous iteration.

---

### 5. **Principal Variation Search (PVS)** (Hard, 10-20% speedup)

Advanced alpha-beta variant that searches first move with full window, rest with null window.

**Complex to implement, but very effective in endgames.**

---

### 6. **Better Evaluation Features** (Medium, accuracy improvement)

Add to `feature_extractor.py`:

**New features:**
- **Trapdoor escape routes:** How many safe squares adjacent to current position?
- **Egg race:** Who's ahead in egg-laying rate? (eggs per turn)
- **Time pressure:** Turns remaining / eggs needed
- **Mobility forecast:** How will mobility change in 2 turns?
- **Turd efficiency:** Are our turds actually blocking opponent?

---

### 7. **Monte Carlo Tree Search (MCTS)** (Alternative approach)

Instead of minimax, use random playouts:
- Good for positions with high uncertainty (trapdoors)
- Can handle longer time horizons
- More robust to evaluation errors

**Not recommended** for this game (minimax + NN is better), but worth knowing.

---

## 🎯 Prioritized Recommendations

**If you have 1 hour:**
1. ✅ Transposition Table (already done!)
2. Iterative Deepening (easy, huge benefit)

**If you have 3 hours:**
1. ✅ Transposition Table
2. Iterative Deepening
3. Null Move Pruning

**If you have a full day:**
1. ✅ Transposition Table
2. Iterative Deepening
3. Null Move Pruning
4. Late Move Reduction
5. Better evaluation features

---

## 🧪 Testing Improvements

After each improvement:

```bash
# Test locally
cd dist/engine
python3 run_local_agents.py AgentPro Yolanda

# Check:
# 1. Does it still work?
# 2. Is search faster? (check time per move)
# 3. Does it play better? (win rate)
```

---

## 📊 Expected Performance

**Current (with transposition table):**
- Effective depth: 4-6 (adaptive)
- Time per move: ~0.3s
- Search nodes: ~500-1000 (with pruning)

**With all improvements:**
- Effective depth: 5-7
- Time per move: ~0.2s
- Search nodes: ~300-600 (better pruning)
- **Strength:** Easily beat Max, compete for #1

---

## ⚠️ Warnings

**Don't implement everything at once!**
- Add one feature at a time
- Test after each change
- Some optimizations can introduce bugs

**Diminishing returns:**
- Transposition table: 2-5× speedup ✅
- Iterative deepening: Better time mgmt
- Null move: 20-30% speedup
- LMR: 15-25% speedup
- Each additional optimization gives smaller gains

---

## 🏆 Tournament Strategy

**For the tournament submission:**
1. ✅ Use improved heuristics (mobility, endgame, anti-repetition)
2. ✅ Use transposition table
3. Train v2 model on 30k depth-9 positions
4. Test thoroughly locally
5. Submit 24 hours before deadline (buffer for bugs)

**Good luck!** 🐔

