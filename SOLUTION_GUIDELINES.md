# CS3600 Chicken Tournament - Solution Guidelines

## Problem Overview

This is a competitive two-player game where chickens navigate an 8x8 board to lay eggs while avoiding hidden trapdoors. The key challenges are:

1. **Trapdoor Localization**: Inferring hidden trapdoor locations from noisy binary sensor data
2. **Strategic Decision-Making**: Maximizing eggs (win condition) while balancing exploration and safety
3. **Adversarial Play**: Competing against an intelligent opponent with limited computation time (6 minutes total)

---

## Part 1: Modeling Trapdoors via Bayesian Inference / Pose Graph Optimization

### Problem Formulation

**State Space:**
- 2 trapdoors: one on white squares (i+j even), one on black squares (i+j odd)
- Each trapdoor can be at one of 32 positions (64 squares ÷ 2 colors)
- Prior distribution: weighted toward center (see `trapdoor_manager.py` lines 77-82)

**Observation Model:**
At each position, you receive sensor data: `[(heard_white, felt_white), (heard_black, felt_black)]`

**Sensor Probabilities (from `game_map.py`):**
```python
# Adjacent (shares edge): (dx, dy) = (1, 0) or (0, 1)
P(hear | adjacent) = 0.5
P(feel | adjacent) = 0.3

# Diagonal: (dx, dy) = (1, 1)
P(hear | diagonal) = 0.25
P(feel | diagonal) = 0.15

# Two squares away (edge): (dx, dy) = (2, 0) or (0, 2) or (1, 2) or (2, 1)
P(hear | two_away) = 0.1
P(feel | two_away) = 0.0

# Farther:
P(hear | far) = 0.0
P(feel | far) = 0.0
```

### Recommended Approach: Particle Filtering

**Why Particle Filtering over Pose Graph?**
- Discrete state space (64 possible locations per trapdoor)
- Non-Gaussian posteriors (weighted prior)
- Sequential observations with binary sensors
- Computationally efficient for this problem size

**Implementation Strategy:**

```python
class TrapdoorLocalizer:
    def __init__(self, map_size=8):
        self.map_size = map_size

        # Initialize particles for each trapdoor (white and black)
        # Each particle is (location, weight)
        self.particles_white = self._init_particles(parity=0)  # even squares
        self.particles_black = self._init_particles(parity=1)  # odd squares

        # Probability distribution over all squares
        self.prob_map_white = np.zeros((map_size, map_size))
        self.prob_map_black = np.zeros((map_size, map_size))

    def _init_particles(self, parity, n_particles=1000):
        """Initialize particles according to prior distribution"""
        # Replicate the weighted prior from trapdoor_manager.py
        weights = np.zeros((self.map_size, self.map_size))
        weights[2:6, 2:6] = 1.0
        weights[3:5, 3:5] = 2.0

        # Sample from prior
        particles = []
        for _ in range(n_particles):
            loc = self._sample_location(weights, parity)
            particles.append({'loc': loc, 'weight': 1.0 / n_particles})

        return particles

    def update(self, current_pos, sensor_data):
        """Bayesian update after observing sensor data"""
        heard_white, felt_white = sensor_data[0]
        heard_black, felt_black = sensor_data[1]

        # Update white trapdoor particles
        self._update_particles(
            self.particles_white,
            current_pos,
            heard_white,
            felt_white
        )

        # Update black trapdoor particles
        self._update_particles(
            self.particles_black,
            current_pos,
            heard_black,
            felt_black
        )

        # Compute probability maps
        self._compute_prob_maps()

    def _update_particles(self, particles, obs_loc, heard, felt):
        """Update particle weights using sensor likelihood"""
        for particle in particles:
            trapdoor_loc = particle['loc']
            dx = abs(trapdoor_loc[0] - obs_loc[0])
            dy = abs(trapdoor_loc[1] - obs_loc[1])

            # Likelihood P(observation | trapdoor at this location)
            p_hear = self._prob_hear(dx, dy)
            p_feel = self._prob_feel(dx, dy)

            # Binary observation likelihood
            likelihood = 1.0
            if heard:
                likelihood *= p_hear
            else:
                likelihood *= (1 - p_hear)

            if felt:
                likelihood *= p_feel
            else:
                likelihood *= (1 - p_feel)

            particle['weight'] *= likelihood

        # Normalize weights
        total_weight = sum(p['weight'] for p in particles)
        if total_weight > 0:
            for particle in particles:
                particle['weight'] /= total_weight

        # Resample if effective sample size is too low
        self._resample_if_needed(particles)

    def _prob_hear(self, dx, dy):
        """Probability of hearing at distance (dx, dy)"""
        if dx > 2 or dy > 2 or (dx == 2 and dy == 2):
            return 0.0
        if dx == 2 or dy == 2:
            return 0.1
        if dx == 1 and dy == 1:
            return 0.25
        if dx == 1 or dy == 1:
            return 0.5
        return 0.0

    def _prob_feel(self, dx, dy):
        """Probability of feeling at distance (dx, dy)"""
        if dx > 1 or dy > 1:
            return 0.0
        if dx == 1 and dy == 1:
            return 0.15
        if dx == 1 or dy == 1:
            return 0.3
        return 0.0

    def get_danger_score(self, location):
        """Get probability that location is a trapdoor"""
        x, y = location
        parity = (x + y) % 2

        if parity == 0:
            return self.prob_map_white[x, y]
        else:
            return self.prob_map_black[x, y]

    def get_safe_locations(self, threshold=0.05):
        """Return set of locations with low trapdoor probability"""
        safe = set()
        for i in range(self.map_size):
            for j in range(self.map_size):
                if self.prob_map_white[i, j] < threshold and \
                   self.prob_map_black[i, j] < threshold:
                    safe.add((i, j))
        return safe
```

**Alternative: Grid-based Bayesian Filter**

If you prefer explicit probability grids over particles:

```python
class GridBasedLocalizer:
    def __init__(self, map_size=8):
        self.map_size = map_size

        # Initialize prior probability for each square
        # P(trapdoor at location)
        self.belief_white = self._init_prior(parity=0)
        self.belief_black = self._init_prior(parity=1)

    def _init_prior(self, parity):
        """Initialize prior according to trapdoor distribution"""
        prior = np.zeros((self.map_size, self.map_size))

        # Replicate weighted distribution
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i + j) % 2 != parity:
                    continue

                # Weight by distance from edge
                dist_from_edge = min(i, j, 7-i, 7-j)
                if dist_from_edge >= 3:
                    prior[i, j] = 2.0
                elif dist_from_edge == 2:
                    prior[i, j] = 1.0
                # Edge squares have weight 0

        # Normalize
        prior /= np.sum(prior)
        return prior

    def update(self, obs_loc, sensor_data):
        """Bayesian update: P(loc|obs) ∝ P(obs|loc) * P(loc)"""
        heard_white, felt_white = sensor_data[0]
        heard_black, felt_black = sensor_data[1]

        self._bayesian_update(self.belief_white, obs_loc, heard_white, felt_white)
        self._bayesian_update(self.belief_black, obs_loc, heard_black, felt_black)

    def _bayesian_update(self, belief, obs_loc, heard, felt):
        """Apply Bayes rule to update belief"""
        likelihood = np.ones_like(belief)

        for i in range(self.map_size):
            for j in range(self.map_size):
                if belief[i, j] == 0:
                    continue

                dx = abs(i - obs_loc[0])
                dy = abs(j - obs_loc[1])

                p_hear = self._prob_hear(dx, dy)
                p_feel = self._prob_feel(dx, dy)

                # P(observation | trapdoor at (i,j))
                likelihood[i, j] = (
                    (p_hear if heard else 1 - p_hear) *
                    (p_feel if felt else 1 - p_feel)
                )

        # Posterior = likelihood * prior
        belief *= likelihood

        # Normalize
        total = np.sum(belief)
        if total > 0:
            belief /= total
```

---

## Part 2: Neural Network Training via Self-Play (AlphaZero-style)

### Architecture Inspiration: Stockfish NNUE vs AlphaZero

**Stockfish NNUE (Efficiently Updateable NN):**
- Specialized for chess with incremental updates
- Trained on supervised learning from millions of games
- Uses efficiently updatable feature extraction

**AlphaZero (More Applicable Here):**
- Policy network π(a|s) + Value network v(s)
- Self-play training with MCTS
- No human game knowledge required

**Recommended Approach for Chicken Game:**

### 1. Network Architecture

```python
import torch
import torch.nn as nn

class ChickenPolicyValueNet(nn.Module):
    """Combined policy and value network"""

    def __init__(self, board_size=8):
        super().__init__()
        self.board_size = board_size

        # Input channels:
        # - My eggs (1)
        # - My turds (1)
        # - Enemy eggs (1)
        # - Enemy turds (1)
        # - My position (1)
        # - Enemy position (1)
        # - Trapdoor probability white (1)
        # - Trapdoor probability black (1)
        # - Legal egg squares for me (1)
        # - Legal egg squares for enemy (1)
        # Total: 10 channels

        self.conv_layers = nn.Sequential(
            # Initial convolution
            nn.Conv2d(10, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Residual blocks (AlphaZero-style)
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
        )

        # Policy head: outputs action probabilities
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4 * 3)  # 4 directions × 3 move types
        )

        # Value head: outputs win probability
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, x):
        """
        x: (batch, 10, 8, 8) board state
        Returns: (policy_logits, value)
        """
        features = self.conv_layers(x)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = torch.relu(out)
        return out
```

### 2. Self-Play Training Loop

```python
class SelfPlayTrainer:
    def __init__(self, model):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.replay_buffer = []

    def generate_self_play_games(self, n_games=100):
        """Generate training data through self-play"""
        for game_idx in range(n_games):
            game_states = []

            # Initialize game
            board = Board(...)
            trapdoor_localizer = TrapdoorLocalizer()

            while not board.is_game_over():
                # Current state
                state = self._encode_state(board, trapdoor_localizer)

                # Get move from MCTS + network
                move, mcts_policy = self._mcts_search(board, state)

                # Store (state, policy, None) - value filled later
                game_states.append((state, mcts_policy, None))

                # Execute move
                board.apply_move(move)

                # Update trapdoor beliefs
                sensor_data = board.get_sensor_data()
                trapdoor_localizer.update(board.current_location, sensor_data)

            # Game finished - backfill values
            winner = board.get_winner()
            for i, (state, policy, _) in enumerate(game_states):
                # Value from perspective of player who made this move
                player = i % 2
                value = 1.0 if winner == player else -1.0
                game_states[i] = (state, policy, value)
                self.replay_buffer.append(game_states[i])

            # Keep buffer size manageable
            if len(self.replay_buffer) > 50000:
                self.replay_buffer = self.replay_buffer[-50000:]

    def train_on_batch(self, batch_size=256):
        """Train network on batch from replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return

        # Sample batch
        batch = random.sample(self.replay_buffer, batch_size)
        states, target_policies, target_values = zip(*batch)

        states = torch.stack(states)
        target_policies = torch.stack(target_policies)
        target_values = torch.tensor(target_values).float()

        # Forward pass
        policy_logits, values = self.model(states)

        # Loss: combine policy loss + value loss
        policy_loss = F.cross_entropy(policy_logits, target_policies)
        value_loss = F.mse_loss(values.squeeze(), target_values)

        loss = policy_loss + value_loss

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _mcts_search(self, board, state, n_simulations=100):
        """
        MCTS guided by neural network
        Returns: (best_move, improved_policy)
        """
        # This is simplified - full MCTS implementation needed
        root = MCTSNode(board)

        for _ in range(n_simulations):
            # Selection: traverse tree using UCB
            node = root
            while not node.is_leaf():
                node = node.select_child()

            # Expansion: add children
            if not node.is_terminal():
                node.expand()
                node = node.select_child()

            # Evaluation: use network
            state_tensor = self._encode_state(node.board, None)
            with torch.no_grad():
                _, value = self.model(state_tensor.unsqueeze(0))

            # Backpropagation
            node.backup(value.item())

        # Return best move based on visit counts
        best_move = root.get_best_move()
        improved_policy = root.get_policy()

        return best_move, improved_policy
```

### 3. Simpler Alternative: Deep Q-Learning (DQN)

If MCTS + self-play is too complex, a simpler approach:

```python
class DQNAgent:
    def __init__(self):
        self.q_network = ChickenQNetwork()
        self.target_network = ChickenQNetwork()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters())
        self.replay_buffer = []

    def train_step(self, batch):
        states, actions, rewards, next_states, dones = batch

        # Current Q-values
        q_values = self.q_network(states).gather(1, actions)

        # Target Q-values (from target network)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            targets = rewards + 0.99 * next_q_values * (1 - dones)

        # Loss
        loss = F.mse_loss(q_values, targets.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
```

**Key Training Strategy:**
1. Start with supervised learning on simple heuristic agents
2. Gradually increase opponent strength through self-play
3. Use curriculum learning: start with no trapdoors, then add them
4. Train separate models for early-game, mid-game, end-game

---

## Part 3: Integration - Trapdoor-Aware Strategic Decision Making

### Unified Decision Framework

The key is to integrate trapdoor probability into the state representation and value function:

```python
class TrapdoorAwareAgent:
    def __init__(self):
        self.trapdoor_localizer = TrapdoorLocalizer()
        self.policy_network = ChickenPolicyValueNet()

        # Load pre-trained weights
        self.policy_network.load_state_dict(torch.load('best_model.pth'))

    def play(self, board, sensor_data, time_left):
        """Main decision function"""

        # 1. Update trapdoor beliefs
        current_pos = board.chicken_player.get_location()
        self.trapdoor_localizer.update(current_pos, sensor_data)

        # 2. Encode state (includes trapdoor probabilities)
        state = self._encode_state(board)

        # 3. Get policy from network
        with torch.no_grad():
            policy_logits, value = self.policy_network(state.unsqueeze(0))

        # 4. Mask invalid moves
        valid_moves = board.get_valid_moves()
        policy = self._mask_and_normalize(policy_logits, valid_moves)

        # 5. Apply safety filter
        safe_moves = self._filter_dangerous_moves(
            valid_moves,
            self.trapdoor_localizer,
            threshold=0.1  # Don't move if >10% trapdoor probability
        )

        # 6. Choose move
        if safe_moves:
            # Choose from safe moves using network policy
            move = self._sample_move(safe_moves, policy)
        else:
            # No safe moves - choose least dangerous
            move = self._choose_least_dangerous(valid_moves, self.trapdoor_localizer)

        return move

    def _encode_state(self, board):
        """Encode board state as tensor including trapdoor beliefs"""
        state = torch.zeros(10, 8, 8)

        # Channel 0: My eggs
        for egg in board.eggs_player:
            state[0, egg[0], egg[1]] = 1

        # Channel 1: My turds
        for turd in board.turds_player:
            state[1, turd[0], turd[1]] = 1

        # Channel 2: Enemy eggs
        for egg in board.eggs_enemy:
            state[2, egg[0], egg[1]] = 1

        # Channel 3: Enemy turds
        for turd in board.turds_enemy:
            state[3, turd[0], turd[1]] = 1

        # Channel 4: My position
        my_pos = board.chicken_player.get_location()
        state[4, my_pos[0], my_pos[1]] = 1

        # Channel 5: Enemy position
        enemy_pos = board.chicken_enemy.get_location()
        state[5, enemy_pos[0], enemy_pos[1]] = 1

        # Channel 6-7: Trapdoor probabilities (KEY INTEGRATION POINT)
        state[6] = torch.from_numpy(self.trapdoor_localizer.prob_map_white)
        state[7] = torch.from_numpy(self.trapdoor_localizer.prob_map_black)

        # Channel 8: Legal egg squares for me
        parity = board.chicken_player.even_chicken
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == parity:
                    state[8, i, j] = 1

        # Channel 9: Turns remaining (normalized)
        state[9] = board.turns_left_player / 40.0

        return state

    def _filter_dangerous_moves(self, moves, localizer, threshold=0.1):
        """Remove moves that lead to dangerous squares"""
        safe_moves = []

        for direction, move_type in moves:
            target_pos = loc_after_direction(current_pos, direction)
            danger = localizer.get_danger_score(target_pos)

            if danger < threshold:
                safe_moves.append((direction, move_type))

        return safe_moves

    def _choose_least_dangerous(self, moves, localizer):
        """If no safe moves, choose least dangerous"""
        best_move = None
        min_danger = float('inf')

        for direction, move_type in moves:
            target_pos = loc_after_direction(current_pos, direction)
            danger = localizer.get_danger_score(target_pos)

            if danger < min_danger:
                min_danger = danger
                best_move = (direction, move_type)

        return best_move
```

### Risk-Aware Value Function

The value function should trade off egg collection vs trapdoor risk:

```
V(state) = Expected_Eggs - Risk_Penalty

Where:
Expected_Eggs = Current_Eggs + Potential_Future_Eggs
Risk_Penalty = Σ P(trapdoor at next position) × Trapdoor_Cost

Trapdoor_Cost = 4 (enemy bonus) + Lost_Turns + Position_Reset_Cost
```

### Active Information Gathering

Sometimes it's worth moving to squares that don't maximize eggs but maximize information about trapdoor locations:

```python
def compute_information_gain(self, candidate_pos, localizer):
    """
    Compute expected reduction in entropy about trapdoor location
    if we move to candidate_pos
    """
    current_entropy = self._compute_entropy(localizer.belief_white)

    # Expected entropy after observing from candidate_pos
    expected_future_entropy = 0

    for heard in [True, False]:
        for felt in [True, False]:
            # Probability of this observation
            p_obs = self._prob_observation(candidate_pos, heard, felt, localizer)

            # Entropy if we observe this
            future_belief = self._simulate_update(
                localizer.belief_white.copy(),
                candidate_pos,
                heard,
                felt
            )
            future_entropy = self._compute_entropy(future_belief)

            expected_future_entropy += p_obs * future_entropy

    return current_entropy - expected_future_entropy
```

**Early Game Strategy:**
- Prioritize information gathering to localize trapdoors
- Move to positions that maximize sensor information
- Build accurate belief map before aggressive egg-laying

**Late Game Strategy:**
- Exploit known safe regions
- Aggressive egg maximization
- Use turds to block opponent from safe high-value areas

---

## Part 4: Practical Implementation Roadmap

### Phase 1: Baseline Agent (Week 1)
1. Implement simple Bayesian trapdoor localizer (grid-based)
2. Hand-crafted heuristic agent:
   - Avoid squares with >20% trapdoor probability
   - Prioritize corner eggs
   - Basic opponent blocking with turds
3. Test against reference agents (Henny, Max)

### Phase 2: Neural Network (Week 2)
1. Collect training data from self-play of baseline agents
2. Train initial policy network via supervised learning
3. Implement simple MCTS or lookahead search
4. Integrate network evaluation into move selection

### Phase 3: Self-Play Training (Week 3)
1. Implement self-play training loop
2. Iteratively improve agent through competition
3. Curriculum learning: gradually add trapdoor complexity
4. Hyperparameter tuning

### Phase 4: Integration & Optimization (Week 4)
1. Optimize trapdoor localizer (particle filter vs grid)
2. Fine-tune risk thresholds
3. Opening book for common positions
4. Time management strategies (6 min limit)
5. Extensive testing and debugging

---

## Key Implementation Tips

### Computational Efficiency
- **Pre-compute** sensor probability tables
- **Cache** board evaluations during search
- **Pruning**: only consider top-k moves from network policy
- **Time management**: allocate more time to critical decisions (early trapdoor localization, late-game egg races)

### Trapdoor Localization Tips
- **Active exploration**: early game, visit diverse positions to triangulate
- **Confident elimination**: if you visit a square and don't fall, update beliefs significantly
- **Corner strategy**: corners are safer (edges have weight 0 in prior)
- **Memory**: if you step on a trapdoor, you KNOW its location - huge information gain

### Neural Network Training Tips
- **Data augmentation**: 8-fold symmetry (rotations + reflections)
- **Curriculum learning**: train on simpler variants first
- **Reward shaping**: intermediate rewards for information gathering
- **Ensemble**: train multiple models, use voting/averaging

### Adversarial Considerations
- **Opponent modeling**: predict opponent's likely moves
- **Deception**: sometimes take risky moves to mislead opponent's trapdoor beliefs
- **Turd blocking**: block opponent's safe corridors in late game
- **Time pressure**: force opponent into time trouble with complex positions

---

## Expected Performance

**With Bayesian Localizer Only:**
- Should beat Henny (70%) easily
- Competitive with Max (90%)

**With Neural Network + MCTS:**
- Should comfortably beat Max
- Competitive for top rankings

**With Optimized Integration:**
- Strong contender for tournament victory
- Key differentiator: balance between safety and aggression

---

## Resources in Codebase

- `trapdoor_manager.py`: Prior distribution and sensor sampling
- `game_map.py`: Sensor probability functions
- `board.py`: Game state, `forecast_move()` for lookahead
- `docs/AgentA/agent.py`: Example strategic agent with basic trapdoor avoidance

Good luck with the tournament!
