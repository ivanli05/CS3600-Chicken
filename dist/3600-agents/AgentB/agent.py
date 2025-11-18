from collections.abc import Callable
from typing import List, Set, Tuple, Optional, Dict
import numpy as np
from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

"""
AgentB is an advanced strategic agent that combines:
1. Neural network for position evaluation
2. Adversarial search (minimax with alpha-beta pruning)
3. Strategic turd placement to trap/block opponents
4. Learning from game outcomes

Key strategies:
- Use minimax to look ahead and predict opponent moves
- Place turds strategically to block enemy paths to valuable positions
- Trap enemies by cutting off escape routes
- Balance egg-laying with defensive blocking
"""

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available, using simpler evaluation function")


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
else:
    class PositionEvaluator:
        """
        Simple linear evaluator when PyTorch is not available.
        """
        
        def __init__(self, input_size=64, hidden_size=128):
            # Fallback: simple weights
            self.weights = np.random.randn(input_size)
        
        def forward(self, x):
            # Simple linear evaluation
            return np.dot(x, self.weights)


class PlayerAgent:
    """
    AgentB - A strategic agent using neural networks and adversarial search
    """
    
    def __init__(self, board: board_module.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.time_left = time_left
        
        # Initialize neural network evaluator
        if TORCH_AVAILABLE:
            self.evaluator = PositionEvaluator()
            self.evaluator.eval()  # Set to evaluation mode
        else:
            self.evaluator = PositionEvaluator()
        
        # Game state tracking
        self.suspected_trapdoors: Set[Tuple[int, int]] = set()
        self.known_trapdoors: Set[Tuple[int, int]] = set()
        self.position_history: List[Tuple[int, int]] = []
        
        # Minimax parameters
        self.max_depth = 3  # Depth of search tree
        self.time_limit = 0.8  # Use 80% of available time
        
        # Learning parameters
        self.game_history: List[Dict] = []
        
    def _extract_features(self, board: board_module.Board) -> np.ndarray:
        """
        Extract features from board state for neural network evaluation.
        Returns a feature vector.
        """
        features = np.zeros(64)  # 8x8 board = 64 features
        
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        map_size = self.map_size
        
        # Position features (0-15)
        features[0] = my_loc[0] / map_size  # Normalized x
        features[1] = my_loc[1] / map_size  # Normalized y
        features[2] = enemy_loc[0] / map_size
        features[3] = enemy_loc[1] / map_size
        
        # Distance features (4-7)
        features[4] = abs(my_loc[0] - enemy_loc[0]) / map_size
        features[5] = abs(my_loc[1] - enemy_loc[1]) / map_size
        features[6] = (abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])) / (2 * map_size)
        
        # Corner proximity (8-11)
        corners = [(0, 0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
        my_min_dist = min(abs(my_loc[0] - c[0]) + abs(my_loc[1] - c[1]) for c in corners) / map_size
        enemy_min_dist = min(abs(enemy_loc[0] - c[0]) + abs(enemy_loc[1] - c[1]) for c in corners) / map_size
        features[8] = my_min_dist
        features[9] = enemy_min_dist
        
        # Egg counts (12-13)
        features[12] = board.chicken_player.get_eggs_laid() / 40.0  # Normalized
        features[13] = board.chicken_enemy.get_eggs_laid() / 40.0
        
        # Turd counts (14-15)
        features[14] = board.chicken_player.get_turds_left() / 10.0
        features[15] = board.chicken_enemy.get_turds_left() / 10.0
        
        # Board occupancy (16-31): Count eggs and turds in each quadrant
        quadrants = [
            (0, map_size//2, 0, map_size//2),
            (map_size//2, map_size, 0, map_size//2),
            (0, map_size//2, map_size//2, map_size),
            (map_size//2, map_size, map_size//2, map_size)
        ]
        for i, (x1, x2, y1, y2) in enumerate(quadrants):
            my_eggs = sum(1 for e in board.eggs_player if x1 <= e[0] < x2 and y1 <= e[1] < y2)
            enemy_eggs = sum(1 for e in board.eggs_enemy if x1 <= e[0] < x2 and y1 <= e[1] < y2)
            my_turds = sum(1 for t in board.turds_player if x1 <= t[0] < x2 and y1 <= t[1] < y2)
            enemy_turds = sum(1 for t in board.turds_enemy if x1 <= t[0] < x2 and y1 <= t[1] < y2)
            features[16 + i*4] = my_eggs / 10.0
            features[17 + i*4] = enemy_eggs / 10.0
            features[18 + i*4] = my_turds / 5.0
            features[19 + i*4] = enemy_turds / 5.0
        
        # Turn and time features (32-33)
        features[32] = board.turn_count / 80.0
        features[33] = board.turns_left_player / 40.0
        
        # Mobility features (34-35): How many valid moves available
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        features[34] = my_moves / 8.0
        features[35] = enemy_moves / 8.0
        
        # Blocking potential (36-39): Can we block enemy?
        blocking_score = self._calculate_blocking_potential(board)
        features[36] = blocking_score
        
        # Trapdoor awareness (40-41)
        features[40] = len(self.known_trapdoors) / 4.0
        features[41] = len(self.suspected_trapdoors) / 10.0
        
        # Fill remaining features with strategic indicators
        features[42:64] = self._get_strategic_features(board)
        
        return features.astype(np.float32)
    
    def _get_strategic_features(self, board: board_module.Board) -> np.ndarray:
        """Get additional strategic features"""
        features = np.zeros(22)
        
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        
        # Can we lay egg? (0)
        egg_moves = [m for m in board.get_valid_moves() if m[1] == MoveType.EGG]
        features[0] = len(egg_moves) / 4.0
        
        # Can we place turd? (1)
        turd_moves = [m for m in board.get_valid_moves() if m[1] == MoveType.TURD]
        features[1] = len(turd_moves) / 4.0
        
        # Are we in corner? (2)
        features[2] = 1.0 if self._is_corner(my_loc) else 0.0
        features[3] = 1.0 if self._is_corner(enemy_loc) else 0.0
        
        # Egg difference (4)
        features[4] = (board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid()) / 20.0
        
        # Can enemy reach corner? (5-8)
        corners = [(0, 0), (0, self.map_size-1), (self.map_size-1, 0), (self.map_size-1, self.map_size-1)]
        for i, corner in enumerate(corners):
            dist = abs(enemy_loc[0] - corner[0]) + abs(enemy_loc[1] - corner[1])
            features[5 + i] = 1.0 / (dist + 1) if dist <= 3 else 0.0
        
        return features
    
    def _calculate_blocking_potential(self, board: board_module.Board) -> float:
        """Calculate how well we can block the enemy"""
        if board.chicken_player.get_turds_left() == 0:
            return 0.0
        
        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()
        
        # Check if we can place turd near enemy
        blocking_score = 0.0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if abs(dx) + abs(dy) > 2:
                    continue
                test_loc = (enemy_loc[0] + dx, enemy_loc[1] + dy)
                if board.is_valid_cell(test_loc) and board.can_lay_turd_at_loc(test_loc):
                    # This turd would block enemy movement
                    blocking_score += 0.2
        
        return min(blocking_score, 1.0)
    
    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is a corner"""
        x, y = loc
        return (x == 0 or x == self.map_size - 1) and (y == 0 or y == self.map_size - 1)
    
    def _evaluate_position(self, board: board_module.Board) -> float:
        """
        Evaluate a board position using neural network or heuristic.
        Returns a score where positive is good for us.
        """
        # Extract features
        features = self._extract_features(board)
        
        # Use neural network if available
        if TORCH_AVAILABLE:
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                score = self.evaluator(features_tensor).item()
        else:
            score = self.evaluator.forward(features)
            if isinstance(score, np.ndarray):
                score = float(score)
        
        # Add heuristic bonuses
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = (my_eggs - enemy_eggs) * 10.0
        
        # Corner eggs are valuable
        my_loc = board.chicken_player.get_location()
        if my_loc in board.eggs_player and self._is_corner(my_loc):
            egg_diff += 20.0
        
        # Mobility advantage
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        mobility_diff = (my_moves - enemy_moves) * 2.0
        
        return score * 50.0 + egg_diff + mobility_diff
    
    def _minimax(
        self,
        board: board_module.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        time_left: float
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """
        Minimax algorithm with alpha-beta pruning.
        Returns (score, best_move)
        """
        # Terminal conditions
        if depth == 0 or time_left < 0.01:
            score = self._evaluate_position(board)
            return score, None
        
        if board.is_game_over():
            my_eggs = board.chicken_player.get_eggs_laid()
            enemy_eggs = board.chicken_enemy.get_eggs_laid()
            if my_eggs > enemy_eggs:
                return 10000.0, None
            elif my_eggs < enemy_eggs:
                return -10000.0, None
            else:
                return 0.0, None
        
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            # No moves available - bad position
            return -5000.0 if maximizing else 5000.0, None
        
        if maximizing:
            max_score = float('-inf')
            best_move = None
            
            # Sort moves by heuristic to improve alpha-beta pruning
            move_scores = [(self._quick_evaluate_move(m, board), m) for m in valid_moves]
            move_scores.sort(reverse=True, key=lambda x: x[0])
            
            for _, move in move_scores[:min(10, len(move_scores))]:  # Limit branching
                try:
                    forecast = board.forecast_move(move[0], move[1], check_ok=False)
                    if forecast is None:
                        continue
                    
                    forecast.reverse_perspective()
                    score, _ = self._minimax(forecast, depth - 1, alpha, beta, False, time_left - 0.01)
                    forecast.reverse_perspective()
                    
                    if score > max_score:
                        max_score = score
                        best_move = move
                    
                    alpha = max(alpha, score)
                    if beta <= alpha:
                        break  # Alpha-beta pruning
                except:
                    continue
            
            return max_score if best_move else -5000.0, best_move
        else:
            min_score = float('inf')
            best_move = None
            
            move_scores = [(self._quick_evaluate_move(m, board), m) for m in valid_moves]
            move_scores.sort(key=lambda x: x[0])  # Enemy wants to minimize
            
            for _, move in move_scores[:min(10, len(move_scores))]:
                try:
                    forecast = board.forecast_move(move[0], move[1], check_ok=False)
                    if forecast is None:
                        continue
                    
                    forecast.reverse_perspective()
                    score, _ = self._minimax(forecast, depth - 1, alpha, beta, True, time_left - 0.01)
                    forecast.reverse_perspective()
                    
                    if score < min_score:
                        min_score = score
                        best_move = move
                    
                    beta = min(beta, score)
                    if beta <= alpha:
                        break
                except:
                    continue
            
            return min_score if best_move else 5000.0, best_move
    
    def _quick_evaluate_move(self, move: Tuple[Direction, MoveType], board: board_module.Board) -> float:
        """Quick heuristic evaluation of a move for move ordering"""
        dir, move_type = move
        my_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(my_loc, dir)
        enemy_loc = board.chicken_enemy.get_location()
        
        score = 0.0
        
        # Eggs are valuable
        if move_type == MoveType.EGG:
            score += 100.0
            if self._is_corner(new_loc):
                score += 150.0
        
        # Turds for blocking
        elif move_type == MoveType.TURD:
            if board.chicken_player.get_turds_left() > 0:
                # Block enemy paths
                dist_to_enemy = abs(new_loc[0] - enemy_loc[0]) + abs(new_loc[1] - enemy_loc[1])
                if dist_to_enemy <= 3:
                    score += 50.0
                    # Block paths to corners
                    corners = [(0, 0), (0, self.map_size-1), (self.map_size-1, 0), (self.map_size-1, self.map_size-1)]
                    for corner in corners:
                        enemy_dist = abs(enemy_loc[0] - corner[0]) + abs(enemy_loc[1] - corner[1])
                        if enemy_dist <= 3:
                            score += 30.0
        
        # Avoid suspected trapdoors
        if new_loc in self.suspected_trapdoors or new_loc in self.known_trapdoors:
            score -= 1000.0
        
        return score
    
    def _update_trapdoor_beliefs(
        self,
        current_loc: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]]
    ):
        """Update beliefs about trapdoor locations"""
        for trapdoor_idx, (heard, felt) in enumerate(sensor_data):
            if heard or felt:
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if abs(dx) > 2 or abs(dy) > 2:
                            continue
                        if abs(dx) == 2 and abs(dy) == 2:
                            continue
                        
                        candidate = (current_loc[0] + dx, current_loc[1] + dy)
                        if (0 <= candidate[0] < self.map_size and 
                            0 <= candidate[1] < self.map_size):
                            if felt:
                                if abs(dx) <= 1 and abs(dy) <= 1:
                                    self.suspected_trapdoors.add(candidate)
                            elif heard:
                                if abs(dx) <= 2 and abs(dy) <= 2:
                                    if not (abs(dx) == 2 and abs(dy) == 2):
                                        self.suspected_trapdoors.add(candidate)
    
    def _find_trapping_moves(self, board: board_module.Board) -> List[Tuple[Direction, MoveType]]:
        """
        Find moves that can trap the enemy by placing turds strategically.
        """
        if board.chicken_player.get_turds_left() == 0:
            return []
        
        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()
        trapping_moves = []
        
        # Find positions where turds would block enemy escape routes
        for move in board.get_valid_moves():
            if move[1] != MoveType.TURD:
                continue
            
            dir, _ = move
            turd_loc = loc_after_direction(my_loc, dir)
            
            # Check if this turd would block enemy's path to valuable positions
            if board.can_lay_turd_at_loc(turd_loc):
                # Count how many enemy moves this would block
                board.reverse_perspective()
                enemy_moves_before = len(board.get_valid_moves())
                
                # Simulate placing turd
                test_turds = set(board.turds_player)
                test_turds.add(turd_loc)
                
                # Count blocked moves
                blocked_count = 0
                for enemy_move in board.get_valid_moves():
                    enemy_dir, _ = enemy_move
                    enemy_new_loc = loc_after_direction(enemy_loc, enemy_dir)
                    # Check if turd blocks this move
                    if (abs(enemy_new_loc[0] - turd_loc[0]) <= 1 and 
                        abs(enemy_new_loc[1] - turd_loc[1]) <= 1):
                        blocked_count += 1
                
                board.reverse_perspective()
                
                if blocked_count >= 2:  # Blocks at least 2 enemy moves
                    trapping_moves.append(move)
        
        return trapping_moves
    
    def play(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Main play method using neural network evaluation and minimax search.
        """
        location = board.chicken_player.get_location()
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()
        turns_left = board.turns_left_player
        turn_count = board.turn_count
        
        print(f"\n{'='*60}")
        print(f"AgentB - Turn {turn_count}")
        print(f"{'='*60}")
        print(f"Position: {location}")
        print(f"Eggs: AgentB={my_eggs} | Enemy={enemy_eggs} | Diff={my_eggs - enemy_eggs:+d}")
        print(f"Turds: AgentB={my_turds} | Enemy={enemy_turds}")
        print(f"Turns remaining: {turns_left}")
        print(f"Time left: {time_left():.2f}s")
        print(f"Trapdoor sensors: A[heard={sensor_data[0][0]}, felt={sensor_data[0][1]}] | B[heard={sensor_data[1][0]}, felt={sensor_data[1][1]}]")
        
        # Check if game is over
        if board.is_game_over():
            winner = board.get_winner()
            win_reason = board.get_win_reason() if hasattr(board, 'win_reason') else None
            
            print(f"\n{'='*60}")
            print(f"GAME OVER!")
            print(f"{'='*60}")
            if winner is not None:
                if hasattr(winner, 'name'):
                    winner_name = winner.name
                else:
                    winner_name = str(winner)
                
                if win_reason is not None:
                    if hasattr(win_reason, 'name'):
                        reason = win_reason.name
                    else:
                        reason = str(win_reason)
                    print(f"Winner: {winner_name} by {reason}")
                else:
                    print(f"Winner: {winner_name}")
                
                # Determine who actually won based on egg count
                if my_eggs > enemy_eggs:
                    print(f"AgentB WINS! ({my_eggs} eggs vs {enemy_eggs} eggs)")
                elif enemy_eggs > my_eggs:
                    print(f"Enemy WINS! ({enemy_eggs} eggs vs {my_eggs} eggs)")
                else:
                    print(f"TIE! (Both have {my_eggs} eggs)")
            else:
                # Fallback: determine winner by eggs
                if my_eggs > enemy_eggs:
                    print(f"AgentB WINS! ({my_eggs} eggs vs {enemy_eggs} eggs)")
                elif enemy_eggs > my_eggs:
                    print(f"Enemy WINS! ({enemy_eggs} eggs vs {my_eggs} eggs)")
                else:
                    print(f"TIE! (Both have {my_eggs} eggs)")
            print(f"{'='*60}\n")
            return (Direction.UP, MoveType.PLAIN)
        
        # Update trapdoor beliefs
        self._update_trapdoor_beliefs(location, sensor_data)
        if hasattr(board, 'found_trapdoors'):
            self.known_trapdoors.update(board.found_trapdoors)
            if board.found_trapdoors:
                print(f"Known trapdoors: {board.found_trapdoors}")
        
        # Track position
        self.position_history.append(location)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("WARNING: No valid moves available!")
            return (Direction.UP, MoveType.PLAIN)
        
        print(f"Valid moves: {len(valid_moves)}")
        
        # Calculate available time
        available_time = time_left()
        search_time = min(available_time * self.time_limit, 2.0)  # Cap at 2 seconds
        print(f"Using {search_time:.2f}s for search (depth={self.max_depth})")
        
        # Look for trapping moves first
        trapping_moves = self._find_trapping_moves(board)
        if trapping_moves:
            print(f"[TRAP] Found {len(trapping_moves)} trapping moves!")
            # Evaluate trapping moves with minimax
            best_trap_move = None
            best_trap_score = float('-inf')
            
            for move in trapping_moves[:3]:  # Limit to top 3
                try:
                    forecast = board.forecast_move(move[0], move[1], check_ok=False)
                    if forecast is None:
                        continue
                    forecast.reverse_perspective()
                    score, _ = self._minimax(forecast, 2, float('-inf'), float('inf'), False, search_time / 3)
                    forecast.reverse_perspective()
                    if score > best_trap_score:
                        best_trap_score = score
                        best_trap_move = move
                except:
                    continue
            
            if best_trap_move:
                dir, move_type = best_trap_move
                new_loc = loc_after_direction(location, dir)
                print(f"[TRAP] Playing TRAPPING move: {Direction(dir).name} + {MoveType(move_type).name} -> {new_loc}")
                print(f"   Score: {best_trap_score:.2f}")
                return best_trap_move
        
        # Use minimax to find best move
        try:
            print("[SEARCH] Running minimax search...")
            score, best_move = self._minimax(
                board,
                self.max_depth,
                float('-inf'),
                float('inf'),
                True,
                search_time
            )
            
            if best_move:
                dir, move_type = best_move
                new_loc = loc_after_direction(location, dir)
                print(f"[MOVE] Minimax chose: {Direction(dir).name} + {MoveType(move_type).name} -> {new_loc}")
                print(f"   Evaluation score: {score:.2f}")
                
                # Show move type breakdown
                egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
                turd_moves = [m for m in valid_moves if m[1] == MoveType.TURD]
                plain_moves = [m for m in valid_moves if m[1] == MoveType.PLAIN]
                print(f"   Move options: {len(egg_moves)} eggs, {len(turd_moves)} turds, {len(plain_moves)} plain")
                
                return best_move
        except Exception as e:
            print(f"[ERROR] Minimax error: {e}")
        
        # Fallback: use heuristic evaluation
        print("[FALLBACK] Using heuristic fallback...")
        move_scores = [(self._quick_evaluate_move(m, board), m) for m in valid_moves]
        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        best_move = move_scores[0][1]
        dir, move_type = best_move
        new_loc = loc_after_direction(location, dir)
        print(f"[MOVE] Heuristic chose: {Direction(dir).name} + {MoveType(move_type).name} -> {new_loc}")
        print(f"   Score: {move_scores[0][0]:.2f}")
        print(f"{'='*60}\n")
        return best_move

