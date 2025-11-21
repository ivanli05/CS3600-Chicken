"""
Heuristics and Move Evaluation for AgentB

This module contains strategic evaluation functions for moves and positions.
"""

from typing import Tuple, List
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module


class MoveEvaluator:
    """
    Evaluates moves and positions using heuristic knowledge.
    """

    def __init__(self, map_size: int = 8):
        self.map_size = map_size

    def quick_evaluate_move(
        self,
        move: Tuple[Direction, MoveType],
        board: board_module.Board,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> float:
        """
        Quick heuristic evaluation of a move for move ordering.

        This is used to order moves before searching, improving alpha-beta pruning.
        """
        direction, move_type = move
        my_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(my_loc, direction)
        enemy_loc = board.chicken_enemy.get_location()

        score = 0.0
        
        # 0.5. MASSIVE penalty for blocked locations (enemy eggs/barriers)
        # This prevents wasting turns by repeatedly hitting the same barrier
        if blocked_locations is not None and new_loc in blocked_locations:
            score -= 50000.0  # Extremely strong penalty - never waste a turn on a known barrier
        
        # Also check if board says it's blocked (catches newly placed enemy eggs/turds)
        if board.is_cell_blocked(new_loc):
            score -= 50000.0  # Same penalty for currently blocked locations

        # 0. ENDGAME BONUS: Eggs are MORE valuable near end of game!
        # Game ends at turn 40, so maximize eggs in last 10 turns
        turns_left = getattr(board, 'turns_left_player', 40)
        endgame_multiplier = 1.0
        if turns_left <= 10:
            # Last 10 turns: eggs are 2x more valuable!
            endgame_multiplier = 2.0
        elif turns_left <= 20:
            # Last 20 turns: eggs are 1.5x more valuable
            endgame_multiplier = 1.5

        # 1. Egg moves are HIGHLY valuable (direct scoring) - EMPHASIZED!
        if move_type == MoveType.EGG:
            score += 300.0 * endgame_multiplier  # Increased in endgame!
            
            # Bonus for laying eggs in new/unexplored areas - encourages spreading eggs
            if visited_squares is None or new_loc not in visited_squares:
                score += 150.0  # INCREASED from 80.0 - stronger bonus for eggs in new areas
            
            # Bonus for laying eggs far from existing eggs (spread out, don't cluster)
            if hasattr(board, 'eggs_player') and board.eggs_player:
                min_dist_to_existing_egg = min(
                    abs(new_loc[0] - egg[0]) + abs(new_loc[1] - egg[1])
                    for egg in board.eggs_player
                )
                # Bonus for spreading eggs out (farther from existing eggs = better)
                if min_dist_to_existing_egg >= 4:
                    score += 60.0  # INCREASED from 40.0 - better spread bonus
                elif min_dist_to_existing_egg >= 3:
                    score += 30.0  # INCREASED from 20.0
                elif min_dist_to_existing_egg <= 1:
                    score -= 30.0  # Penalty for clustering too close
            
            # Corner eggs are MUCH better - they give 3 eggs instead of 1 (3x value!)
            # But only if this chicken can lay eggs on this corner (parity check)
            if self._is_corner(new_loc):
                # Check if this chicken can lay eggs on this corner
                can_lay_on_corner = board.chicken_player.can_lay_egg(new_loc)
                if can_lay_on_corner:
                    score += 500.0  # INCREASED from 200.0 - even more massive bonus for corner eggs!
                else:
                    score += 20.0  # INCREASED from 10.0
            # Center eggs control the board
            if self._is_center(new_loc):
                score += 50.0  # INCREASED from 30.0

        # 1.5. Plain moves that help exploration are valuable
        # Plain moves are necessary to reach new egg-laying locations
        # BUT: prioritize egg moves over plain moves when possible
        if move_type == MoveType.PLAIN:
            # Bonus for plain moves to new squares (helps exploration)
            if visited_squares is None or new_loc not in visited_squares:
                score += 40.0  # Good bonus for exploring via plain moves
            # Small bonus for plain moves that get us closer to unexplored areas
            # (This encourages movement even when not laying eggs)
            
            # However, plain moves should be less valuable than egg moves
            # This is already the case since egg moves get +300 base score

        # 2. Turd moves for strategic blocking
        elif move_type == MoveType.TURD:
            if board.chicken_player.get_turds_left() > 0:
                # Turds near enemy are valuable (blocking)
                dist_to_enemy = abs(new_loc[0] - enemy_loc[0]) + abs(new_loc[1] - enemy_loc[1])
                if dist_to_enemy <= 2:
                    score += 60.0
                elif dist_to_enemy <= 4:
                    score += 30.0

                # Blocking paths to valuable squares
                if self._blocks_valuable_square(new_loc, enemy_loc, board):
                    score += 40.0

        # 3. Avoid trapdoors! (CRITICAL - costs 4 eggs = 400 points!)
        # Egg moves give +300, so trapdoor penalty must be MUCH higher
        if trapdoor_tracker:
            danger = trapdoor_tracker.get_danger_score(new_loc)
            # Check if this is a known trapdoor
            is_known_trapdoor = new_loc in trapdoor_tracker.known_trapdoors
            
            if is_known_trapdoor:
                # ABSOLUTE penalty for known trapdoors - never go there!
                # This must be higher than any possible benefit (eggs, etc.)
                score -= 1000000.0
            else:
                # Heavy penalty for likely trapdoors - scale increases with probability
                # Use exponential penalty to strongly discourage even moderate probabilities
                # Even a 1% chance of losing 4 eggs is worse than most benefits
                if danger > 0.2:  # 20% or more probability
                    score -= danger * 200000.0  # Extreme penalty
                elif danger > 0.1:  # 10-20% probability
                    score -= danger * 100000.0  # Very heavy penalty
                elif danger > 0.05:  # 5-10% probability
                    score -= danger * 50000.0  # Heavy penalty
                elif danger > 0.01:  # 1-5% probability
                    score -= danger * 20000.0  # Significant penalty
                else:
                    score -= danger * 10000.0  # Still meaningful penalty
            
            # 3.5. AVOID STAYING AROUND KNOWN TRAPDOORS - explore outwards!
            # Once we know where trapdoors are, move away from them
            # Add pseudo-randomness based on location hash to break diamond patterns
            # This makes different escape directions more attractive without true randomness
            if trapdoor_tracker.known_trapdoors:
                min_dist_to_trapdoor = min(
                    abs(new_loc[0] - trap[0]) + abs(new_loc[1] - trap[1])
                    for trap in trapdoor_tracker.known_trapdoors
                )
                
                # Generate pseudo-random value from location hash (deterministic but varied)
                # This breaks diamond patterns by making different escape directions more attractive
                location_hash = hash(new_loc) % 1000
                location_random = 0.8 + (location_hash % 40) / 100.0  # 0.8 to 1.2 (20% variation)
                escape_random = 0.9 + (location_hash % 40) / 100.0  # 0.9 to 1.3 (30% variation)
                exploration_random = 1.0 + (location_hash % 50) / 100.0  # 1.0 to 1.5 (50% variation)
                
                # STRONG penalty for being near known trapdoors (within 2 squares)
                # This encourages moving away from trapdoor areas
                if min_dist_to_trapdoor <= 1:
                    score -= 2000.0  # Very strong penalty - don't stay adjacent to trapdoors!
                    # Add small pseudo-random variation to penalty to break patterns
                    penalty_variation = (location_hash % 200)
                    score -= penalty_variation * location_random
                elif min_dist_to_trapdoor <= 2:
                    score -= 800.0  # Strong penalty - avoid staying close to trapdoors
                    # Add small pseudo-random variation
                    penalty_variation = (location_hash % 150)
                    score -= penalty_variation * location_random
                elif min_dist_to_trapdoor <= 3:
                    score -= 300.0  # Moderate penalty
                    # Add small pseudo-random variation
                    penalty_variation = (location_hash % 100)
                    score -= penalty_variation * location_random
                
                # BONUS for moving far away from known trapdoors (encourages exploration)
                # Add pseudo-randomness to make different escape directions more attractive
                if min_dist_to_trapdoor >= 5:
                    base_bonus = 200.0  # Good bonus for getting far from trapdoors
                    score += base_bonus * escape_random
                elif min_dist_to_trapdoor >= 4:
                    base_bonus = 100.0  # Decent bonus
                    score += base_bonus * escape_random
                
                # EXTRA exploration bonus when moving away from trapdoors to new areas
                # Add significant pseudo-randomness to encourage varied exploration paths
                if min_dist_to_trapdoor >= 3:
                    if visited_squares is None or new_loc not in visited_squares:
                        # Strong bonus for exploring new areas away from trapdoors
                        # Add pseudo-randomness to break diamond patterns
                        score += 150.0 * exploration_random
                    else:
                        # Even if visited, still give some bonus for moving away (with variation)
                        score += 50.0 * escape_random
                
                # ADDITIONAL: Pseudo-random exploration incentive when near trapdoors
                # This helps break diamond patterns by making some directions more attractive
                if min_dist_to_trapdoor <= 3:
                    # Add a pseudo-random exploration bonus that varies by location
                    # This makes the agent try different escape paths
                    exploration_bonus = (location_hash % 200) - 50  # -50 to +150
                    if visited_squares is None or new_loc not in visited_squares:
                        # Stronger pseudo-random bonus for new areas
                        exploration_bonus = 50 + (location_hash % 150)  # 50 to 200
                    score += exploration_bonus

        # 4. Positional factors
        # Moving toward center is good (more options)
        center_dist = self._distance_to_center(new_loc)
        score -= center_dist * 2.0

        # 5. Don't move into cramped positions
        if self._is_edge(new_loc):
            score -= 10.0
        
        # 6. MASSIVE anti-repetition: heavily penalize revisiting squares, especially recent ones
        # This prevents the agent from wasting moves going back and forth
        if recent_positions is not None and len(recent_positions) > 0:
            # Check if this location was visited recently (last 8 moves)
            if new_loc in recent_positions:
                # EXTREME penalty for recently visited squares - absolutely prevents loops!
                # The more recent, the worse (check position in list)
                recent_index = recent_positions.index(new_loc)
                # More recent = higher penalty (last move = worst, 8 moves ago = less bad)
                recency_penalty = (len(recent_positions) - recent_index) * 300.0
                score -= 2000.0 + recency_penalty  # Massive base penalty + recency multiplier

                # Extra penalty if this creates a loop (going back to same square multiple times)
                visit_count = recent_positions.count(new_loc)
                if visit_count > 1:
                    score -= visit_count * 1000.0  # Each repeat visit costs 1000 points

        # 6.5. Encourage exploration - penalize revisiting ANY visited squares
        if visited_squares is not None:
            if new_loc in visited_squares:
                # Strong penalty for revisiting any visited square (even if not recent)
                score -= 800.0  # DOUBLED from 400 - very strong discouragement

                # Count how many times we've visited this square
                visit_count = sum(1 for pos in (recent_positions or []) if pos == new_loc)
                if visit_count > 0:
                    # Exponential penalty for multiple visits to same square
                    score -= visit_count * visit_count * 500.0  # 1st=500, 2nd=2000, 3rd=4500
            else:
                # HUGE bonus for exploring new squares - strongly rewards exploration
                score += 300.0  # DOUBLED from 150 - very strong reward for new areas
                
                # Extra bonus for exploring different regions of the map
                # Encourage visiting all quadrants/areas
                region_bonus = self._get_region_exploration_bonus(new_loc, visited_squares)
                score += region_bonus
                
                # Bonus for moving away from recently visited areas
                if recent_positions is not None and len(recent_positions) > 0:
                    min_dist_to_recent = min(
                        abs(new_loc[0] - pos[0]) + abs(new_loc[1] - pos[1])
                        for pos in recent_positions
                    )
                    # Bonus for being far from recently visited squares
                    if min_dist_to_recent >= 4:
                        score += 50.0  # Good distance from recent positions
                    elif min_dist_to_recent >= 3:
                        score += 25.0  # Decent distance
        
        # 7. Encourage exploration toward accessible corners (3x egg value!)
        # Only corners where this chicken can lay eggs (parity match)
        accessible_corners = self._get_accessible_corners(board)
        if accessible_corners:
            # Find closest accessible corner
            min_dist_to_corner = min(
                abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1])
                for corner in accessible_corners
            )
            # Bonus for moving toward accessible corners (especially if not visited)
            if visited_squares is None or new_loc not in visited_squares:
                # Closer to corner = better (max distance is ~14 on 8x8 board)
                corner_bonus = max(0, (14 - min_dist_to_corner) * 5.0)
                score += corner_bonus
                
                # Extra bonus if we're very close to an accessible corner
                if min_dist_to_corner <= 2:
                    score += 30.0
                elif min_dist_to_corner <= 4:
                    score += 15.0

        return score

    def evaluate_position(
        self,
        board: board_module.Board,
        nn_evaluator=None,
        feature_extractor=None,
        trapdoor_tracker=None
    ) -> float:
        """
        Comprehensive position evaluation.

        Uses neural network if available, otherwise falls back to heuristics.
        Returns a score where positive is good for current player.
        """
        # Try neural network first (if available and loaded)
        if nn_evaluator is not None and feature_extractor is not None:
            try:
                import torch

                # Extract features
                features = feature_extractor.extract_features(board, trapdoor_tracker)

                # Convert to tensor
                features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

                # Get NN prediction
                with torch.no_grad():
                    nn_output = nn_evaluator(features_tensor)
                    nn_score_raw = nn_output.item()

                # NN was trained with normalized scores (roughly [-2, +2])
                # Convert back to heuristic scale (multiply by ~1500)
                nn_score = nn_score_raw * 1500.0

                # Blend with heuristics (80% NN, 20% heuristic for stability)
                heuristic_score = self._get_heuristic_score(board)
                blended_score = 0.8 * nn_score + 0.2 * heuristic_score

                return blended_score

            except Exception as e:
                # NN failed, fall back to heuristics
                pass

        # Fallback: pure heuristic evaluation
        return self._get_heuristic_score(board)

    def _get_heuristic_score(self, board: board_module.Board) -> float:
        """Get heuristic-based score (fallback when NN not available)"""
        # Material advantage (most important)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = (my_eggs - enemy_eggs) * 300.0  # Use improved 300 per egg

        # Mobility advantage - CRITICAL for avoiding traps!
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()

        # If enemy has NO moves, they lose and we get 5 eggs (1500 points)!
        if enemy_moves == 0:
            mobility_score = 2000.0  # Winning position!
        # If WE have no moves, we lose and enemy gets 5 eggs
        elif my_moves == 0:
            mobility_score = -2000.0  # Losing position!
        else:
            # Normal mobility advantage - very important!
            # Each move difference is worth ~50 points (not 5!)
            # Having more moves = safer from traps + more options
            mobility_score = (my_moves - enemy_moves) * 50.0

            # Extra penalty for low mobility (danger of getting trapped)
            if my_moves <= 2:
                mobility_score -= 200.0  # Very dangerous!
            elif my_moves <= 3:
                mobility_score -= 100.0  # Risky

            # Bonus for reducing enemy mobility (trying to trap them)
            if enemy_moves <= 2:
                mobility_score += 200.0  # We're trapping them!
            elif enemy_moves <= 3:
                mobility_score += 100.0

        # Positional factors
        positional_score = self._evaluate_position_quality(board)

        # Combine scores
        total = egg_diff + mobility_score + positional_score

        return total

    def _evaluate_position_quality(self, board: board_module.Board) -> float:
        """Evaluate positional factors like territory control"""
        score = 0.0

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        # Center control
        my_center_dist = self._distance_to_center(my_loc)
        enemy_center_dist = self._distance_to_center(enemy_loc)
        score += (enemy_center_dist - my_center_dist) * 3.0

        # Turd advantage
        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()
        score += (my_turds - enemy_turds) * 10.0

        # Egg clusters (eggs close together are harder to block)
        my_egg_cluster = self._count_egg_clusters(board.eggs_player)
        enemy_egg_cluster = self._count_egg_clusters(board.eggs_enemy)
        score += (my_egg_cluster - enemy_egg_cluster) * 5.0

        return score

    def _blocks_valuable_square(
        self,
        turd_loc: Tuple[int, int],
        enemy_loc: Tuple[int, int],
        board: board_module.Board
    ) -> bool:
        """Check if placing a turd here blocks enemy from valuable squares"""
        # Check if turd is between enemy and valuable egg-laying squares
        valuable_squares = self._get_valuable_egg_squares(board)

        for square in valuable_squares:
            # Simple line-of-sight check
            if self._is_between(turd_loc, enemy_loc, square):
                return True

        return False

    def _get_valuable_egg_squares(self, board: board_module.Board) -> List[Tuple[int, int]]:
        """Get list of valuable egg-laying positions"""
        valuable = []

        # Center squares are valuable
        center = self.map_size // 2
        for i in range(center - 1, center + 2):
            for j in range(center - 1, center + 2):
                if 0 <= i < self.map_size and 0 <= j < self.map_size:
                    valuable.append((i, j))

        # Corners are valuable (defensible)
        corners = [
            (0, 0), (0, self.map_size - 1),
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]
        valuable.extend(corners)

        return valuable

    def _is_between(
        self,
        point: Tuple[int, int],
        start: Tuple[int, int],
        end: Tuple[int, int]
    ) -> bool:
        """Check if point is roughly between start and end"""
        px, py = point
        sx, sy = start
        ex, ey = end

        # Simple Manhattan distance check
        dist_start_to_end = abs(sx - ex) + abs(sy - ey)
        dist_start_to_point = abs(sx - px) + abs(sy - py)
        dist_point_to_end = abs(px - ex) + abs(py - ey)

        # Point is "between" if total distance is close to direct distance
        return dist_start_to_point + dist_point_to_end <= dist_start_to_end + 2

    def _count_egg_clusters(self, eggs: set) -> int:
        """Count eggs that are adjacent to other eggs (clustering bonus)"""
        cluster_count = 0
        for egg in eggs:
            # Check if this egg has neighbors
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = (egg[0] + dx, egg[1] + dy)
                    if neighbor in eggs:
                        cluster_count += 1
                        break
        return cluster_count

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        """Check if location is a corner"""
        x, y = loc
        return (x == 0 or x == self.map_size - 1) and \
               (y == 0 or y == self.map_size - 1)
    
    def _get_accessible_corners(self, board: board_module.Board) -> List[Tuple[int, int]]:
        """
        Get list of corners where this chicken can lay eggs (parity match).
        These corners give 3x egg value, so they're very valuable!
        """
        corners = [
            (0, 0), (0, self.map_size - 1),
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]
        
        # Filter to only corners this chicken can lay eggs on
        accessible = []
        for corner in corners:
            if board.chicken_player.can_lay_egg(corner):
                accessible.append(corner)
        
        return accessible

    def _is_center(self, loc: Tuple[int, int]) -> bool:
        """Check if location is in center area"""
        x, y = loc
        center = self.map_size // 2
        return abs(x - center) <= 1 and abs(y - center) <= 1

    def _is_edge(self, loc: Tuple[int, int]) -> bool:
        """Check if location is on the edge"""
        x, y = loc
        return x == 0 or x == self.map_size - 1 or \
               y == 0 or y == self.map_size - 1

    def _distance_to_center(self, loc: Tuple[int, int]) -> float:
        """Manhattan distance to center"""
        x, y = loc
        center = self.map_size / 2.0
        return abs(x - center) + abs(y - center)
    
    def _get_region_exploration_bonus(self, loc: Tuple[int, int], visited_squares: set) -> float:
        """
        Give bonus for exploring different regions of the map.
        This encourages spreading out and exploring the whole board.
        """
        if not visited_squares or len(visited_squares) < 2:
            return 0.0
        
        # Divide map into 4 quadrants
        mid = self.map_size / 2.0
        x, y = loc
        
        # Determine which quadrant this location is in
        if x < mid and y < mid:
            region = 'top_left'
        elif x < mid and y >= mid:
            region = 'top_right'
        elif x >= mid and y < mid:
            region = 'bottom_left'
        else:
            region = 'bottom_right'
        
        # Count how many visited squares are in the same region
        same_region_count = 0
        for visited in visited_squares:
            vx, vy = visited
            if vx < mid and vy < mid and region == 'top_left':
                same_region_count += 1
            elif vx < mid and vy >= mid and region == 'top_right':
                same_region_count += 1
            elif vx >= mid and vy < mid and region == 'bottom_left':
                same_region_count += 1
            elif vx >= mid and vy >= mid and region == 'bottom_right':
                same_region_count += 1
        
        # Bonus for exploring less-visited regions
        if same_region_count == 0:
            return 30.0  # First visit to this region
        elif same_region_count <= 2:
            return 15.0  # Early exploration of this region
        else:
            return 0.0  # Already well-explored region

    def find_trapping_moves(
        self,
        board: board_module.Board
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Find moves that can trap the enemy by blocking escape routes.
        """
        if board.chicken_player.get_turds_left() == 0:
            return []

        enemy_loc = board.chicken_enemy.get_location()
        my_loc = board.chicken_player.get_location()
        trapping_moves = []

        # Find turd positions that would block enemy
        for move in board.get_valid_moves():
            if move[1] != MoveType.TURD:
                continue

            direction, _ = move
            turd_loc = loc_after_direction(my_loc, direction)

            if not board.can_lay_turd_at_loc(turd_loc):
                continue

            # Count how many enemy moves this would block
            blocked_count = self._count_blocked_enemy_moves(
                turd_loc, enemy_loc, board
            )

            if blocked_count >= 2:
                trapping_moves.append(move)

        return trapping_moves

    def _count_blocked_enemy_moves(
        self,
        turd_loc: Tuple[int, int],
        enemy_loc: Tuple[int, int],
        board: board_module.Board
    ) -> int:
        """Count how many enemy moves would be blocked by a turd at turd_loc"""
        blocked = 0

        # Turds block adjacent squares
        for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
            potential_move = loc_after_direction(enemy_loc, direction)

            # Check if this move would be blocked by the turd
            # (can't move into turd or squares adjacent to turd)
            if potential_move == turd_loc:
                blocked += 1
            elif abs(potential_move[0] - turd_loc[0]) + abs(potential_move[1] - turd_loc[1]) == 1:
                blocked += 1

        return blocked

