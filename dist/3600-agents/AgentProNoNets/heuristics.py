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
        recent_positions=None
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

        # 1. Egg moves are highly valuable (direct scoring)
        if move_type == MoveType.EGG:
            score += 100.0
            
            # Bonus for laying eggs in new/unexplored areas - encourages spreading eggs
            if visited_squares is None or new_loc not in visited_squares:
                score += 80.0  # Large bonus for eggs in new areas
            
            # Bonus for laying eggs far from existing eggs (spread out, don't cluster)
            if hasattr(board, 'eggs_player') and board.eggs_player:
                min_dist_to_existing_egg = min(
                    abs(new_loc[0] - egg[0]) + abs(new_loc[1] - egg[1])
                    for egg in board.eggs_player
                )
                # Bonus for spreading eggs out (farther from existing eggs = better)
                if min_dist_to_existing_egg >= 4:
                    score += 40.0  # Good spread
                elif min_dist_to_existing_egg >= 3:
                    score += 20.0  # Decent spread
                elif min_dist_to_existing_egg <= 1:
                    score -= 30.0  # Penalty for clustering too close
            
            # Corner eggs are MUCH better - they give 3 eggs instead of 1 (3x value!)
            # But only if this chicken can lay eggs on this corner (parity check)
            if self._is_corner(new_loc):
                # Check if this chicken can lay eggs on this corner
                can_lay_on_corner = board.chicken_player.can_lay_egg(new_loc)
                if can_lay_on_corner:
                    score += 200.0  # Massive bonus for accessible corner eggs (3x value!)
                else:
                    score += 10.0  # Small bonus even if can't lay (still valuable position)
            # Center eggs control the board
            if self._is_center(new_loc):
                score += 30.0

        # 1.5. Plain moves that help exploration are valuable
        # Plain moves are necessary to reach new egg-laying locations
        if move_type == MoveType.PLAIN:
            # Bonus for plain moves to new squares (helps exploration)
            if visited_squares is None or new_loc not in visited_squares:
                score += 40.0  # Good bonus for exploring via plain moves
            # Small bonus for plain moves that get us closer to unexplored areas
            # (This encourages movement even when not laying eggs)

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
        # Egg moves give +100, so trapdoor penalty must be MUCH higher
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

        # 4. Positional factors
        # Moving toward center is good (more options)
        center_dist = self._distance_to_center(new_loc)
        score -= center_dist * 2.0

        # 5. Don't move into cramped positions
        if self._is_edge(new_loc):
            score -= 10.0
        
        # 6. STRONG anti-repetition: penalize revisiting squares, especially recent ones
        # This prevents the agent from repeating the same few squares
        if recent_positions is not None and len(recent_positions) > 0:
            # Check if this location was visited recently (last 8 moves)
            if new_loc in recent_positions:
                # VERY STRONG penalty for recently visited squares - prevents repetition!
                # The more recent, the worse (check position in list)
                recent_index = recent_positions.index(new_loc)
                # More recent = higher penalty (last move = worst, 8 moves ago = less bad)
                recency_penalty = (len(recent_positions) - recent_index) * 150.0
                score -= 500.0 + recency_penalty  # Base 500 + recency multiplier
        
        # 6.5. Encourage exploration - penalize revisiting ANY visited squares
        if visited_squares is not None:
            if new_loc in visited_squares:
                # Penalty for revisiting any visited square (even if not recent)
                score -= 400.0  # Increased from 200 to strongly discourage repetition
            else:
                # LARGE bonus for exploring new squares - encourages map exploration and egg placement
                score += 150.0  # Increased from 120 to strongly reward exploration
                
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
        nn_evaluator=None
    ) -> float:
        """
        Comprehensive position evaluation.

        Returns a score where positive is good for current player.
        """
        # Material advantage (most important)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = (my_eggs - enemy_eggs) * 100.0

        # Mobility advantage
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        mobility_diff = (my_moves - enemy_moves) * 5.0

        # Positional factors
        positional_score = self._evaluate_position_quality(board)

        # Neural network evaluation (if available and trained)
        # Note: nn_evaluator is kept for compatibility but not used (no torch)
        nn_score = 0.0
        if nn_evaluator is not None:
            try:
                # This would use the neural network
                # For now, placeholder (no torch implementation)
                nn_score = 0.0
            except:
                pass

        # Combine scores
        total = egg_diff + mobility_diff + positional_score + nn_score

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

