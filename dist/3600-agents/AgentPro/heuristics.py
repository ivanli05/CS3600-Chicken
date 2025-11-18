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
        trapdoor_tracker=None
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
            # Corner eggs are even better (harder to block)
            if self._is_corner(new_loc):
                score += 50.0
            # Center eggs control the board
            if self._is_center(new_loc):
                score += 30.0

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

        # 3. Avoid trapdoors!
        if trapdoor_tracker:
            danger = trapdoor_tracker.get_danger_score(new_loc)
            # Heavy penalty for likely trapdoors (costs 4 eggs!)
            score -= danger * 1000.0

        # 4. Positional factors
        # Moving toward center is good (more options)
        center_dist = self._distance_to_center(new_loc)
        score -= center_dist * 2.0

        # 5. Don't move into cramped positions
        if self._is_edge(new_loc):
            score -= 10.0

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
        nn_score = 0.0
        if nn_evaluator is not None:
            try:
                # This would use the neural network
                # For now, placeholder
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
