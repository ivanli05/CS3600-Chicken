"""
Search Engine for AgentB

Implements minimax search with alpha-beta pruning, move ordering,
and other optimizations for efficient game tree exploration.
"""

from typing import Tuple, Optional, List, Callable
from game.enums import Direction, MoveType
import game.board as board_module
from .heuristics import MoveEvaluator


class SearchEngine:
    """
    Game tree search engine using minimax with alpha-beta pruning.
    """

    def __init__(
        self,
        evaluator: MoveEvaluator,
        max_depth: int = 3,
        time_limit: float = 0.8
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit

        # Optimization structures
        self.killer_moves = {}  # Killer heuristic: moves that caused cutoffs
        self.history_table = {}  # History heuristic: move success rates

    def search(
        self,
        board: board_module.Board,
        time_left: Callable,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """
        Main search entry point.

        Returns: (score, best_move)
        """
        available_time = time_left()
        search_time = min(available_time * self.time_limit, 2.0)

        # Use minimax with alpha-beta pruning
        score, best_move = self._minimax(
            board=board,
            depth=self.max_depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing=True,
            time_left=search_time,
            trapdoor_tracker=trapdoor_tracker,
            visited_squares=visited_squares,
            recent_positions=recent_positions,
            blocked_locations=blocked_locations
        )

        return score, best_move

    def _minimax(
        self,
        board: board_module.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        time_left: float,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """
        Minimax algorithm with alpha-beta pruning.

        Args:
            board: Current board state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing: True if maximizing player's turn
            time_left: Remaining search time
            trapdoor_tracker: Trapdoor probability tracker

        Returns: (score, best_move)
        """
        # Terminal conditions
        if depth == 0 or time_left < 0.01:
            score = self.evaluator.evaluate_position(board)
            return score, None

        if board.is_game_over():
            return self._evaluate_terminal(board), None

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            # No moves available - bad position
            return (-5000.0 if maximizing else 5000.0), None

        # Order moves for better pruning
        ordered_moves = self._order_moves(
            valid_moves, board, depth, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
        )

        if maximizing:
            return self._maximize(
                board, ordered_moves, depth, alpha, beta,
                time_left, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
            )
        else:
            return self._minimize(
                board, ordered_moves, depth, alpha, beta,
                time_left, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
            )

    def _maximize(
        self,
        board: board_module.Board,
        moves: List[Tuple[Direction, MoveType]],
        depth: int,
        alpha: float,
        beta: float,
        time_left: float,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """Maximizing player's turn"""
        max_score = float('-inf')
        best_move = None

        # Limit branching factor for efficiency
        moves_to_search = moves[:min(12, len(moves))]

        for move in moves_to_search:
            try:
                # Forecast this move
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                # Switch perspective and recurse
                forecast.reverse_perspective()
                score, _ = self._minimax(
                    forecast, depth - 1, alpha, beta, False,
                    time_left - 0.01, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
                )
                forecast.reverse_perspective()

                # Update best move
                if score > max_score:
                    max_score = score
                    best_move = move

                # Alpha-beta pruning
                alpha = max(alpha, score)
                if beta <= alpha:
                    # Beta cutoff - record killer move
                    self._record_killer(move, depth)
                    break

            except Exception:
                continue

        return max_score if best_move else -5000.0, best_move

    def _minimize(
        self,
        board: board_module.Board,
        moves: List[Tuple[Direction, MoveType]],
        depth: int,
        alpha: float,
        beta: float,
        time_left: float,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """Minimizing player's turn"""
        min_score = float('inf')
        best_move = None

        # Limit branching factor
        moves_to_search = moves[:min(12, len(moves))]

        for move in moves_to_search:
            try:
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                forecast.reverse_perspective()
                score, _ = self._minimax(
                    forecast, depth - 1, alpha, beta, True,
                    time_left - 0.01, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
                )
                forecast.reverse_perspective()

                if score < min_score:
                    min_score = score
                    best_move = move

                beta = min(beta, score)
                if beta <= alpha:
                    self._record_killer(move, depth)
                    break

            except Exception:
                continue

        return min_score if best_move else 5000.0, best_move

    def _order_moves(
        self,
        moves: List[Tuple[Direction, MoveType]],
        board: board_module.Board,
        depth: int,
        trapdoor_tracker=None,
        visited_squares=None,
        recent_positions=None,
        blocked_locations=None
    ) -> List[Tuple[Direction, MoveType]]:
        """
        Order moves for better alpha-beta pruning efficiency.

        Good move ordering can improve pruning by 10x or more!
        """
        move_scores = []

        for move in moves:
            score = 0.0

            # 1. Killer moves (moves that caused cutoffs at this depth)
            if depth in self.killer_moves and move in self.killer_moves[depth]:
                score += 5000.0

            # 2. History heuristic (historically good moves)
            score += self.history_table.get(move, 0) * 10.0

            # 3. Move type priority: Egg > Turd > Plain (EMPHASIZED!)
            if move[1] == MoveType.EGG:
                score += 3000.0  # INCREASED from 1000.0 - much stronger priority for egg moves!
            elif move[1] == MoveType.TURD:
                score += 500.0

            # 4. Positional evaluation
            score += self.evaluator.quick_evaluate_move(
                move, board, trapdoor_tracker, visited_squares, recent_positions, blocked_locations
            )

            move_scores.append((score, move))

        # Sort by score (highest first)
        move_scores.sort(reverse=True, key=lambda x: x[0])
        return [move for _, move in move_scores]

    def _evaluate_terminal(self, board: board_module.Board) -> float:
        """Evaluate terminal game state"""
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()

        if my_eggs > enemy_eggs:
            return 10000.0  # Win
        elif my_eggs < enemy_eggs:
            return -10000.0  # Loss
        else:
            return 0.0  # Tie

    def _record_killer(self, move: Tuple[Direction, MoveType], depth: int):
        """Record a killer move (caused alpha-beta cutoff)"""
        if depth not in self.killer_moves:
            self.killer_moves[depth] = []

        if move not in self.killer_moves[depth]:
            self.killer_moves[depth].append(move)

            # Keep only best 2 killers per depth
            if len(self.killer_moves[depth]) > 2:
                self.killer_moves[depth].pop(0)

    def record_best_move(self, move: Tuple[Direction, MoveType], depth: int):
        """Update history table with successful move"""
        if move not in self.history_table:
            self.history_table[move] = 0

        # Increase score based on depth (deeper = more important)
        self.history_table[move] += depth * depth

    def clear_search_data(self):
        """Clear search optimizations (call at start of new game)"""
        self.killer_moves.clear()
        self.history_table.clear()

