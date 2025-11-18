"""
AgentB - Advanced Strategic Chicken Agent

A competitive agent that combines:
- Bayesian trapdoor inference
- Minimax search with alpha-beta pruning
- Strategic move evaluation
- Optional neural network evaluation

Author: AgentB Team
"""

from collections.abc import Callable
from typing import List, Tuple, Optional
import numpy as np

from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

# Import our modules
from .evaluator import PositionEvaluator, TORCH_AVAILABLE
from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .heuristics import MoveEvaluator


class PlayerAgent:
    """
    AgentB - A strategic agent using search algorithms and probabilistic inference.
    """

    def __init__(self, board: board_module.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.time_left = time_left

        # Initialize components
        self.trapdoor_tracker = TrapdoorTracker(map_size=self.map_size)
        self.move_evaluator = MoveEvaluator(map_size=self.map_size)
        self.search_engine = SearchEngine(
            evaluator=self.move_evaluator,
            max_depth=4,  # Search depth
            time_limit=0.7  # Use 70% of available time per move
        )

        # Neural network evaluator (optional, currently not used)
        if TORCH_AVAILABLE:
            self.nn_evaluator = PositionEvaluator()
            self.nn_evaluator.load_weights()  # Try to load trained weights
            self.nn_evaluator.eval()
        else:
            self.nn_evaluator = None

        # Game state tracking
        self.position_history: List[Tuple[int, int]] = []
        self.turn_count = 0

        print(f"✓ AgentB initialized (search_depth={self.search_engine.max_depth})")

    def play(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """
        Main play method - called each turn to choose a move.

        Args:
            board: Current board state
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
            time_left: Function returning remaining time in seconds

        Returns: (direction, move_type) tuple
        """
        self.turn_count += 1
        location = board.chicken_player.get_location()

        # Print turn information
        self._print_turn_info(board, sensor_data, time_left)

        # Handle game over
        if board.is_game_over():
            self._print_game_over(board)
            return (Direction.UP, MoveType.PLAIN)

        # Update trapdoor beliefs based on sensor data
        self.trapdoor_tracker.update_beliefs(location, sensor_data)

        # Track position history
        self.position_history.append(location)
        if len(self.position_history) > 20:
            self.position_history.pop(0)

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("⚠ WARNING: No valid moves available!")
            return (Direction.UP, MoveType.PLAIN)

        print(f"Valid moves: {len(valid_moves)}")

        # Strategy 1: Look for high-value trapping moves
        trapping_move = self._evaluate_trapping_moves(board)
        if trapping_move:
            return trapping_move

        # Strategy 2: Use minimax search to find best move
        best_move = self._search_best_move(board, time_left)
        if best_move:
            return best_move

        # Fallback: Use heuristic evaluation only
        print("[FALLBACK] Using heuristic evaluation...")
        return self._fallback_move(board, valid_moves)

    def _evaluate_trapping_moves(
        self,
        board: board_module.Board
    ) -> Optional[Tuple[Direction, MoveType]]:
        """
        Evaluate moves that could trap the opponent.
        Returns best trapping move if found, None otherwise.
        """
        trapping_moves = self.move_evaluator.find_trapping_moves(board)

        if not trapping_moves:
            return None

        print(f"[TRAP] Found {len(trapping_moves)} potential trapping moves")

        # Evaluate trapping moves with shallow search
        best_trap_move = None
        best_trap_score = float('-inf')

        for move in trapping_moves[:3]:  # Evaluate top 3
            try:
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                # Quick evaluation
                forecast.reverse_perspective()
                score = self.move_evaluator.evaluate_position(forecast)
                forecast.reverse_perspective()

                if score > best_trap_score:
                    best_trap_score = score
                    best_trap_move = move

            except Exception:
                continue

        if best_trap_move:
            direction, move_type = best_trap_move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
            print(f"[TRAP] Selected: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
            print(f"       Score: {best_trap_score:.1f}")
            return best_trap_move

        return None

    def _search_best_move(
        self,
        board: board_module.Board,
        time_left: Callable
    ) -> Optional[Tuple[Direction, MoveType]]:
        """
        Use minimax search to find the best move.
        """
        try:
            print(f"[SEARCH] Running minimax (depth={self.search_engine.max_depth})...")

            score, best_move = self.search_engine.search(
                board=board,
                time_left=time_left,
                trapdoor_tracker=self.trapdoor_tracker
            )

            if best_move:
                direction, move_type = best_move
                new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

                print(f"[MOVE] Minimax chose: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
                print(f"       Evaluation: {score:.1f}")

                # Update history for move ordering
                self.search_engine.record_best_move(best_move, self.search_engine.max_depth)

                return best_move

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")

        return None

    def _fallback_move(
        self,
        board: board_module.Board,
        valid_moves: List[Tuple[Direction, MoveType]]
    ) -> Tuple[Direction, MoveType]:
        """
        Fallback heuristic evaluation when search fails.
        """
        move_scores = [
            (
                self.move_evaluator.quick_evaluate_move(
                    m, board, self.trapdoor_tracker
                ),
                m
            )
            for m in valid_moves
        ]

        move_scores.sort(reverse=True, key=lambda x: x[0])
        best_score, best_move = move_scores[0]

        direction, move_type = best_move
        new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

        print(f"[MOVE] Heuristic chose: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
        print(f"       Score: {best_score:.1f}")

        return best_move

    def _print_turn_info(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable
    ):
        """Print information about current turn"""
        location = board.chicken_player.get_location()
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        my_turds = board.chicken_player.get_turds_left()
        turns_left = board.turns_left_player

        print(f"\n{'=' * 60}")
        print(f"AgentB - Turn {self.turn_count}")
        print(f"{'=' * 60}")
        print(f"Position: {location}")
        print(f"Eggs: Me={my_eggs} | Enemy={enemy_eggs} | Diff={my_eggs - enemy_eggs:+d}")
        print(f"Turds left: {my_turds}")
        print(f"Turns remaining: {turns_left}")
        print(f"Time left: {time_left():.2f}s")

        # Sensor information
        heard_w, felt_w = sensor_data[0]
        heard_b, felt_b = sensor_data[1]
        print(f"Sensors: White[H={heard_w}, F={felt_w}] | Black[H={heard_b}, F={felt_b}]")

        # Trapdoor beliefs
        likely_traps = self.trapdoor_tracker.get_most_likely_trapdoors(3)
        if likely_traps:
            print("Most likely trapdoors:")
            for (x, y), prob in likely_traps:
                color = "white" if (x + y) % 2 == 0 else "black"
                print(f"  ({x},{y}) [{color}]: {prob:.1%}")

    def _print_game_over(self, board: board_module.Board):
        """Print game over information"""
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()

        print(f"\n{'=' * 60}")
        print(f"GAME OVER!")
        print(f"{'=' * 60}")

        if my_eggs > enemy_eggs:
            print(f"✓ AgentB WINS! ({my_eggs} eggs vs {enemy_eggs} eggs)")
        elif enemy_eggs > my_eggs:
            print(f"✗ Enemy wins ({enemy_eggs} eggs vs {my_eggs} eggs)")
        else:
            print(f"⚖ TIE! (Both have {my_eggs} eggs)")

        print(f"{'=' * 60}\n")
