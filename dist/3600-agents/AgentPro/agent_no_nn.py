"""
AgentPro - No Neural Network Version (Pure Heuristics + Search)

This version completely disables the neural network to see pure search behavior.
Use this to test and tune your heuristics before training a new model.
"""

from collections.abc import Callable
from typing import List, Tuple, Optional

from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

# Import our modules
from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .heuristics import MoveEvaluator


class PlayerAgent:
    """
    AgentPro - Pure search version (NO neural network)
    """

    def __init__(self, board: board_module.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.time_left = time_left

        # Initialize components
        self.trapdoor_tracker = TrapdoorTracker(map_size=self.map_size)
        self.move_evaluator = MoveEvaluator(map_size=self.map_size)

        # NO neural network!
        self.nn_evaluator = None
        self.use_nn_eval = False

        self.search_engine = SearchEngine(
            evaluator=self.move_evaluator,
            max_depth=4,  # Search depth (can increase if you want deeper search)
            time_limit=0.7  # Use 70% of available time per move
        )

        # Game state tracking
        self.position_history: List[Tuple[int, int]] = []
        self.recent_positions: List[Tuple[int, int]] = []
        self.visited_squares: set = set()
        self.last_location: Optional[Tuple[int, int]] = None
        self.last_move_target: Optional[Tuple[int, int]] = None
        self.blocked_locations: set = set()
        self.turn_count = 0

    def play(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """Main play method - called each turn to choose a move."""
        self.turn_count += 1
        location = board.chicken_player.get_location()

        # Detect trapdoors
        if self.last_move_target is not None:
            spawn_location = board.chicken_player.get_spawn()
            if location == spawn_location and location != self.last_location:
                self.trapdoor_tracker.mark_trapdoor_found(self.last_move_target)
                print(f"🚨 TRAPDOOR at {self.last_move_target}")
                self.visited_squares.add(self.last_move_target)
            elif location != self.last_move_target and location == self.last_location:
                if self.last_move_target not in self.trapdoor_tracker.known_trapdoors:
                    self.blocked_locations.add(self.last_move_target)
                    print(f"🚫 BLOCKED at {self.last_move_target}")

        # Update from board state
        if hasattr(board, 'found_trapdoors'):
            for trapdoor_loc in board.found_trapdoors:
                self.trapdoor_tracker.mark_trapdoor_found(trapdoor_loc)
                self.visited_squares.add(trapdoor_loc)

        if hasattr(board, 'eggs_enemy'):
            for egg_loc in board.eggs_enemy:
                self.blocked_locations.add(egg_loc)

        if hasattr(board, 'turds_enemy'):
            for turd_loc in board.turds_enemy:
                self.blocked_locations.add(turd_loc)
                for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                    adjacent = loc_after_direction(turd_loc, direction)
                    if board.is_valid_cell(adjacent):
                        self.blocked_locations.add(adjacent)

        self.visited_squares.add(location)
        self.last_location = location

        # Print turn info
        self._print_turn_info(board, sensor_data, time_left)

        if board.is_game_over():
            self._print_game_over(board)
            return (Direction.UP, MoveType.PLAIN)

        # Update trapdoor beliefs
        self.trapdoor_tracker.update_beliefs(location, sensor_data)

        danger = self.trapdoor_tracker.get_danger_score(location)
        if danger > 0.8:
            self.trapdoor_tracker.mark_trapdoor_found(location)
            print(f"⚠ Marked trapdoor at {location} (prob: {danger:.1%})")

        # Track positions
        self.position_history.append(location)
        if len(self.position_history) > 20:
            self.position_history.pop(0)

        self.recent_positions.append(location)
        if len(self.recent_positions) > 8:
            self.recent_positions.pop(0)

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("⚠ No valid moves!")
            return (Direction.UP, MoveType.PLAIN)

        # Filter out dangerous moves
        safe_moves = []
        for move in valid_moves:
            direction, move_type = move
            target_loc = loc_after_direction(location, direction)

            if target_loc in self.trapdoor_tracker.known_trapdoors:
                print(f"⚠ Filtered known trapdoor at {target_loc}")
                continue

            if target_loc in self.blocked_locations:
                print(f"⚠ Filtered blocked location at {target_loc}")
                continue

            if board.is_cell_blocked(target_loc):
                self.blocked_locations.add(target_loc)
                print(f"⚠ Filtered blocked (board) at {target_loc}")
                continue

            danger = self.trapdoor_tracker.get_danger_score(target_loc)
            if danger > 0.3:
                print(f"⚠ Filtered high-risk at {target_loc} (danger: {danger:.1%})")
                continue

            safe_moves.append(move)

        if not safe_moves:
            print("⚠ All moves filtered! Using originals with penalties...")
            safe_moves = valid_moves
        else:
            valid_moves = safe_moves
            print(f"Safe moves: {len(valid_moves)}")

        # Strategy 1: Trapping moves
        trapping_move = self._evaluate_trapping_moves(board)
        if trapping_move:
            return trapping_move

        # Strategy 2: Minimax search (PURE HEURISTICS - NO NN!)
        best_move = self._search_best_move(board, time_left)
        if best_move:
            return best_move

        # Fallback
        print("[FALLBACK] Using heuristic evaluation...")
        return self._fallback_move(board, valid_moves)

    def _evaluate_trapping_moves(self, board: board_module.Board) -> Optional[Tuple[Direction, MoveType]]:
        """Evaluate moves that could trap the opponent."""
        trapping_moves = self.move_evaluator.find_trapping_moves(board)

        if not trapping_moves:
            return None

        print(f"[TRAP] Found {len(trapping_moves)} potential traps")

        best_trap_move = None
        best_trap_score = float('-inf')

        for move in trapping_moves[:3]:
            try:
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                forecast.reverse_perspective()
                score = self.move_evaluator.evaluate_position(forecast)
                forecast.reverse_perspective()

                direction, move_type = move
                new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

                if new_loc in self.trapdoor_tracker.known_trapdoors:
                    score -= 1000000.0
                else:
                    danger = self.trapdoor_tracker.get_danger_score(new_loc)
                    score -= danger * 100000.0

                if new_loc in self.blocked_locations:
                    score -= 50000.0

                if new_loc in self.visited_squares:
                    score -= 400.0
                if new_loc in self.recent_positions:
                    score -= 600.0

                if score > best_trap_score:
                    best_trap_score = score
                    best_trap_move = move

            except Exception:
                continue

        if best_trap_move:
            direction, move_type = best_trap_move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

            if new_loc in self.trapdoor_tracker.known_trapdoors:
                return None
            if new_loc in self.blocked_locations:
                return None
            if board.is_cell_blocked(new_loc):
                return None

            danger = self.trapdoor_tracker.get_danger_score(new_loc)
            if danger > 0.3:
                return None

            print(f"[TRAP] {Direction(direction).name} + {MoveType(move_type).name} → {new_loc} (score: {best_trap_score:.1f})")
            self.last_move_target = new_loc
            return best_trap_move

        return None

    def _search_best_move(self, board: board_module.Board, time_left: Callable) -> Optional[Tuple[Direction, MoveType]]:
        """Use minimax search with PURE HEURISTICS (no neural network)."""
        try:
            print(f"[SEARCH] Minimax (depth={self.search_engine.max_depth}) - PURE HEURISTICS")

            score, best_move = self.search_engine.search(
                board=board,
                time_left=time_left,
                trapdoor_tracker=self.trapdoor_tracker,
                visited_squares=self.visited_squares,
                recent_positions=self.recent_positions,
                blocked_locations=self.blocked_locations
            )

            if best_move:
                direction, move_type = best_move
                new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

                # Safety checks
                if new_loc in self.trapdoor_tracker.known_trapdoors:
                    return None
                if new_loc in self.blocked_locations:
                    return None
                if board.is_cell_blocked(new_loc):
                    return None

                danger = self.trapdoor_tracker.get_danger_score(new_loc)
                if danger > 0.3:
                    return None

                print(f"[MOVE] {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
                print(f"       Heuristic eval: {score:.1f}")

                self.search_engine.record_best_move(best_move, self.search_engine.max_depth)
                self.last_move_target = new_loc
                return best_move

        except Exception as e:
            print(f"[ERROR] Search failed: {e}")

        return None

    def _fallback_move(self, board: board_module.Board, valid_moves: List[Tuple[Direction, MoveType]]) -> Tuple[Direction, MoveType]:
        """Fallback heuristic evaluation."""
        move_scores = [
            (
                self.move_evaluator.quick_evaluate_move(
                    m, board, self.trapdoor_tracker,
                    visited_squares=self.visited_squares,
                    recent_positions=self.recent_positions,
                    blocked_locations=self.blocked_locations
                ),
                m
            )
            for m in valid_moves
        ]

        move_scores.sort(reverse=True, key=lambda x: x[0])

        best_move = None
        for score, move in move_scores:
            direction, move_type = move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

            if new_loc in self.trapdoor_tracker.known_trapdoors:
                continue
            if new_loc in self.blocked_locations:
                continue
            if board.is_cell_blocked(new_loc):
                continue

            danger = self.trapdoor_tracker.get_danger_score(new_loc)
            if danger > 0.3:
                continue

            best_move = move
            best_score = score
            break

        if best_move is None and move_scores:
            best_score, best_move = move_scores[0]
            print("⚠ All fallback moves filtered!")

        direction, move_type = best_move
        new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

        print(f"[MOVE] {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
        print(f"       Score: {best_score:.1f}")

        self.last_move_target = new_loc
        return best_move

    def _print_turn_info(self, board: board_module.Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable):
        """Print turn information."""
        location = board.chicken_player.get_location()
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        turns_left = board.turns_left_player

        print(f"\n{'=' * 60}")
        print(f"AgentPro [NO NN] - Turn {self.turn_count}")
        print(f"{'=' * 60}")
        print(f"Position: {location}")
        print(f"Eggs: Me={my_eggs} | Enemy={enemy_eggs} | Diff={my_eggs - enemy_eggs:+d}")
        print(f"Turns left: {turns_left}")
        print(f"Time: {time_left():.2f}s")

        heard_w, felt_w = sensor_data[0]
        heard_b, felt_b = sensor_data[1]
        print(f"Sensors: W[H={heard_w},F={felt_w}] | B[H={heard_b},F={felt_b}]")

        likely_traps = self.trapdoor_tracker.get_most_likely_trapdoors(3)
        if likely_traps:
            print("Likely trapdoors:")
            for (x, y), prob in likely_traps:
                color = "white" if (x + y) % 2 == 0 else "black"
                print(f"  ({x},{y}) [{color}]: {prob:.1%}")

    def _print_game_over(self, board: board_module.Board):
        """Print game over info."""
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()

        print(f"\n{'=' * 60}")
        print(f"GAME OVER!")
        print(f"{'=' * 60}")

        if my_eggs > enemy_eggs:
            print(f"✓ WIN! ({my_eggs} vs {enemy_eggs})")
        elif enemy_eggs > my_eggs:
            print(f"✗ LOSS ({enemy_eggs} vs {my_eggs})")
        else:
            print(f"⚖ TIE ({my_eggs})")

        print(f"{'=' * 60}\n")
