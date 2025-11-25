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
            max_depth=7,  # Search depth (can increase if you want deeper search)
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
        self.just_respawned = False  # Track if we just hit a trapdoor
        self.oscillation_count = 0  # Track if we're stuck oscillating
        self.last_two_positions: List[Tuple[int, int]] = []  # For oscillation detection
        
        # Smart loop-breaking state
        self.in_loop = False
        self.loop_center: Optional[Tuple[int, int]] = None
        self.force_outward_movement = False

    def play(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ) -> Tuple[Direction, MoveType]:
        """Main play method - called each turn to choose a move."""
        self.turn_count += 1
        location = board.chicken_player.get_location()

        # Detect trapdoors and respawns
        if self.last_move_target is not None:
            spawn_location = board.chicken_player.get_spawn()
            if location == spawn_location and location != self.last_location:
                # TRAPDOOR HIT - we respawned!
                self.trapdoor_tracker.mark_trapdoor_found(self.last_move_target)
                print(f"🚨 TRAPDOOR at {self.last_move_target}")
                self.visited_squares.add(self.last_move_target)

                # CRITICAL FIX: Clear recent position history to explore new areas!
                # After respawning, we should NOT be penalized for revisiting old squares
                print(f"🔄 RESPAWNED! Clearing recent position history to explore new areas...")
                self.recent_positions.clear()
                self.just_respawned = True
                # Keep visited_squares but clear recent memory - fresh exploration!

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

        # Reset force_outward_movement after a few moves (give it time to work)
        if self.force_outward_movement:
            # Keep it active for a few turns to ensure we break out
            if len(self.recent_positions) >= 4:
                # Check if we've moved away from loop center
                if self.loop_center is not None:
                    dist_to_center = abs(location[0] - self.loop_center[0]) + abs(location[1] - self.loop_center[1])
                    if dist_to_center >= 4:  # We've moved far enough away
                        print(f"✓ Successfully broke out of loop (distance from center: {dist_to_center})")
                        self.force_outward_movement = False
                        self.loop_center = None
                        self.in_loop = False

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

        # Reset just_respawned flag after a few moves
        if self.just_respawned and len(self.recent_positions) >= 3:
            self.just_respawned = False
            print("✓ Respawn period over - back to normal exploration")

        # SMART LOOP DETECTION: Detect loops and break them intelligently
        self.last_two_positions.append(location)
        if len(self.last_two_positions) > 8:  # Track more positions for better loop detection
            self.last_two_positions.pop(0)

        # Track loop state for heuristics
        self.in_loop = False
        self.loop_center = None
        self.force_outward_movement = False

        # Check for small loops (2-4 positions)
        if len(self.last_two_positions) >= 4:
            positions_set = set(self.last_two_positions)
            unique_count = len(positions_set)

            # Small loop detected (2-4 unique positions in recent moves)
            if unique_count <= 3:
                self.oscillation_count += 1
                self.in_loop = True
                
                # Calculate loop center (average position of loop)
                if unique_count >= 2:
                    loop_positions = list(positions_set)
                    avg_x = sum(pos[0] for pos in loop_positions) / len(loop_positions)
                    avg_y = sum(pos[1] for pos in loop_positions) / len(loop_positions)
                    self.loop_center = (int(avg_x), int(avg_y))
                
                print(f"⚠ SMALL LOOP DETECTED! Count: {self.oscillation_count} | Positions: {positions_set} | Center: {self.loop_center}")

                # SMART ACTION: Only clear recent_positions (keep visited_squares for exploration direction!)
                # CRITICAL FIX: Don't clear ALL recent positions - keep last 2 to prevent immediate re-looping
                if self.oscillation_count >= 2:
                    print("🔄 BREAKING LOOP - Partially clearing recent positions (keeping visited_squares for exploration direction)")
                    # Keep last 2 positions to prevent squares from immediately becoming "new" again
                    if len(self.recent_positions) > 2:
                        self.recent_positions = self.recent_positions[-2:]
                    if len(self.last_two_positions) > 2:
                        self.last_two_positions = self.last_two_positions[-2:]
                    self.oscillation_count = 0
                    self.force_outward_movement = True  # Force movement away from loop center
                    print(f"   → Forcing outward movement from loop center: {self.loop_center}")
            else:
                # Good movement - reset counter
                if self.oscillation_count > 0:
                    print("✓ Loop broken - resuming normal exploration")
                self.oscillation_count = 0

        # BIG LOOP DETECTION: Check for larger loops in recent positions
        if len(self.recent_positions) >= 8:
            position_counts = {}
            for pos in self.recent_positions:
                position_counts[pos] = position_counts.get(pos, 0) + 1

            max_revisits = max(position_counts.values()) if position_counts else 0
            
            # If we've visited same square 3+ times, we're in a bigger loop
            if max_revisits >= 3:
                most_visited_positions = [pos for pos, count in position_counts.items() if count == max_revisits]
                most_visited = most_visited_positions[0]
                
                # Calculate center of visited area (bigger loop)
                if len(most_visited_positions) >= 2:
                    avg_x = sum(pos[0] for pos in most_visited_positions) / len(most_visited_positions)
                    avg_y = sum(pos[1] for pos in most_visited_positions) / len(most_visited_positions)
                    self.loop_center = (int(avg_x), int(avg_y))
                else:
                    self.loop_center = most_visited
                
                self.in_loop = True
                self.force_outward_movement = True
                
                print(f"🚨 BIG LOOP DETECTED! Stuck at {most_visited} ({max_revisits} times) - Forcing outward movement!")
                print(f"   → Loop center: {self.loop_center} - Will force movement away from this area")
                
                # SMART ACTION: Partially clear recent_positions, keep visited_squares
                # CRITICAL FIX: Keep last 2 positions to prevent immediate re-looping
                if len(self.recent_positions) > 2:
                    self.recent_positions = self.recent_positions[-2:]
                # Allow revisiting the stuck square (but force movement away)
                self.visited_squares.discard(most_visited)

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("⚠ No valid moves!")
            return (Direction.UP, MoveType.PLAIN)

        # Filter out dangerous moves BUT ALLOW EGG MOVES (eggs are worth the risk!)
        safe_moves = []
        egg_moves = []  # Track egg moves separately

        for move in valid_moves:
            direction, move_type = move
            target_loc = loc_after_direction(location, direction)

            # Always allow egg moves (they're the primary goal!)
            if move_type == MoveType.EGG:
                # Still filter known trapdoors and physically blocked locations
                if target_loc in self.trapdoor_tracker.known_trapdoors:
                    print(f"⚠ Filtered EGG move to known trapdoor at {target_loc}")
                    continue

                if target_loc in self.blocked_locations or board.is_cell_blocked(target_loc):
                    print(f"⚠ Filtered EGG move to blocked location at {target_loc}")
                    continue

                # Allow risky egg moves! Let heuristics decide risk/reward
                egg_moves.append(move)
                safe_moves.append(move)
                continue

            # For PLAIN and TURD moves: apply safety filters
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
            if egg_moves:
                print(f"Safe moves: {len(valid_moves)} (including {len(egg_moves)} EGG moves)")
            else:
                print(f"Safe moves: {len(valid_moves)}")

        # Strategy 0: EGG-FIRST! If we can lay an egg, do it immediately!
        if egg_moves:
            print(f"[EGG-FIRST] {len(egg_moves)} egg moves available - prioritizing eggs!")
            best_egg = self._choose_best_egg_move(board, egg_moves)
            if best_egg:
                return best_egg

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

    def _choose_best_egg_move(self, board: board_module.Board, egg_moves: List[Tuple[Direction, MoveType]]) -> Optional[Tuple[Direction, MoveType]]:
        """Choose the best egg move from available options using minimax evaluation."""
        if not egg_moves:
            return None

        # If only one egg move, use it!
        if len(egg_moves) == 1:
            direction, move_type = egg_moves[0]
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
            print(f"[EGG] {Direction(direction).name} + EGG → {new_loc} (only option)")
            self.last_move_target = new_loc
            return egg_moves[0]

        # Multiple egg moves - use MINIMAX to evaluate strategic value!
        print(f"[EGG-EVAL] Evaluating {len(egg_moves)} egg moves with minimax...")
        best_egg = None
        best_score = float('-inf')

        for move in egg_moves:
            direction, move_type = move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

            # Try forecasting this egg move
            try:
                forecast = board.forecast_move(direction, move_type, check_ok=False)
                if forecast is None:
                    continue

                # Evaluate position AFTER laying this egg (from opponent's perspective)
                forecast.reverse_perspective()
                opponent_score = self.move_evaluator.evaluate_position(
                    forecast,
                    trapdoor_tracker=self.trapdoor_tracker
                )
                forecast.reverse_perspective()

                # Our score is negative of opponent's score
                strategic_score = -opponent_score

                # Add positional bonuses
                positional_bonus = 0.0

                # Prefer corners (3x egg value!)
                if self.move_evaluator._is_corner(new_loc):
                    if board.chicken_player.can_lay_egg(new_loc):
                        positional_bonus += 1500.0  # HUGE bonus for corners

                # Prefer center
                if self.move_evaluator._is_center(new_loc):
                    positional_bonus += 200.0

                # Prefer unvisited squares
                if new_loc not in self.visited_squares:
                    positional_bonus += 150.0

                # Prefer squares far from existing eggs (spread out)
                if hasattr(board, 'eggs_player') and board.eggs_player:
                    min_dist = min(
                        abs(new_loc[0] - egg[0]) + abs(new_loc[1] - egg[1])
                        for egg in board.eggs_player
                    )
                    positional_bonus += min_dist * 40.0

                # Light penalty for trapdoor danger (but eggs are still worth it!)
                danger = self.trapdoor_tracker.get_danger_score(new_loc)
                positional_bonus -= danger * 80.0

                # Combine strategic minimax score with positional heuristics
                # Weight strategic score more heavily (70/30 split)
                total_score = strategic_score * 0.7 + positional_bonus * 0.3

                if total_score > best_score:
                    best_score = total_score
                    best_egg = move

            except Exception:
                # Forecast failed - fall back to simple heuristics for this move
                score = 0.0
                # CRITICAL: Only incentivize corners where this chicken can lay eggs (parity check)
                if self.move_evaluator._is_corner(new_loc):
                    if board.chicken_player.can_lay_egg(new_loc):
                        score += 1000.0  # Accessible corner - huge bonus!
                    else:
                        score += 50.0  # Inaccessible corner - minimal bonus
                if self.move_evaluator._is_center(new_loc):
                    score += 100.0
                if new_loc not in self.visited_squares:
                    score += 200.0

                if score > best_score:
                    best_score = score
                    best_egg = move

        if best_egg:
            direction, move_type = best_egg
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
            print(f"[EGG] {Direction(direction).name} + EGG → {new_loc} (minimax score: {best_score:.1f})")
            self.last_move_target = new_loc
            return best_egg

        return None

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
                blocked_locations=self.blocked_locations,
                is_oscillating=self.in_loop,
                loop_center=self.loop_center,
                force_outward_movement=self.force_outward_movement
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
                    blocked_locations=self.blocked_locations,
                    just_respawned=self.just_respawned,
                    is_oscillating=self.in_loop,
                    loop_center=self.loop_center,
                    force_outward_movement=self.force_outward_movement
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
