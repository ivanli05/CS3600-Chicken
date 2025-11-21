"""
AgentC - Advanced Strategic Chicken Agent

A competitive agent that combines:
- Enhanced Bayesian trapdoor inference
- Minimax search with alpha-beta pruning
- Strategic move evaluation with enemy tracking
- Aggressive corner seeking (3x egg value)
- Enemy egg location tracking to avoid blocked squares

Author: AgentC Team
"""

from collections.abc import Callable
from typing import List, Tuple, Optional, Set
import numpy as np

from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

# Import our modules
from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .heuristics import MoveEvaluator


class PlayerAgent:
    """
    AgentC - A strategic agent using search algorithms, probabilistic inference,
    and enemy tracking to maximize egg laying.
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
            time_limit=0.75  # Use 75% of available time per move
        )

        # Game state tracking
        self.position_history: List[Tuple[int, int]] = []
        self.recent_positions: List[Tuple[int, int]] = []  # Track recent positions (last 10 moves)
        self.visited_squares: set = set()  # Track all visited squares for exploration
        self.last_location: Optional[Tuple[int, int]] = None
        self.last_move_target: Optional[Tuple[int, int]] = None
        self.blocked_locations: set = set()  # Track enemy eggs/barriers that block movement
        self.turn_count = 0

        # ENEMY TRACKING - Critical for avoiding enemy egg locations
        self.enemy_egg_locations: Set[Tuple[int, int]] = set()  # Track where enemy has laid eggs
        self.enemy_position_history: List[Tuple[int, int]] = []  # Track enemy movement
        self.last_enemy_eggs_count = 0  # Track enemy egg count to detect new eggs
        self.last_enemy_location: Optional[Tuple[int, int]] = None

        print(f"✓ AgentC initialized (search_depth={self.search_engine.max_depth})")

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
        enemy_location = board.chicken_enemy.get_location()

        # Detect if we stepped on a trapdoor (location was reset unexpectedly)
        if self.last_move_target is not None:
            spawn_location = board.chicken_player.get_spawn()
            if location == spawn_location and location != self.last_location:
                # We were reset to spawn - the last move target was a trapdoor!
                self.trapdoor_tracker.mark_trapdoor_found(self.last_move_target)
                print(f"🚨 TRAPDOOR DETECTED at {self.last_move_target} (location reset to spawn)")
                self.visited_squares.add(self.last_move_target)
            elif location != self.last_move_target and location == self.last_location:
                # We tried to move but didn't move - likely hit an enemy egg/barrier!
                if self.last_move_target not in self.trapdoor_tracker.known_trapdoors:
                    self.blocked_locations.add(self.last_move_target)
                    print(f"🚫 BLOCKED LOCATION DETECTED at {self.last_move_target} (enemy egg/barrier)")
        
        # Check if board has found_trapdoors attribute
        if hasattr(board, 'found_trapdoors'):
            for trapdoor_loc in board.found_trapdoors:
                self.trapdoor_tracker.mark_trapdoor_found(trapdoor_loc)
                self.visited_squares.add(trapdoor_loc)
        
        # Update blocked locations from current board state
        if hasattr(board, 'eggs_enemy'):
            for egg_loc in board.eggs_enemy:
                self.blocked_locations.add(egg_loc)
                # Track enemy egg locations
                self.enemy_egg_locations.add(egg_loc)
        
        # Check enemy turd zones (turds block the square and adjacent squares)
        if hasattr(board, 'turds_enemy'):
            for turd_loc in board.turds_enemy:
                self.blocked_locations.add(turd_loc)
                for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                    adjacent = loc_after_direction(turd_loc, direction)
                    if board.is_valid_cell(adjacent):
                        self.blocked_locations.add(adjacent)
        
        # ENEMY TRACKING: Detect where enemy laid eggs
        current_enemy_eggs_count = board.chicken_enemy.get_eggs_laid()
        if current_enemy_eggs_count > self.last_enemy_eggs_count:
            # Enemy laid an egg! Try to figure out where
            # If enemy was at a location last turn and now has more eggs, they likely laid it there
            if self.last_enemy_location is not None:
                # Check if enemy is still at that location (they might have moved)
                # If enemy moved, the egg is at the previous location
                if enemy_location != self.last_enemy_location:
                    # Enemy moved, so egg is at previous location
                    self.enemy_egg_locations.add(self.last_enemy_location)
                    self.blocked_locations.add(self.last_enemy_location)
                    print(f"📍 Detected enemy egg at {self.last_enemy_location} (enemy moved)")
                else:
                    # Enemy didn't move, egg is likely at current location
                    self.enemy_egg_locations.add(enemy_location)
                    self.blocked_locations.add(enemy_location)
                    print(f"📍 Detected enemy egg at {enemy_location} (enemy stayed)")
            
            # Also check board's enemy eggs set
            if hasattr(board, 'eggs_enemy'):
                for egg_loc in board.eggs_enemy:
                    self.enemy_egg_locations.add(egg_loc)
        
        self.last_enemy_eggs_count = current_enemy_eggs_count
        self.last_enemy_location = enemy_location
        
        # Track enemy position history
        self.enemy_position_history.append(enemy_location)
        if len(self.enemy_position_history) > 20:
            self.enemy_position_history.pop(0)
        
        # Mark current location as visited
        self.visited_squares.add(location)
        self.last_location = location

        # Print turn information
        self._print_turn_info(board, sensor_data, time_left)

        # Handle game over
        if board.is_game_over():
            self._print_game_over(board)
            return (Direction.UP, MoveType.PLAIN)

        # Update trapdoor beliefs based on sensor data
        self.trapdoor_tracker.update_beliefs(location, sensor_data)
        
        # Check if current location should be marked as a known trapdoor
        danger = self.trapdoor_tracker.get_danger_score(location)
        if danger > 0.8:  # Very high probability
            self.trapdoor_tracker.mark_trapdoor_found(location)
            print(f"⚠ Marked trapdoor at {location} (high probability: {danger:.1%})")

        # Track position history
        self.position_history.append(location)
        if len(self.position_history) > 20:
            self.position_history.pop(0)
        
        # Track recent positions to avoid repetition
        self.recent_positions.append(location)
        if len(self.recent_positions) > 10:
            self.recent_positions.pop(0)

        # Get valid moves
        valid_moves = board.get_valid_moves()
        if not valid_moves:
            print("⚠ WARNING: No valid moves available!")
            return (Direction.UP, MoveType.PLAIN)

        # FILTER OUT moves to known trapdoors, blocked locations, and enemy egg locations
        safe_moves = []
        for move in valid_moves:
            direction, move_type = move
            target_loc = loc_after_direction(location, direction)
            
            # Skip known trapdoors entirely
            if target_loc in self.trapdoor_tracker.known_trapdoors:
                print(f"⚠ Filtered out move to known trapdoor at {target_loc}")
                continue
            
            # Skip blocked locations (enemy eggs/barriers)
            if target_loc in self.blocked_locations:
                print(f"⚠ Filtered out move to blocked location at {target_loc}")
                continue
            
            # Skip enemy egg locations (can't lay eggs there)
            if target_loc in self.enemy_egg_locations:
                print(f"⚠ Filtered out move to enemy egg location at {target_loc}")
                continue
            
            # Check if board says it's blocked
            if board.is_cell_blocked(target_loc):
                self.blocked_locations.add(target_loc)
                print(f"⚠ Filtered out move to blocked location at {target_loc} (board reports blocked)")
                continue
            
            # Skip moves with very high trapdoor probability (>30%)
            danger = self.trapdoor_tracker.get_danger_score(target_loc)
            if danger > 0.3:
                print(f"⚠ Filtered out move to high-risk location {target_loc} (danger: {danger:.1%})")
                continue
            
            safe_moves.append(move)
        
        # If we filtered out all moves, use original moves but with heavy penalties
        if not safe_moves:
            print("⚠ WARNING: All moves filtered! Using original moves with heavy penalties...")
            safe_moves = valid_moves
        else:
            valid_moves = safe_moves
            print(f"Valid safe moves: {len(valid_moves)} (filtered {len(board.get_valid_moves()) - len(valid_moves)} unsafe moves)")

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
    
    def get_visited_squares(self) -> set:
        """Get the set of visited squares for exploration penalty"""
        return self.visited_squares
    
    def get_recent_positions(self) -> List[Tuple[int, int]]:
        """Get recent positions to avoid repetition"""
        return self.recent_positions
    
    def get_blocked_locations(self) -> set:
        """Get the set of blocked locations (enemy eggs/barriers)"""
        return self.blocked_locations
    
    def get_enemy_egg_locations(self) -> Set[Tuple[int, int]]:
        """Get the set of enemy egg locations"""
        return self.enemy_egg_locations

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

                forecast.reverse_perspective()
                score = self.move_evaluator.evaluate_position(forecast)
                forecast.reverse_perspective()
                
                direction, move_type = move
                new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
                
                # Massive penalty for known trapdoors
                if new_loc in self.trapdoor_tracker.known_trapdoors:
                    score -= 1000000.0
                else:
                    danger = self.trapdoor_tracker.get_danger_score(new_loc)
                    if danger > 0.1:
                        score -= danger * 100000.0
                    else:
                        score -= danger * 50000.0
                
                # Massive penalty for blocked locations
                if new_loc in self.blocked_locations:
                    score -= 50000.0
                
                # Penalty for enemy egg locations
                if new_loc in self.enemy_egg_locations:
                    score -= 30000.0
                
                # Strong penalties for revisiting
                if new_loc in self.visited_squares:
                    score -= 500.0
                if new_loc in self.recent_positions:
                    score -= 700.0

                if score > best_trap_score:
                    best_trap_score = score
                    best_trap_move = move

            except Exception:
                continue

        if best_trap_move:
            direction, move_type = best_trap_move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
            
            # Double-check: never return a move to a known trapdoor
            if new_loc in self.trapdoor_tracker.known_trapdoors:
                print(f"⚠ Rejected trapping move to known trapdoor at {new_loc}")
                return None
            
            # Double-check: never return a move to a blocked location
            if new_loc in self.blocked_locations:
                print(f"⚠ Rejected trapping move to blocked location at {new_loc}")
                return None
            
            # Check enemy egg locations
            if new_loc in self.enemy_egg_locations:
                print(f"⚠ Rejected trapping move to enemy egg location at {new_loc}")
                return None
            
            # Check if board says it's blocked
            if board.is_cell_blocked(new_loc):
                self.blocked_locations.add(new_loc)
                print(f"⚠ Rejected trapping move to blocked location at {new_loc}")
                return None
            
            # Check danger level
            danger = self.trapdoor_tracker.get_danger_score(new_loc)
            if danger > 0.3:
                print(f"⚠ Rejected trapping move to high-risk location {new_loc} (danger: {danger:.1%})")
                return None
            
            print(f"[TRAP] Selected: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
            print(f"       Score: {best_trap_score:.1f}")
            self.last_move_target = new_loc
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
                trapdoor_tracker=self.trapdoor_tracker,
                visited_squares=self.visited_squares,
                recent_positions=self.recent_positions,
                blocked_locations=self.blocked_locations,
                enemy_egg_locations=self.enemy_egg_locations
            )

            if best_move:
                direction, move_type = best_move
                new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
                
                # Final safety check: never return a move to a known trapdoor
                if new_loc in self.trapdoor_tracker.known_trapdoors:
                    print(f"⚠ Rejected minimax move to known trapdoor at {new_loc}")
                    return None
                
                # Final safety check: never return a move to a blocked location
                if new_loc in self.blocked_locations:
                    print(f"⚠ Rejected minimax move to blocked location at {new_loc}")
                    return None
                
                # Check enemy egg locations
                if new_loc in self.enemy_egg_locations:
                    print(f"⚠ Rejected minimax move to enemy egg location at {new_loc}")
                    return None
                
                # Check if board says it's blocked
                if board.is_cell_blocked(new_loc):
                    self.blocked_locations.add(new_loc)
                    print(f"⚠ Rejected minimax move to blocked location at {new_loc}")
                    return None
                
                # Check danger level
                danger = self.trapdoor_tracker.get_danger_score(new_loc)
                if danger > 0.3:
                    print(f"⚠ Rejected minimax move to high-risk location {new_loc} (danger: {danger:.1%})")
                    return None

                print(f"[MOVE] Minimax chose: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
                print(f"       Evaluation: {score:.1f}")

                self.search_engine.record_best_move(best_move, self.search_engine.max_depth)
                
                self.last_move_target = new_loc
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
                    m, board, self.trapdoor_tracker, 
                    visited_squares=self.visited_squares,
                    recent_positions=self.recent_positions,
                    blocked_locations=self.blocked_locations,
                    enemy_egg_locations=self.enemy_egg_locations
                ),
                m
            )
            for m in valid_moves
        ]

        move_scores.sort(reverse=True, key=lambda x: x[0])
        
        # Find the best move that's safe
        best_move = None
        for score, move in move_scores:
            direction, move_type = move
            new_loc = loc_after_direction(board.chicken_player.get_location(), direction)
            
            # Skip known trapdoors
            if new_loc in self.trapdoor_tracker.known_trapdoors:
                continue
            
            # Skip blocked locations
            if new_loc in self.blocked_locations:
                continue
            
            # Skip enemy egg locations
            if new_loc in self.enemy_egg_locations:
                continue
            
            # Check if board says it's blocked
            if board.is_cell_blocked(new_loc):
                self.blocked_locations.add(new_loc)
                continue
            
            # Skip high-risk locations
            danger = self.trapdoor_tracker.get_danger_score(new_loc)
            if danger > 0.3:
                continue
            
            best_move = move
            best_score = score
            break
        
        # If all moves were filtered, use the best one anyway
        if best_move is None and move_scores:
            best_score, best_move = move_scores[0]
            print("⚠ WARNING: All fallback moves filtered, using best available (risky!)")

        direction, move_type = best_move
        new_loc = loc_after_direction(board.chicken_player.get_location(), direction)

        print(f"[MOVE] Heuristic chose: {Direction(direction).name} + {MoveType(move_type).name} → {new_loc}")
        print(f"       Score: {best_score:.1f}")
        
        self.last_move_target = new_loc
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
        print(f"AgentC - Turn {self.turn_count}")
        print(f"{'=' * 60}")
        print(f"Position: {location}")
        print(f"Eggs: Me={my_eggs} | Enemy={enemy_eggs} | Diff={my_eggs - enemy_eggs:+d}")
        print(f"Turds left: {my_turds}")
        print(f"Turns remaining: {turns_left}")
        print(f"Time left: {time_left():.2f}s")
        print(f"Enemy egg locations tracked: {len(self.enemy_egg_locations)}")

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
            print(f"✓ AgentC WINS! ({my_eggs} eggs vs {enemy_eggs} eggs)")
        elif enemy_eggs > my_eggs:
            print(f"✗ Enemy wins ({enemy_eggs} eggs vs {my_eggs} eggs)")
        else:
            print(f"⚖ TIE! (Both have {my_eggs} eggs)")

        print(f"{'=' * 60}\n")

