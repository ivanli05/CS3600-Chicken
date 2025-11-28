"""
AgentE - Territory Control (UNIFIED)

1. Unified Safety Threshold (0.35) prevents paralysis.
2. Uses "Smart Turds" to deny enemy egg counts.
3. Fallback logic prioritizes highest heuristic score.
"""
from collections.abc import Callable
from typing import List, Tuple, Optional
from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

from .trapdoor_tracker import TrapdoorTracker
from .search_engine import SearchEngine
from .heuristics import MoveEvaluator

# Hyperparameters - adjust these to tune agent behavior
MAXDEPTH = 7  # Moderate depth for balanced performance
TIME_LIMIT = 0.8  # Fraction of remaining time to use per move

class PlayerAgent:
    def __init__(self, board: board_module.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.trapdoor_tracker = TrapdoorTracker(map_size=self.map_size)
        self.move_evaluator = MoveEvaluator(map_size=self.map_size)
        
        self.search_engine = SearchEngine(
            evaluator=self.move_evaluator,
            max_depth=MAXDEPTH,
            time_limit=TIME_LIMIT
        )
        
        self.turn_count = 0
        self.last_location = None
        self.last_move_attempt = None
        self.last_egg_count = 0
        self.visited_squares = set()  # Track visited squares for exploration bonus

    def play(self, board: board_module.Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable) -> Tuple[Direction, MoveType]:
        self.turn_count += 1
        my_loc = board.chicken_player.get_location()

        # Track visited squares for exploration bonus
        self.visited_squares.add(my_loc)

        # 1. Update Beliefs
        self._detect_teleport(board, my_loc)
        self.trapdoor_tracker.update_beliefs(my_loc, sensor_data)
        
        if hasattr(board, 'found_trapdoors') and board.found_trapdoors:
            for t in board.found_trapdoors:
                self.trapdoor_tracker.mark_trapdoor_found(t)

        self.last_location = my_loc
        self.last_egg_count = board.chicken_player.get_eggs_laid()

        # 2. Safety Filter
        valid_moves = board.get_valid_moves()
        safe_moves = []
        
        # Unified Threshold: If it's safe enough to egg, it's safe enough to walk.
        base_threshold = 0.35
        if self.turn_count > 60: base_threshold = 0.45 

        for move in valid_moves:
            target = loc_after_direction(my_loc, move[0])
            if target in self.trapdoor_tracker.known_trapdoors: continue
            
            danger = self.trapdoor_tracker.get_danger_score(target)
            if danger < base_threshold:
                safe_moves.append(move)
        
        # Fallback: Best risky move
        if not safe_moves:
            if valid_moves:
                print(f"⚠ Turn {self.turn_count}: FORCED RISK. Picking best from valid_moves.")
                # Sort valid moves by your heuristic.
                # This ensures we pick a move that might score points or at least
                # has the LOWEST danger score (since heuristic penalizes danger).
                valid_moves.sort(key=lambda m: self.move_evaluator.quick_evaluate_move(
                    m, board, self.trapdoor_tracker
                ), reverse=True)
                
                # Assign the least bad move
                safe_moves.append(valid_moves[0])
            else:
                # If valid_moves is empty, the game is likely over (stuck), 
                # but we return a dummy move to prevent a crash.
                return (Direction.UP, MoveType.PLAIN)
        
        # 3. Search (removed pre-search turd logic - let minimax decide)
        score, best_move = self.search_engine.search(
            board,
            time_left,
            trapdoor_tracker=self.trapdoor_tracker,
            root_moves=safe_moves,
            visited_squares=self.visited_squares
        )
        
        if best_move:
            self.last_move_attempt = loc_after_direction(my_loc, best_move[0])
            return best_move
            
        return safe_moves[0]

    def _detect_teleport(self, board, current_loc):
        spawn = board.chicken_player.get_spawn()
        if self.last_move_attempt and self.last_move_attempt != spawn:
            if current_loc == spawn:
                if board.chicken_player.get_eggs_laid() < self.last_egg_count:
                    print(f"🚨 TRAPDOOR DETECTED at {self.last_move_attempt}")
                    self.trapdoor_tracker.mark_trapdoor_found(self.last_move_attempt)