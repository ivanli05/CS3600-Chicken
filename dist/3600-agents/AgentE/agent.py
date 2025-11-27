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

class PlayerAgent:
    def __init__(self, board: board_module.Board, time_left: Callable):
        self.map_size = board.game_map.MAP_SIZE
        self.trapdoor_tracker = TrapdoorTracker(map_size=self.map_size)
        self.move_evaluator = MoveEvaluator(map_size=self.map_size)
        
        self.search_engine = SearchEngine(
            evaluator=self.move_evaluator,
            max_depth=20,  # Iterative deepening up to depth 20
            time_limit=0.8
        )
        
        self.turn_count = 0
        self.last_location = None
        self.last_move_attempt = None
        self.last_egg_count = 0

    def play(self, board: board_module.Board, sensor_data: List[Tuple[bool, bool]], time_left: Callable) -> Tuple[Direction, MoveType]:
        self.turn_count += 1
        my_loc = board.chicken_player.get_location()
        
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
        
        print(f"\n--- DEBUGGING TURN {self.turn_count} ---")
        # We check the first few moves in safe_moves
        moves_to_check = safe_moves[:1]
        
        for move in moves_to_check:
            # We must forecast the move to evaluate the resulting board state
            # check_ok=False is safe here because we pulled these from get_valid_moves
            sim_board = board.forecast_move(move[0], move[1], check_ok=False)
            if sim_board:
                move_str = f"{move[0].name}_{move[1].name}"
                # Call the debug function you added to heuristics.py
                self.move_evaluator.debug_evaluate_position(
                    sim_board, 
                    self.trapdoor_tracker, 
                    move_name=move_str
                )
        print("-----------------------------------")

        # 3. Smart Turds (Strategic Blocking)
        # Check if placing a turd reduces the opponent's "Egg Count" heuristic
        best_turd_move = None
        max_turd_impact = 0.0
        turd_moves = [m for m in safe_moves if m[1] == MoveType.TURD]
        turds_left = board.chicken_player.get_turds_left()
        
        if turds_left > 0:
            # 1. Get enemy potential BEFORE the move
            enemy_potential_before, _ = self.move_evaluator._analyze_reachability(board, is_me=False, trapdoor_tracker=None)
            
            for move in turd_moves:
                # FIX: Remove check_ok=False. We NEED the full update to register the turd.
                forecast = board.forecast_move(move[0], move[1])
                
                if forecast:
                    # 2. Check enemy potential AFTER the move
                    enemy_potential_after, _ = self.move_evaluator._analyze_reachability(forecast, is_me=False, trapdoor_tracker=None)
                    
                    impact = enemy_potential_before - enemy_potential_after
                    
                    # Logic: If we have lots of turds, use them more loosely (+1 bonus).
                    # Otherwise, require a solid impact (>= 2.0 reduction in enemy potential).
                    score_bonus = 1.0 if turds_left >= 3 else 0.0
                    
                    if (impact + score_bonus) >= 2.0: 
                        if impact > max_turd_impact:
                            max_turd_impact = impact
                            best_turd_move = move

            if best_turd_move and max_turd_impact >= 2.0:
                print(f"!!! STRATEGIC TURD: Denied {max_turd_impact:.2f} enemy potential!")
                # Verify safety one last time with a quick depth-1 check to avoid suicide
                check_board = board.forecast_move(best_turd_move[0], best_turd_move[1])
                safety_score = self.move_evaluator.evaluate_position(check_board, self.trapdoor_tracker)
                
                if safety_score > -5000: # As long as we don't die instantly
                    self.last_move_attempt = loc_after_direction(my_loc, best_turd_move[0])
                    return best_turd_move
        # 4. Search
        score, best_move = self.search_engine.search(
            board, 
            time_left, 
            trapdoor_tracker=self.trapdoor_tracker,
            root_moves=safe_moves
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