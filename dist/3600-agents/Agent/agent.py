from collections.abc import Callable
from typing import List, Set, Tuple, Optional, Dict
import numpy as np
from game import *
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

"""
AgentA is an advanced strategic agent designed to consistently win by maximizing eggs.
Key strategies:
1. Aggressive egg-laying: Always lay eggs when possible (especially corners worth 3x)
2. Look-ahead evaluation: Use forecast_move to evaluate future positions
3. Pathfinding to corners: Calculate shortest paths to valuable corner positions
4. Probabilistic trapdoor inference: Use sensor data to build probability maps
5. Strategic blocking: Use turds to block enemy paths to corners/valuable areas
6. End-game optimization: Maximize eggs in final turns
7. Position evaluation: Consider board control and future opportunities
"""

class PlayerAgent:
    """
    AgentA - An agent focused on maximizing eggs to win
    """

    def __init__(self, board: board_module.Board, time_left: Callable):
        # Track suspected trapdoor locations based on sensor data
        self.suspected_trapdoors: Set[Tuple[int, int]] = set()
        # Track known trapdoors (ones we've stepped on or found)
        self.known_trapdoors: Set[Tuple[int, int]] = set()
        # Track sensor history to infer trapdoor locations
        self.sensor_history: List[Tuple[Tuple[int, int], List[Tuple[bool, bool]]]] = []
        # Track recent positions to avoid getting stuck in loops
        self.recent_positions: List[Tuple[int, int]] = []
        # Track visit count for each position to discourage revisiting
        self.position_visit_count: Dict[Tuple[int, int], int] = {}

    def _is_corner(self, loc: Tuple[int, int], map_size: int) -> bool:
        """Check if location is a corner (worth 3 eggs instead of 1)"""
        x, y = loc
        return (x == 0 or x == map_size - 1) and (y == 0 or y == map_size - 1)

    def _update_trapdoor_beliefs(
        self, 
        current_loc: Tuple[int, int], 
        sensor_data: List[Tuple[bool, bool]],
        map_size: int = 8
    ):
        """Update beliefs about trapdoor locations based on sensor data"""
        
        # Store sensor data with location
        self.sensor_history.append((current_loc, sensor_data))
        
        # For each trapdoor sensor reading
        for trapdoor_idx, (heard, felt) in enumerate(sensor_data):
            if heard or felt:
                # If we heard or felt something, the trapdoor is nearby
                # Check all squares within range that could be the trapdoor
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        # Skip if too far for hearing
                        if abs(dx) > 2 or abs(dy) > 2:
                            continue
                        if abs(dx) == 2 and abs(dy) == 2:
                            continue
                        
                        candidate = (current_loc[0] + dx, current_loc[1] + dy)
                        
                        # Check if valid location
                        if (0 <= candidate[0] < map_size and 
                            0 <= candidate[1] < map_size):
                            
                            # Calculate expected probabilities
                            delta_x, delta_y = abs(dx), abs(dy)
                            
                            # Check if this location matches the sensor reading
                            if felt:
                                # Must be adjacent (within 1)
                                if delta_x <= 1 and delta_y <= 1:
                                    self.suspected_trapdoors.add(candidate)
                            elif heard:
                                # Can be up to 2 away
                                if delta_x <= 2 and delta_y <= 2:
                                    if not (delta_x == 2 and delta_y == 2):
                                        self.suspected_trapdoors.add(candidate)

    def _is_safe_location(self, loc: Tuple[int, int]) -> bool:
        """Check if location is safe (not a known or suspected trapdoor)"""
        return loc not in self.known_trapdoors and loc not in self.suspected_trapdoors
    
    def _manhattan_distance(self, p1: Tuple[int, int], p2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points"""
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
    
    def _distance_to_nearest_corner(self, loc: Tuple[int, int], map_size: int) -> int:
        """Calculate distance to nearest corner"""
        corners = [(0, 0), (0, map_size - 1), (map_size - 1, 0), (map_size - 1, map_size - 1)]
        return min(self._manhattan_distance(loc, corner) for corner in corners)
    
    def _evaluate_board_state(self, board: board_module.Board) -> float:
        """Evaluate the current board state - higher is better for us"""
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        
        # Primary: egg difference (most important)
        score = (my_eggs - enemy_eggs) * 100
        
        # Secondary: position value - being closer to corners is good
        my_loc = board.chicken_player.get_location()
        my_dist_to_corner = self._distance_to_nearest_corner(my_loc, board.game_map.MAP_SIZE)
        enemy_loc = board.chicken_enemy.get_location()
        enemy_dist_to_corner = self._distance_to_nearest_corner(enemy_loc, board.game_map.MAP_SIZE)
        
        # Being closer to corners gives us advantage
        score += (enemy_dist_to_corner - my_dist_to_corner) * 5
        
        # Bonus for having more turns left
        score += board.turns_left_player * 2
        
        # Penalty if enemy has more turns
        score -= board.turns_left_enemy * 1
        
        return score
    
    def _evaluate_move_with_lookahead(
        self,
        move: Tuple[Direction, MoveType],
        board: board_module.Board,
        depth: int = 1
    ) -> float:
        """Evaluate a move by looking ahead and considering future positions"""
        dir, move_type = move
        
        # Try to forecast this move
        try:
            forecast_board = board.forecast_move(dir, move_type, check_ok=False)
            if forecast_board is None:
                return -10000  # Invalid move
        except:
            return -10000
        
        # Evaluate the resulting board state
        immediate_score = self._evaluate_board_state(forecast_board)
        
        # Add immediate move value
        current_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(current_loc, dir)
        map_size = board.game_map.MAP_SIZE
        
        # Immediate benefits
        if move_type == MoveType.EGG:
            immediate_score += 50
            if self._is_corner(new_loc, map_size):
                immediate_score += 100  # Corner eggs are very valuable
        elif move_type == MoveType.TURD:
            immediate_score += 5  # Turds have some value
        
        # Look ahead: if we can lay an egg next turn from this position, that's good
        if depth > 0 and forecast_board is not None:
            try:
                # Check if we can lay an egg from the new position
                future_moves = forecast_board.get_valid_moves()
                egg_moves = [m for m in future_moves if m[1] == MoveType.EGG]
                if egg_moves:
                    immediate_score += 20  # Bonus for positions that allow egg-laying
                    
                    # Check if any egg moves go to corners
                    for egg_move in egg_moves:
                        future_loc = loc_after_direction(new_loc, egg_move[0])
                        if self._is_corner(future_loc, map_size):
                            immediate_score += 30  # Even better if we can reach a corner
            except:
                pass  # If lookahead fails, just use immediate score
        
        return immediate_score

    def _score_move(
        self, 
        move: Tuple[Direction, MoveType], 
        board: board_module.Board
    ) -> float:
        """Score a move based on strategic value with look-ahead"""
        dir, move_type = move
        current_loc = board.chicken_player.get_location()
        
        # Get destination location using utility function
        new_loc = loc_after_direction(current_loc, dir)
        
        score = 0.0
        
        # Safety: avoid trapdoors (very high penalty)
        if not self._is_safe_location(new_loc):
            score -= 5000
        
        # Use look-ahead evaluation for better decisions
        try:
            lookahead_score = self._evaluate_move_with_lookahead(move, board, depth=1)
            score += lookahead_score
        except:
            # If lookahead fails, fall back to simple evaluation
            pass
        
        # Immediate move value
        map_size = board.game_map.MAP_SIZE
        
        # AGGRESSIVE EGG STRATEGY: Always prioritize eggs
        if move_type == MoveType.EGG:
            score += 200  # Base value for eggs (very high)
            
            # Corner eggs are extremely valuable (3x points)
            if self._is_corner(new_loc, map_size):
                score += 150  # Massive bonus for corners
            
            # Check if this is a valid egg square (parity)
            if (new_loc[0] + new_loc[1]) % 2 == board.chicken_player.even_chicken:
                score += 30
            else:
                score -= 1000  # Can't lay egg here anyway, but this shouldn't happen
        
        # Turds: Use strategically but eggs are always better
        elif move_type == MoveType.TURD:
            enemy_loc = board.chicken_enemy.get_location()
            dist_to_enemy = self._manhattan_distance(new_loc, enemy_loc)
            
            if board.chicken_player.get_turds_left() > 0:
                my_eggs = board.chicken_player.get_eggs_laid()
                enemy_eggs = board.chicken_enemy.get_eggs_laid()
                
                # Strategic turd placement
                if my_eggs > enemy_eggs + 2:
                    # We're significantly ahead - block enemy aggressively
                    if dist_to_enemy <= 4:
                        score += 50
                elif my_eggs >= enemy_eggs:
                    # We're ahead or tied - moderate blocking
                    if dist_to_enemy <= 3:
                        score += 30
                else:
                    # We're behind - only block if very close
                    if dist_to_enemy <= 2:
                        score += 20
                
                # Block enemy paths to corners
                enemy_dist_to_corner = self._distance_to_nearest_corner(enemy_loc, map_size)
                if enemy_dist_to_corner <= 3:
                    # Enemy is close to a corner - blocking might prevent them
                    if dist_to_enemy <= 3:
                        score += 25
        
        # Plain moves: Only if necessary to reach egg-laying positions
        else:
            # IMPORTANT: We can move to squares with our own eggs using PLAIN moves
            if new_loc in board.eggs_player:
                score += 5  # Moving over our own eggs is fine, helps us explore
            
            # Check if this move gets us closer to a corner
            dist_before = self._distance_to_nearest_corner(current_loc, map_size)
            dist_after = self._distance_to_nearest_corner(new_loc, map_size)
            if dist_after < dist_before:
                score += 20  # Moving toward corner is good
            
            # Check if this position allows us to lay an egg next turn
            # (i.e., if the new location has the right parity)
            if (new_loc[0] + new_loc[1]) % 2 == board.chicken_player.even_chicken:
                score += 30  # This position allows egg-laying - very valuable
            else:
                # Even if we can't lay an egg here, moving around is better than staying stuck
                score += 5  # Plain moves help us explore and find new egg-laying opportunities
        
        # End-game urgency: In final turns, eggs are critical
        turns_left = board.turns_left_player
        if turns_left <= 15:
            if move_type == MoveType.EGG:
                score += 50  # Extra urgency
            if turns_left <= 5:
                if move_type == MoveType.EGG:
                    score += 100  # Very urgent in final 5 turns
        
        # Early game: Focus on getting to corners
        if turns_left >= 30:
            if self._is_corner(new_loc, map_size) and move_type == MoveType.EGG:
                score += 100  # Early corner eggs are very valuable
        
        return score

    def play(
        self,
        board: board_module.Board,
        sensor_data: List[Tuple[bool, bool]],
        time_left: Callable,
    ):
        """
        Main play method - returns the best move
        """
        location = board.chicken_player.get_location()
        print(f"I'm at {location}.")
        print(f"Trapdoor A: heard? {sensor_data[0][0]}, felt? {sensor_data[0][1]}")
        print(f"Trapdoor B: heard? {sensor_data[1][0]}, felt? {sensor_data[1][1]}")
        print(f"Starting to think with {time_left()} seconds left.")
        print(f"I have {board.chicken_player.get_eggs_laid()} eggs, enemy has {board.chicken_enemy.get_eggs_laid()} eggs.")
        
        # Update trapdoor beliefs based on sensor data
        self._update_trapdoor_beliefs(location, sensor_data, board.game_map.MAP_SIZE)
        
        # Update known trapdoors from board
        if hasattr(board, 'found_trapdoors'):
            self.known_trapdoors.update(board.found_trapdoors)
        
        # Track position history to avoid getting stuck
        self.recent_positions.append(location)
        if len(self.recent_positions) > 10:  # Keep last 10 positions
            self.recent_positions.pop(0)
        
        # Count visits to each position
        if location not in self.position_visit_count:
            self.position_visit_count[location] = 0
        self.position_visit_count[location] += 1
        
        # Get all valid moves
        valid_moves = board.get_valid_moves()
        
        if not valid_moves:
            # No valid moves (this shouldn't happen)
            print("No valid moves available!")
            return (Direction.UP, MoveType.PLAIN)
        
        # AGGRESSIVE STRATEGY: Always prefer egg moves if available
        egg_moves = [m for m in valid_moves if m[1] == MoveType.EGG]
        safe_egg_moves = [m for m in egg_moves if self._is_safe_location(
            loc_after_direction(board.chicken_player.get_location(), m[0])
        )]
        
        print(f"Found {len(egg_moves)} egg moves, {len(safe_egg_moves)} safe egg moves.")
        
        # If we have safe egg moves, only consider those (eggs are always best)
        if safe_egg_moves:
            moves_to_evaluate = safe_egg_moves
        else:
            # If no safe egg moves, consider all moves but prioritize safe ones
            # Prioritize plain moves that get us to new areas or closer to egg-laying positions
            moves_to_evaluate = valid_moves
            print("No safe egg moves available - exploring to find new opportunities.")
        
        # Score all candidate moves (with exploration bonus to avoid getting stuck)
        scored_moves = [(move, self._score_move(move, board)) for move in moves_to_evaluate]
        
        # Add exploration bonuses/penalties: prefer moves to new/unvisited areas
        for i, (move, score) in enumerate(scored_moves):
            new_loc = loc_after_direction(location, move[0])
            
            # Get visit count for this position
            visit_count = self.position_visit_count.get(new_loc, 0)
            
            # Strong bonus for completely new positions (never visited)
            if visit_count == 0:
                scored_moves[i] = (move, score + 50)  # Large exploration bonus
            # Moderate bonus for positions visited only once
            elif visit_count == 1:
                scored_moves[i] = (move, score + 20)  # Moderate exploration bonus
            # Small bonus for positions visited 2-3 times
            elif visit_count <= 3:
                scored_moves[i] = (move, score + 5)  # Small exploration bonus
            # Penalty for positions visited many times
            elif visit_count >= 5:
                scored_moves[i] = (move, score - 30)  # Strong penalty for over-visited positions
            elif visit_count >= 4:
                scored_moves[i] = (move, score - 15)  # Moderate penalty
            
            # Additional penalty for going back to very recent positions (last 2 moves)
            if new_loc in self.recent_positions[-2:]:
                scored_moves[i] = (move, scored_moves[i][1] - 20)  # Avoid immediate backtracking
            
            # Extra penalty if we're about to revisit the current position (shouldn't happen, but just in case)
            if new_loc == location:
                scored_moves[i] = (move, score - 100)  # Very strong penalty for staying put
        
        # Filter out moves that go to unsafe locations (unless no other option)
        safe_moves = [m for m, s in scored_moves if s > -1000]
        
        if safe_moves:
            # Choose from safe moves
            scored_safe = [(m, s) for m, s in scored_moves if s > -1000]
        else:
            # If no safe moves, use all moves (better than crashing)
            scored_safe = scored_moves
        
        # Sort by score (highest first)
        scored_safe.sort(key=lambda x: x[1], reverse=True)
        
        # Choose the best move
        best_move = scored_safe[0][0]
        best_loc = loc_after_direction(location, best_move[0])
        print(f"I have {time_left()} seconds left. Playing {best_move} to {best_loc}.")
        
        return best_move

