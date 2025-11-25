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
        recent_positions=None,
        blocked_locations=None,
        just_respawned=False,
        is_oscillating=False,
        loop_center=None,
        force_outward_movement=False
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

        # 0. OPPONENT AVOIDANCE: Stay away from enemy to maximize freedom!
        # BUT: Don't let this prevent exploration - reduced penalties
        dist_to_enemy = abs(new_loc[0] - enemy_loc[0]) + abs(new_loc[1] - enemy_loc[1])

        # Strong bonus for staying far from opponent
        if dist_to_enemy >= 6:
            score += 400.0  # Very far - maximum freedom!
        elif dist_to_enemy >= 5:
            score += 300.0  # Far - good freedom
        elif dist_to_enemy >= 4:
            score += 200.0  # Moderate distance
        elif dist_to_enemy >= 3:
            score += 100.0  # Some distance
        elif dist_to_enemy <= 2:
            # REDUCED penalty - sometimes we need to explore near opponent
            # Especially for egg moves, this is acceptable
            if move_type == MoveType.EGG:
                score -= 100.0  # Light penalty for eggs (reduced from 300)
            else:
                score -= 150.0  # Moderate penalty for plain moves (reduced from 300)

        # EXPLORATION SPREADING: Strongly prefer moving to unexplored quadrants
        # Divide map into 4 quadrants and spread out
        if visited_squares is not None and len(visited_squares) > 0:
            # Calculate which quadrant is least explored
            quadrant_visits = self._count_quadrant_visits(new_loc, visited_squares)
            current_quadrant_visits = quadrant_visits.get(self._get_quadrant(new_loc), 0)
            total_visits = sum(quadrant_visits.values())

            if total_visits > 0:
                # Bonus for moving to less-explored quadrants
                exploration_ratio = 1.0 - (current_quadrant_visits / total_visits)
                score += exploration_ratio * 300.0  # Up to +300 for unexplored quadrants

        # 0.5. MASSIVE penalty for blocked locations (enemy eggs/barriers)
        # CRITICAL FIX: Check board.turds_enemy directly to avoid treating own turds as obstacles
        # After perspective reversal in minimax, board.turds_enemy contains actual enemy turds
        if hasattr(board, 'turds_enemy') and board.turds_enemy:
            # Check if location is an enemy turd or adjacent to one
            if new_loc in board.turds_enemy:
                score -= 50000.0  # Extremely strong penalty - enemy turd
            else:
                # Check if adjacent to enemy turd
                for turd_loc in board.turds_enemy:
                    if abs(new_loc[0] - turd_loc[0]) + abs(new_loc[1] - turd_loc[1]) == 1:
                        score -= 50000.0  # Adjacent to enemy turd
                        break
        
        # Check blocked_locations for other things (trapdoors, etc.) but NOT turds
        # (we check turds directly from board above)
        if blocked_locations is not None and new_loc in blocked_locations:
            # Only apply penalty if it's NOT an enemy turd (already handled above)
            # This prevents double-penalizing, but still blocks trapdoors, etc.
            if not (hasattr(board, 'turds_enemy') and new_loc in board.turds_enemy):
                score -= 50000.0  # Extremely strong penalty - never waste a turn on a known barrier
        
        # Also check if board says it's blocked (catches newly placed enemy eggs/turds)
        if board.is_cell_blocked(new_loc):
            score -= 50000.0  # Same penalty for currently blocked locations

        # 0. ENDGAME BONUS: Eggs are MORE valuable near end of game!
        # Game ends at turn 40, so maximize eggs in last 10 turns
        turns_left = getattr(board, 'turns_left_player', 40)
        endgame_multiplier = 1.0
        if turns_left <= 10:
            # Last 10 turns: eggs are 3x more valuable!
            endgame_multiplier = 3.0
        elif turns_left <= 20:
            # Last 20 turns: eggs are 2x more valuable
            endgame_multiplier = 2.0

        # 1. Egg moves are HIGHLY valuable (direct scoring) - EMPHASIZED!
        # PRIORITY: Eggs are THE PRIMARY GOAL - make them extremely attractive!
        if move_type == MoveType.EGG:
            score += 6000.0 * endgame_multiplier  # INCREASED from 5000 - eggs are THE PRIMARY GOAL!
            
            # EXTRA BONUS: Strongly encourage laying eggs, even in slightly risky situations
            # This helps overcome trapdoor fear and encourages exploration
            score += 500.0  # Base exploration bonus for egg moves

            # DESPERATION BONUS: When we have few valid moves (trapped in corner), ALWAYS lay eggs!
            my_valid_moves = len(board.get_valid_moves())
            if my_valid_moves <= 3:
                score += 10000.0  # DESPERATE - lay eggs when trapped!
            elif my_valid_moves <= 5:
                score += 5000.0  # Limited mobility - prioritize eggs!

            # Bonus for laying eggs in new/unexplored areas - encourages spreading eggs
            if visited_squares is None or new_loc not in visited_squares:
                score += 300.0  # Extra bonus for eggs in completely new areas
            else:
                # Even if visited before (walked through), still good to lay egg!
                score += 150.0  # Moderate bonus - it's okay to lay eggs on visited squares

            # Check if there's already an egg here (shouldn't happen, but just in case)
            if hasattr(board, 'eggs_player') and new_loc in board.eggs_player:
                score -= 100000.0  # Can't lay egg where one already exists

            # Bonus for laying eggs far from existing eggs (spread out, don't cluster)
            if hasattr(board, 'eggs_player') and board.eggs_player:
                min_dist_to_existing_egg = min(
                    abs(new_loc[0] - egg[0]) + abs(new_loc[1] - egg[1])
                    for egg in board.eggs_player
                )
                # Bonus for spreading eggs out (farther from existing eggs = better)
                if min_dist_to_existing_egg >= 4:
                    score += 120.0  # Good spread bonus
                elif min_dist_to_existing_egg >= 3:
                    score += 60.0
                elif min_dist_to_existing_egg >= 2:
                    score += 30.0  # Even moderate distance is okay
                elif min_dist_to_existing_egg <= 1:
                    score -= 60.0  # Light penalty for clustering too close
            else:
                # First egg! Extra bonus
                score += 200.0

            # Corner eggs are MUCH better - they give 3 eggs instead of 1 (3x value!)
            # CRITICAL: Only incentivize corners where this chicken can lay eggs (parity check)
            # Each chicken can only lay eggs on 2 of the 4 corners (parity-based)
            if self._is_corner(new_loc):
                # Check if this chicken can lay eggs on this corner (parity check)
                can_lay_on_corner = board.chicken_player.can_lay_egg(new_loc)
                if can_lay_on_corner:
                    score += 1200.0  # INCREASED: HUGE bonus for accessible corner eggs! (3x value)
                else:
                    # Inaccessible corner - minimal bonus (can't lay eggs here anyway)
                    score += 20.0
            # Center eggs control the board
            if self._is_center(new_loc):
                score += 80.0

        # 1.5. POST-RESPAWN EXPLORATION: After hitting a trapdoor, strongly encourage exploring NEW areas!
        # Detect respawn: recent_positions is empty or very small
        recently_respawned = (recent_positions is None or len(recent_positions) <= 3)

        if just_respawned or recently_respawned:
            if move_type == MoveType.PLAIN:
                # STRONG bonus for exploring completely new squares after respawn
                if visited_squares is None or new_loc not in visited_squares:
                    score += 500.0  # HUGE bonus - go to new areas!
                else:
                    # Penalty for revisiting old squares (unless necessary)
                    score -= 300.0  # Discourage going back to old areas

                # EXTRA penalty for squares with our own eggs (can't lay another egg there)
                if hasattr(board, 'eggs_player') and new_loc in board.eggs_player:
                    score -= 800.0  # Strong penalty - that square is done!

            elif move_type == MoveType.EGG:
                # For egg moves: prefer unvisited squares even more!
                if visited_squares is None or new_loc not in visited_squares:
                    score += 1000.0  # MASSIVE bonus - new egg in new area!
                else:
                    # Still good to lay eggs in visited squares, but prefer new ones
                    score += 200.0  # Moderate bonus

        # 1.6. Plain moves that help exploration are valuable
        # Plain moves are necessary to reach new egg-laying locations
        # INCREASED bonuses to encourage exploration!
        if move_type == MoveType.PLAIN:
            # MUCH STRONGER bonus for plain moves to new squares (encourages exploration)
            if visited_squares is None or new_loc not in visited_squares:
                score += 200.0  # INCREASED from 40 - strong incentive to explore!
                
                # EXTRA bonus for exploring toward unvisited areas far from current position
                # This encourages spreading out across the map
                if visited_squares and len(visited_squares) > 5:
                    # Calculate how far this is from already-explored areas
                    min_dist_to_visited = min(
                        abs(new_loc[0] - v[0]) + abs(new_loc[1] - v[1])
                        for v in visited_squares
                    )
                    if min_dist_to_visited >= 4:
                        score += 150.0  # Strong bonus for exploring far from known areas
                    elif min_dist_to_visited >= 3:
                        score += 80.0
                    elif min_dist_to_visited >= 2:
                        score += 40.0
                
                # AGGRESSIVE EXPLORATION BONUS: If we've laid few eggs, explore more aggressively!
                # This encourages early-game exploration to find good egg-laying spots
                if hasattr(board, 'eggs_player'):
                    eggs_laid = len(board.eggs_player) if board.eggs_player else 0
                    if eggs_laid <= 3:
                        score += 300.0  # HUGE bonus for exploring when we have few eggs
                    elif eggs_laid <= 6:
                        score += 150.0  # Good bonus for early exploration
                
                # CORNER EXPLORATION BONUS: Strongly encourage moving toward accessible corners
                # These corners give 3x egg value, so they're extremely valuable targets!
                # BUT: Don't target corners that already have eggs!
                accessible_corners = self._get_accessible_corners(board)
                if accessible_corners:
                    # Filter out corners that already have eggs
                    corners_without_eggs = [c for c in accessible_corners 
                                           if not (hasattr(board, 'eggs_player') and board.eggs_player and c in board.eggs_player)]
                    
                    if corners_without_eggs:
                        min_dist_to_accessible_corner = min(
                            abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1])
                            for corner in corners_without_eggs
                        )
                        # Bonus for moving toward accessible corners WITHOUT eggs (the right corners for this chicken)
                        if min_dist_to_accessible_corner <= 3:
                            score += 200.0  # Strong bonus for getting close to accessible corner
                        elif min_dist_to_accessible_corner <= 5:
                            score += 100.0  # Good bonus for moderate distance
                        elif min_dist_to_accessible_corner <= 7:
                            score += 50.0  # Small bonus for far distance
                    
                    # PENALTY: If moving toward a corner that already has an egg
                    if hasattr(board, 'eggs_player') and board.eggs_player:
                        for corner in accessible_corners:
                            if corner in board.eggs_player:
                                dist_to_corner_with_egg = abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1])
                                if dist_to_corner_with_egg <= 2:
                                    score -= 200.0  # Penalty for getting close to corner with egg
                                elif dist_to_corner_with_egg == 0:
                                    score -= 500.0  # Strong penalty for being at corner with egg
            else:
                # Even for visited squares, give small bonus if it helps reach new areas
                # This prevents getting stuck when all nearby squares are visited
                score += 20.0  # Small bonus to keep moving

            # However, plain moves should be less valuable than egg moves
            # This is already the case since egg moves get +5000 base score

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
        # BUT: Don't let fear of trapdoors prevent egg-laying and exploration!
        if trapdoor_tracker:
            danger = trapdoor_tracker.get_danger_score(new_loc)
            # Check if this is a known trapdoor
            is_known_trapdoor = new_loc in trapdoor_tracker.known_trapdoors
            
            if is_known_trapdoor:
                # ABSOLUTE penalty for known trapdoors - never go there!
                # This must be higher than any possible benefit (eggs, etc.)
                score -= 1000000.0
            else:
                # REDUCED penalties to encourage exploration and egg-laying
                # EGG MOVES: Much lighter penalties - eggs are worth calculated risks!
                if move_type == MoveType.EGG:
                    # For egg moves, only penalize high-probability trapdoors
                    # Low probabilities are acceptable risks for egg-laying
                    if danger > 0.3:  # 30% or more - too risky even for eggs
                        score -= danger * 50000.0
                    elif danger > 0.15:  # 15-30% - significant risk
                        score -= danger * 20000.0
                    elif danger > 0.05:  # 5-15% - moderate risk, but eggs are valuable
                        score -= danger * 5000.0
                    # Below 5%: Very light penalty - eggs are worth it!
                    elif danger > 0.01:
                        score -= danger * 1000.0
                    else:
                        score -= danger * 200.0  # Minimal penalty for very low risk
                else:
                    # PLAIN/TURD moves: Moderate penalties (but still reduced from before)
                    if danger > 0.2:  # 20% or more probability
                        score -= danger * 30000.0  # Reduced from 200000
                    elif danger > 0.1:  # 10-20% probability
                        score -= danger * 15000.0  # Reduced from 100000
                    elif danger > 0.05:  # 5-10% probability
                        score -= danger * 5000.0  # Reduced from 50000
                    elif danger > 0.01:  # 1-5% probability
                        score -= danger * 2000.0  # Reduced from 20000
                    else:
                        score -= danger * 500.0  # Reduced from 10000
            
            # 3.5. AVOID STAYING AROUND KNOWN TRAPDOORS - explore outwards!
            # Once we know where trapdoors are, move away from them
            # IMPORTANT: Egg moves get MUCH lighter penalties - we need to lay eggs!
            # Add pseudo-randomness based on location hash to break diamond patterns
            # This makes different escape directions more attractive without true randomness
            if trapdoor_tracker.known_trapdoors:
                min_dist_to_trapdoor = min(
                    abs(new_loc[0] - trap[0]) + abs(new_loc[1] - trap[1])
                    for trap in trapdoor_tracker.known_trapdoors
                )

                # Generate pseudo-random value from location hash (deterministic but varied)
                # This breaks diamond patterns by making different escape directions more attractive
                location_hash = hash(new_loc) % 1000
                location_random = 0.8 + (location_hash % 40) / 100.0  # 0.8 to 1.2 (20% variation)
                escape_random = 0.9 + (location_hash % 40) / 100.0  # 0.9 to 1.3 (30% variation)
                exploration_random = 1.0 + (location_hash % 50) / 100.0  # 1.0 to 1.5 (50% variation)

                # EGG MOVES: Only penalize if VERY close (adjacent) - eggs are the goal!
                # PLAIN MOVES: LIGHT penalties for being near trapdoors (reduced to prevent loops!)
                if move_type == MoveType.EGG:
                    # For eggs, only penalize if immediately adjacent to trapdoor
                    if min_dist_to_trapdoor <= 1:
                        score -= 100.0  # Very light penalty - eggs are ALWAYS worth it!
                    # No other proximity penalties for egg moves - we need eggs!
                elif move_type == MoveType.PLAIN:
                    # PLAIN moves: REDUCED penalties (was causing too much oscillation)
                    if min_dist_to_trapdoor <= 1:
                        score -= 400.0  # Moderate penalty (was 2000!)
                        # Add small pseudo-random variation to penalty to break patterns
                        penalty_variation = (location_hash % 100)
                        score -= penalty_variation * location_random
                    elif min_dist_to_trapdoor <= 2:
                        score -= 150.0  # Light penalty (was 800!)
                        # Add small pseudo-random variation
                        penalty_variation = (location_hash % 80)
                        score -= penalty_variation * location_random
                    # REMOVED distance 3 penalty - too restrictive!

                # BONUS for moving far away from known trapdoors (encourages exploration)
                # Apply to ALL move types to encourage spreading out
                # Add pseudo-randomness to make different escape directions more attractive
                if min_dist_to_trapdoor >= 5:
                    base_bonus = 400.0  # DOUBLED - strong bonus for getting far from trapdoors
                    score += base_bonus * escape_random
                elif min_dist_to_trapdoor >= 4:
                    base_bonus = 250.0  # INCREASED - good bonus
                    score += base_bonus * escape_random
                elif min_dist_to_trapdoor >= 3:
                    base_bonus = 120.0  # NEW - reward even moderate distance
                    score += base_bonus * escape_random

                # DIRECTIONAL ESCAPE: Reward moves that INCREASE distance from trapdoor
                # REDUCED weight to prevent overshoot cycles
                if my_loc is not None and min_dist_to_trapdoor <= 4:
                    # Find the closest trapdoor
                    closest_trap = min(
                        trapdoor_tracker.known_trapdoors,
                        key=lambda trap: abs(my_loc[0] - trap[0]) + abs(my_loc[1] - trap[1])
                    )

                    # Calculate distance from current location to closest trap
                    current_dist_to_trap = abs(my_loc[0] - closest_trap[0]) + abs(my_loc[1] - closest_trap[1])

                    # Calculate distance from new location to closest trap
                    new_dist_to_trap = abs(new_loc[0] - closest_trap[0]) + abs(new_loc[1] - closest_trap[1])

                    # REDUCED: Reward moves that increase distance (moving away)
                    # Reduced from 300 to 100 to prevent overshoot cycles
                    if new_dist_to_trap > current_dist_to_trap:
                        distance_increase = new_dist_to_trap - current_dist_to_trap
                        score += distance_increase * 100.0  # REDUCED from 300 - prevents overshoot
                    # REDUCED: Penalize moves that decrease distance (moving toward)
                    # Reduced from 400 to 150 to be less aggressive
                    elif new_dist_to_trap < current_dist_to_trap:
                        distance_decrease = current_dist_to_trap - new_dist_to_trap
                        score -= distance_decrease * 150.0  # REDUCED from 400 - less aggressive

                # EXTRA exploration bonus when moving away from trapdoors to new areas
                # Add significant pseudo-randomness to encourage varied exploration paths
                if min_dist_to_trapdoor >= 3:
                    if visited_squares is None or new_loc not in visited_squares:
                        # Strong bonus for exploring new areas away from trapdoors
                        # Add pseudo-randomness to break diamond patterns
                        score += 200.0 * exploration_random  # INCREASED
                    else:
                        # Even if visited, still give some bonus for moving away (with variation)
                        score += 80.0 * escape_random  # INCREASED

                # ADDITIONAL: Pseudo-random exploration incentive when near trapdoors
                # This helps break diamond patterns by making some directions more attractive
                # ONLY for PLAIN moves - eggs don't need this
                if move_type == MoveType.PLAIN and min_dist_to_trapdoor <= 3:
                    # Add a pseudo-random exploration bonus that varies by location
                    # This makes the agent try different escape paths
                    exploration_bonus = (location_hash % 200) - 50  # -50 to +150
                    if visited_squares is None or new_loc not in visited_squares:
                        # Stronger pseudo-random bonus for new areas
                        exploration_bonus = 50 + (location_hash % 150)  # 50 to 200
                    score += exploration_bonus

        # 4. Positional factors
        # Moving toward center is good (more options)
        center_dist = self._distance_to_center(new_loc)
        score -= center_dist * 2.0

        # 5. Don't move into cramped positions
        if self._is_edge(new_loc):
            score -= 10.0
        
        # 6. Anti-repetition for PLAIN moves: heavily penalize tight loops
        # BUT: egg moves are exempt from most of this (eggs are the goal!)
        # ALSO: when trapped (few moves), reduce penalties to allow survival
        # CRITICAL: When oscillating, FORCE exploration by reducing ALL repetition penalties!
        if move_type == MoveType.PLAIN:
            # Check if we're trapped (few valid moves)
            my_valid_moves = len(board.get_valid_moves())
            desperation_factor = 1.0
            if is_oscillating:
                # OSCILLATING - drastically reduce ALL penalties to force exploration!
                desperation_factor = 0.01  # Almost zero penalties - must break the loop!
            elif my_valid_moves <= 3:
                desperation_factor = 0.1  # DRASTICALLY reduce penalties when desperate
            elif my_valid_moves <= 5:
                desperation_factor = 0.3  # Significantly reduce penalties when limited

            if recent_positions is not None and len(recent_positions) > 0:
                # Check if this location was visited recently (last 8 moves)
                if new_loc in recent_positions:
                    # Apply penalties ONLY if not oscillating
                    if not is_oscillating:
                        # Strong penalty for recently visited squares (prevents tight loops)
                        recent_index = recent_positions.index(new_loc)
                        # More recent = higher penalty
                        recency_penalty = (len(recent_positions) - recent_index) * 200.0
                        score -= (1000.0 + recency_penalty) * desperation_factor

                        # Extra penalty if this creates a loop (going back to same square multiple times)
                        visit_count = recent_positions.count(new_loc)
                        if visit_count > 1:
                            score -= visit_count * 600.0 * desperation_factor

                # CRITICAL: Add STRONG tie-breaking to prevent oscillation
                # Use multiple factors to ensure different moves have different scores
                if my_valid_moves <= 6 or is_oscillating:
                    # Use location hash to add variation (breaks oscillation)
                    location_hash = (new_loc[0] * 1000 + new_loc[1]) % 1000
                    score += location_hash * 5.0  # INCREASED tiebreaker bonus (0-5000)

                    # STRONG preference for moves away from recent positions
                    if len(recent_positions) >= 2:
                        # Calculate average of recent positions
                        avg_recent_x = sum(pos[0] for pos in recent_positions[-4:]) / min(4, len(recent_positions))
                        avg_recent_y = sum(pos[1] for pos in recent_positions[-4:]) / min(4, len(recent_positions))
                        dist_from_recent_center = abs(new_loc[0] - avg_recent_x) + abs(new_loc[1] - avg_recent_y)
                        score += dist_from_recent_center * 500.0  # HUGE reward for escaping the area

                    # Additional: prefer accessible corners/edges when stuck (forces movement)
                    # CRITICAL: Only incentivize corners where this chicken can lay eggs (parity check)
                    if self._is_corner(new_loc):
                        if board.chicken_player.can_lay_egg(new_loc):
                            score += 500.0  # INCREASED: Accessible corner - strong bonus!
                        else:
                            score += 50.0  # Inaccessible corner - minimal bonus
                    elif self._is_edge(new_loc):
                        score += 150.0

            # Encourage exploration with plain moves (not eggs - eggs can go anywhere)
            if visited_squares is not None:
                if new_loc in visited_squares:
                    # REDUCED penalties for revisiting - sometimes necessary for exploration
                    # Apply penalties ONLY if not oscillating
                    if not is_oscillating:
                        # Lighter penalty for revisiting squares (reduced from 400)
                        score -= 150.0 * desperation_factor  # Reduced from 400

                        # Count how many times we've visited this square
                        visit_count = sum(1 for pos in (recent_positions or []) if pos == new_loc)
                        if visit_count > 0:
                            # Lighter penalty for multiple visits (reduced from 250)
                            score -= visit_count * visit_count * 100.0 * desperation_factor  # Reduced from 250
                else:
                    # STRONGER bonus for exploring new squares with plain moves
                    score += 300.0  # INCREASED from 150 - major incentive to explore!

                    # Extra bonus for exploring different regions of the map
                    region_bonus = self._get_region_exploration_bonus(new_loc, visited_squares)
                    score += region_bonus * 2.0  # Double the region bonus

                    # STRONGER bonus for moving away from recently visited areas
                    if recent_positions is not None and len(recent_positions) > 0:
                        min_dist_to_recent = min(
                            abs(new_loc[0] - pos[0]) + abs(new_loc[1] - pos[1])
                            for pos in recent_positions
                        )
                        # INCREASED bonuses for being far from recently visited squares
                        if min_dist_to_recent >= 4:
                            score += 100.0  # INCREASED from 40
                        elif min_dist_to_recent >= 3:
                            score += 50.0  # INCREASED from 20
                        elif min_dist_to_recent >= 2:
                            score += 25.0  # NEW - reward moderate distance
        
        # 7. STRONGLY encourage exploration toward accessible corners (3x egg value!)
        # CRITICAL: Only corners where this chicken can lay eggs (parity match)
        # Each chicken has exactly 2 accessible corners out of 4 total corners
        accessible_corners = self._get_accessible_corners(board)
        if accessible_corners:
            # CRITICAL: Check if any accessible corners already have eggs laid
            # If so, penalize going to those corners (can't lay another egg there!)
            if hasattr(board, 'eggs_player') and board.eggs_player:
                for corner in accessible_corners:
                    if corner in board.eggs_player:
                        # This corner already has an egg - penalize going there
                        if new_loc == corner:
                            # At the corner with egg - strong penalty
                            if move_type == MoveType.PLAIN:
                                score -= 800.0  # Strong penalty - can't lay another egg here!
                            elif move_type == MoveType.EGG:
                                score -= 100000.0  # Can't lay egg where one exists (already handled, but extra safety)
                            else:
                                score -= 600.0  # Penalty for turd moves too
                        elif abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1]) == 1:
                            # Adjacent to corner with egg - moderate penalty
                            if move_type == MoveType.PLAIN:
                                score -= 300.0  # Don't hang around corners with eggs
                            else:
                                score -= 150.0
            
            # Find closest accessible corner (that doesn't have an egg)
            corners_without_eggs = [c for c in accessible_corners 
                                   if not (hasattr(board, 'eggs_player') and board.eggs_player and c in board.eggs_player)]
            
            if corners_without_eggs:
                min_dist_to_corner = min(
                    abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1])
                    for corner in corners_without_eggs
                )
                
                # STRONGER bonuses for moving toward accessible corners WITHOUT eggs
                # These corners give 3x egg value, so they're extremely valuable!
                if visited_squares is None or new_loc not in visited_squares:
                    # INCREASED: Closer to accessible corner = much better (max distance is ~14 on 8x8 board)
                    corner_bonus = max(0, (14 - min_dist_to_corner) * 10.0)  # Doubled from 5.0
                    score += corner_bonus

                    # INCREASED: Extra bonus if we're very close to an accessible corner
                    if min_dist_to_corner <= 2:
                        score += 100.0  # INCREASED from 30 - very close to valuable corner!
                    elif min_dist_to_corner <= 4:
                        score += 50.0  # INCREASED from 15
                    elif min_dist_to_corner <= 6:
                        score += 25.0  # NEW: Bonus for moderate distance
                
                # EXTRA BONUS: If this move is actually AT an accessible corner WITHOUT egg, huge bonus!
                # This applies to both egg moves (can lay) and plain moves (getting ready to lay)
                if new_loc in corners_without_eggs:
                    if move_type == MoveType.EGG:
                        # Already handled above, but add extra for being at corner
                        score += 200.0  # Extra bonus for egg move at accessible corner
                    elif move_type == MoveType.PLAIN:
                        # Plain move to accessible corner - preparing to lay egg next turn
                        score += 150.0  # Strong bonus for positioning at accessible corner

        # 8. ESCAPE FROM EGG CLUSTERS: If we're near our own eggs, explore away!
        # This prevents circling around eggs we already laid
        # REDUCED penalties to prevent oscillation
        if hasattr(board, 'eggs_player') and board.eggs_player and move_type == MoveType.PLAIN:
            min_dist_to_my_egg = min(
                abs(new_loc[0] - egg[0]) + abs(new_loc[1] - egg[1])
                for egg in board.eggs_player
            )

            # Bonus for moving AWAY from our own eggs (with plain moves)
            if min_dist_to_my_egg >= 5:
                score += 200.0  # Great! Far from our eggs, exploring new territory
            elif min_dist_to_my_egg >= 4:
                score += 120.0  # Good distance
            elif min_dist_to_my_egg >= 3:
                score += 60.0   # Moderate distance
            elif min_dist_to_my_egg <= 1:
                # Very close to our own egg - light penalty only
                score -= 50.0  # Light penalty (was 200!)

        # 9. ESCAPE FROM OWN TURDS: Don't circle around turds we placed!
        # Plain moves should explore away, not revisit turd locations
        # REDUCED penalties to prevent oscillation
        if hasattr(board, 'turds_player') and board.turds_player and move_type == MoveType.PLAIN:
            min_dist_to_my_turd = min(
                abs(new_loc[0] - turd[0]) + abs(new_loc[1] - turd[1])
                for turd in board.turds_player
            )

            # Bonus for moving AWAY from our own turds
            if min_dist_to_my_turd >= 5:
                score += 250.0  # Great! Far from our turds
            elif min_dist_to_my_turd >= 4:
                score += 150.0  # Good distance
            elif min_dist_to_my_turd >= 3:
                score += 80.0  # Moderate distance
            elif min_dist_to_my_turd <= 2:
                # Close to our own turd - light penalty
                score -= 100.0  # Light penalty (was 350!)

            # Adjacent to our own turd squares - moderate penalty
            if new_loc in board.turds_player:
                score -= 200.0  # Moderate penalty (was 500!)

        # 10. SMART LOOP-BREAKING: Force outward movement when in a loop
        # This prevents the "clear all memory" nuclear option
        if force_outward_movement and loop_center is not None and my_loc is not None:
            # Calculate distance from current location to loop center
            current_dist_to_center = abs(my_loc[0] - loop_center[0]) + abs(my_loc[1] - loop_center[1])
            
            # Calculate distance from new location to loop center
            new_dist_to_center = abs(new_loc[0] - loop_center[0]) + abs(new_loc[1] - loop_center[1])
            
            # STRONG bonus for moving AWAY from loop center (breaking the loop)
            if new_dist_to_center > current_dist_to_center:
                distance_increase = new_dist_to_center - current_dist_to_center
                score += distance_increase * 500.0  # HUGE bonus for breaking out of loop
            # STRONG penalty for moving TOWARD loop center (staying in loop)
            elif new_dist_to_center < current_dist_to_center:
                distance_decrease = current_dist_to_center - new_dist_to_center
                score -= distance_decrease * 800.0  # HUGE penalty for staying in loop
            
            # EXTRA: Prefer moves to completely new areas when breaking loops
            if visited_squares is None or new_loc not in visited_squares:
                score += 400.0  # Strong bonus for exploring new areas when breaking loops

        # 11. PARITY-CORRECT CORNER TARGETING: Strongly incentivize accessible corners
        # This helps break loops by giving clear targets
        # BUT: Don't target corners that already have eggs!
        if is_oscillating or force_outward_movement:
            accessible_corners = self._get_accessible_corners(board)
            if accessible_corners:
                # Filter out corners that already have eggs
                corners_without_eggs = [c for c in accessible_corners 
                                       if not (hasattr(board, 'eggs_player') and board.eggs_player and c in board.eggs_player)]
                
                if corners_without_eggs:
                    min_dist_to_accessible_corner = min(
                        abs(new_loc[0] - corner[0]) + abs(new_loc[1] - corner[1])
                        for corner in corners_without_eggs
                    )
                    
                    # HUGE bonus for moving toward accessible corners WITHOUT eggs when in a loop
                    # Corners are clear targets that break loops
                    if min_dist_to_accessible_corner <= 4:
                        score += 600.0  # Very strong bonus for getting close to accessible corner
                    elif min_dist_to_accessible_corner <= 6:
                        score += 300.0  # Good bonus for moderate distance
                    
                    # EXTRA: If we're actually at an accessible corner WITHOUT egg, massive bonus
                    if new_loc in corners_without_eggs:
                        if move_type == MoveType.EGG:
                            score += 1000.0  # HUGE bonus for egg at accessible corner
                        elif move_type == MoveType.PLAIN:
                            score += 500.0  # Strong bonus for positioning at accessible corner
                else:
                    # All accessible corners have eggs - penalize going to them
                    if new_loc in accessible_corners:
                        if move_type == MoveType.PLAIN:
                            score -= 1000.0  # Strong penalty - all corners done, don't loop here!
                        else:
                            score -= 500.0

        # 12. REDUCE TRAPDOOR ESCAPE WEIGHT when in loop
        # Trapdoor escape logic can cause overshoot cycles
        # This is already handled above by reducing the directional escape weights

        return score

    def evaluate_position(
        self,
        board: board_module.Board,
        nn_evaluator=None,
        feature_extractor=None,
        trapdoor_tracker=None
    ) -> float:
        """
        Comprehensive position evaluation.

        Uses neural network if available, otherwise falls back to heuristics.
        Returns a score where positive is good for current player.
        """
        # Try neural network first (if available and loaded)
        if nn_evaluator is not None and feature_extractor is not None:
            try:
                import torch

                # Extract features
                features = feature_extractor.extract_features(board, trapdoor_tracker)

                # Convert to tensor
                features_tensor = torch.FloatTensor(features).unsqueeze(0)  # Add batch dimension

                # Get NN prediction
                with torch.no_grad():
                    nn_output = nn_evaluator(features_tensor)
                    nn_score_raw = nn_output.item()

                # NN was trained with normalized scores (roughly [-2, +2])
                # Convert back to heuristic scale (multiply by ~1500)
                nn_score = nn_score_raw * 1500.0

                # Blend with heuristics (80% NN, 20% heuristic for stability)
                heuristic_score = self._get_heuristic_score(board)
                blended_score = 0.8 * nn_score + 0.2 * heuristic_score

                return blended_score

            except Exception as e:
                # NN failed, fall back to heuristics
                pass

        # Fallback: pure heuristic evaluation
        return self._get_heuristic_score(board)

    def _get_heuristic_score(self, board: board_module.Board) -> float:
        """Get heuristic-based score (fallback when NN not available)"""
        # Material advantage (most important)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = (my_eggs - enemy_eggs) * 300.0  # Use improved 300 per egg

        # Mobility advantage - CRITICAL for avoiding traps!
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()

        # If enemy has NO moves, they lose and we get 5 eggs (1500 points)!
        if enemy_moves == 0:
            mobility_score = 2000.0  # Winning position!
        # If WE have no moves, we lose and enemy gets 5 eggs
        elif my_moves == 0:
            mobility_score = -2000.0  # Losing position!
        else:
            # Normal mobility advantage - very important!
            # Each move difference is worth ~50 points (not 5!)
            # Having more moves = safer from traps + more options
            mobility_score = (my_moves - enemy_moves) * 50.0

            # Extra penalty for low mobility (danger of getting trapped)
            if my_moves <= 2:
                mobility_score -= 200.0  # Very dangerous!
            elif my_moves <= 3:
                mobility_score -= 100.0  # Risky

            # Bonus for reducing enemy mobility (trying to trap them)
            if enemy_moves <= 2:
                mobility_score += 200.0  # We're trapping them!
            elif enemy_moves <= 3:
                mobility_score += 100.0

        # Positional factors
        positional_score = self._evaluate_position_quality(board)

        # Combine scores
        total = egg_diff + mobility_score + positional_score

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
    
    def _get_quadrant(self, loc: Tuple[int, int]) -> str:
        """Get which quadrant a location is in (for spreading strategy)"""
        mid = self.map_size / 2.0
        x, y = loc

        if x < mid and y < mid:
            return 'NW'
        elif x < mid and y >= mid:
            return 'NE'
        elif x >= mid and y < mid:
            return 'SW'
        else:
            return 'SE'

    def _count_quadrant_visits(self, loc: Tuple[int, int], visited_squares: set) -> dict:
        """Count how many squares have been visited in each quadrant"""
        quadrant_counts = {'NW': 0, 'NE': 0, 'SW': 0, 'SE': 0}

        for visited in visited_squares:
            quadrant = self._get_quadrant(visited)
            quadrant_counts[quadrant] += 1

        return quadrant_counts

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

