from typing import Tuple, List, Set, Deque
from collections import deque
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module

class MoveEvaluator:
    def __init__(self, map_size: int = 8):
        self.map_size = map_size

    def evaluate_position(
        self,
        board: board_module.Board,
        trapdoor_tracker=None,
        visited_squares=None,
    ) -> float:
        """
        Calculates the score of the current board state.
        Higher is better for the active player.
        """
        if visited_squares is None:
            visited_squares = set()

        score = 0.0
        
        # --- 1. SAFETY (Highest Priority) ---
        # Immediate trapdoor death check
        my_loc = board.chicken_player.get_location()
        
        if trapdoor_tracker:
            # Absolute death (known trapdoor)
            if my_loc in trapdoor_tracker.known_trapdoors:
                return -1_000_000.0 # Instant loss state
            
            # Probabilistic danger
            danger = trapdoor_tracker.get_danger_score(my_loc)
            # THRESHOLD UPDATED: 0.25 to align with Agent's risk tolerance
            if danger > 0.25:
                # Moderate penalty - balance between aggression and safety
                score -= danger * 5000.0

        # --- 2. MATERIAL (Current Eggs on Board) ---
        # Use get_eggs_laid() which automatically accounts for corner multipliers (3x)
        # This is Patrick's approach - simpler and more accurate
        turns_left = board.turns_left_player
        is_endgame = turns_left < 10

        my_eggs_count = board.chicken_player.get_eggs_laid()
        enemy_eggs_count = board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs_count - enemy_eggs_count

        # Patrick's weight: 150.0, with 2x multiplier in endgame
        # We use similar aggressive weighting
        BASE_W_EGG = 150.0
        w_egg = BASE_W_EGG * (2.0 if is_endgame else 1.0)

        score += w_egg * egg_diff

        # --- 3. REACHABILITY ANALYSIS (BFS Depth 20) ---
        # Calculates "Potential" (future eggs) and "Territory" (movement freedom)
        my_potential, my_territory = self._analyze_reachability(board, is_me=True, trapdoor_tracker=trapdoor_tracker)
        en_potential, en_territory = self._analyze_reachability(board, is_me=False, trapdoor_tracker=None) # Don't assume enemy knows trapdoors

        # Patrick's weights: POTENTIAL=25, SPACE=5
        # We match his philosophy: focus on potential > territory
        BASE_W_POTENTIAL = 25.0
        BASE_W_SPACE = 5.0

        score += (my_potential - en_potential) * BASE_W_POTENTIAL
        score += (my_territory - en_territory) * BASE_W_SPACE

        # --- 4. CORNER POSITIONING ---
        # Bonus just for standing in a corner (if we can lay there)
        if self._is_corner(my_loc) and board.chicken_player.can_lay_egg(my_loc):
             score += 50.0

        # --- 5. DISTANCE BONUS (Prevent idling with plain moves) ---
        # If we have egg potential, reward being closer to valid egg spots
        # This prevents the agent from wandering aimlessly in controlled territory
        if my_potential > 0:
            dist = self._dist_to_nearest_valid_egg_spot(board)
            # Closer is better. Max typical distance is ~14 on 8x8 board
            # Use (max_dist - actual_dist) * weight formula
            max_dist = self.map_size * 2
            dist_bonus = (max_dist - dist) * 3.0
            score += dist_bonus

        # --- 6. TURD PENALTY REMOVED ---
        # Patrick doesn't penalize turd usage in evaluation - let minimax decide naturally
        # Turd penalty remains in quick_evaluate_move() for move ordering only

        # --- 7. EXPLORATION BONUS (Endgame) ---
        # In endgame, reward visiting new squares to break oscillation
        if is_endgame and my_loc not in visited_squares:
            score += 100.0  # Bonus for exploring new territory

        return score

    def quick_evaluate_move(
        self,
        move: Tuple[Direction, MoveType],
        board: board_module.Board,
        trapdoor_tracker=None,
        visited_squares=None,
    ) -> float:
        """
        Used for move ordering. Fast heuristic without BFS.
        """
        if visited_squares is None:
            visited_squares = set()

        direction, move_type = move
        my_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(my_loc, direction)
        score = 0.0

        # Check if we're in endgame
        turns_left = board.turns_left_player
        is_endgame = turns_left < 10

        # FIX: Check if we can ACTUALLY lay an egg here before rewarding it
        if move_type == MoveType.EGG:
            if board.chicken_player.can_lay_egg(my_loc):
                # Base egg bonus
                egg_bonus = 1000.0
                # In endgame, MASSIVELY prioritize egg laying to break oscillation
                if is_endgame:
                    egg_bonus *= 3.0  # 3x multiplier in final 10 turns
                score += egg_bonus

                if self._is_corner(new_loc):
                    corner_bonus = 2000.0
                    if is_endgame:
                        corner_bonus *= 2.0  # Extra corner priority in endgame
                    score += corner_bonus
            else:
                # If we try to lay an egg where we can't, it's just a plain move.
                # Penalize slightly to stop wasting search time on invalid move types.
                score -= 50.0 
        
        elif move_type == MoveType.PLAIN:
            # In endgame, reward moving to new squares to break oscillation
            if is_endgame and new_loc not in visited_squares:
                score += 200.0  # Exploration bonus for PLAIN moves in endgame

        elif move_type == MoveType.TURD:
            # Turds are a limited resource - always penalize in move ordering
            # Let minimax decide if the strategic value outweighs the penalty
            my_loc = board.chicken_player.get_location()
            enemy_loc = board.chicken_enemy.get_location()
            manhattan_dist = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])

            if manhattan_dist <= 3:
                # Close range - lighter penalty but still negative
                score -= 50.0
            else:
                # Far from enemy - heavy penalty
                score -= 200.0 

        # Avoid stepping into high danger for no reason
        if trapdoor_tracker:
            if new_loc in trapdoor_tracker.known_trapdoors:
                score -= 50000.0
            else:
                danger = trapdoor_tracker.get_danger_score(new_loc)
                # THRESHOLD UPDATED: 0.25 to align with Agent's risk tolerance
                if danger > 0.25:
                    score -= danger * 5000.0
        
        return score

    def _analyze_reachability(self, board: board_module.Board, is_me: bool, trapdoor_tracker) -> Tuple[float, float]:
        """
        Runs a BFS to depth 20.
        Returns:
            potential: Weighted count of EMPTY squares with matching parity (actual egg opportunities).
            territory: Weighted count of total squares reachable (mobility/safety).

        KEY IMPROVEMENT: Uses parity check directly instead of can_lay_egg() for efficiency.
        Only counts TRULY EMPTY squares as potential (not ones with our eggs already).
        """
        limit = 20

        if is_me:
            start_node = board.chicken_player.get_location()
            chicken_obj = board.chicken_player
            my_items = self._get_items_from_board(board, board.chicken_player)
            blocker_items = self._get_items_from_board(board, board.chicken_enemy)
            # Get parity from chicken object (0 = even, 1 = odd)
            my_parity = chicken_obj.even_chicken
        else:
            start_node = board.chicken_enemy.get_location()
            chicken_obj = board.chicken_enemy
            my_items = self._get_items_from_board(board, board.chicken_enemy)
            blocker_items = self._get_items_from_board(board, board.chicken_player)
            my_parity = chicken_obj.even_chicken

        # Build lethal zones (3x3 around enemy turds) - Patrick's improvement
        lethal_zones = set()
        enemy_turds = blocker_items - (board.eggs_enemy if is_me else board.eggs_player)
        for tx, ty in enemy_turds:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lx, ly = tx + dx, ty + dy
                    if 0 <= lx < self.map_size and 0 <= ly < self.map_size:
                        lethal_zones.add((lx, ly))

        # BFS Structures
        queue = deque([(start_node, 0)])
        visited = {start_node}

        potential_score = 0.0
        territory_score = 0.0

        while queue:
            curr, dist = queue.popleft()

            if dist >= limit:
                continue

            # Decay factor: rewards reaching spots SOONER
            decay = 1.0 ** dist

            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                nxt = loc_after_direction(curr, direction)

                if nxt in visited:
                    continue

                # Validity Checks
                if not board.is_valid_cell(nxt):
                    continue

                is_blocked = False

                # Check against blocker items (enemy eggs/turds)
                if nxt in blocker_items:
                    is_blocked = True

                # Check lethal zones (3x3 around enemy turds)
                elif nxt in lethal_zones:
                    is_blocked = True

                # Check generic board blocks
                elif board.is_cell_blocked(nxt):
                    # Only passable if it's MY item
                    if nxt not in my_items:
                        is_blocked = True

                # Trapdoor avoidance (only for me)
                if is_me and trapdoor_tracker and nxt in trapdoor_tracker.known_trapdoors:
                    is_blocked = True

                if not is_blocked:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))

                    # 1. TERRITORY SCORE (all reachable squares)
                    territory_score += 1.0 * decay

                    # 2. POTENTIAL SCORE (EMPTY squares with correct parity)
                    # This is the KEY fix: only count truly empty squares we can score on
                    nx, ny = nxt
                    square_parity = (nx + ny) % 2

                    if square_parity == my_parity:
                        # Must be EMPTY (not in my items, not in blocker items)
                        if nxt not in my_items and nxt not in blocker_items:
                            val = 1.0
                            if self._is_corner(nxt):
                                val = 3.0  # Corner eggs are worth 3x
                            potential_score += val * decay

        return potential_score, territory_score

    def _get_items_from_board(self, board: board_module.Board, chicken) -> Set[Tuple[int, int]]:
        """
        CRITICAL FIX: Read items directly from the board object, not chicken attributes!

        The board has TWO separate storage locations:
        1. board.eggs_player / board.eggs_enemy / board.turds_player / board.turds_enemy (SOURCE OF TRUTH)
        2. chicken.dropped_eggs / chicken.dropped_turds (may be out of sync!)

        When forecast_move() updates the board, it modifies board.eggs_player/etc via apply_move(),
        but chicken object attributes may not be updated synchronously.

        Solution: Determine if this is player or enemy, then read directly from board's sets.
        """
        # Determine if this chicken is the player or enemy by comparing references
        if chicken is board.chicken_player:
            # This is the active player
            eggs = board.eggs_player
            turds = board.turds_player
        else:
            # This is the enemy
            eggs = board.eggs_enemy
            turds = board.turds_enemy

        # Return the union of eggs and turds from the board's actual state
        return eggs.union(turds)

    def _get_chicken_items(self, chicken):
        """Helper to safely get sets of items from chicken object."""
        # Use the robust egg getter
        eggs = self._get_chicken_eggs(chicken)
        turds = getattr(chicken, 'dropped_turds', getattr(chicken, 'turds', set()))
        # Return union for easy checking
        return eggs.union(turds), (eggs, turds)

    def _get_chicken_eggs(self, chicken):
        """
        Robustly attempts to find eggs.
        CRITICAL: Never return fake coordinates - they break BFS blocking logic!
        """
        # Try to get the actual set of egg locations
        eggs = getattr(chicken, 'dropped_eggs', getattr(chicken, 'eggs', None))

        if isinstance(eggs, set) and len(eggs) > 0:
            # Validate these are real coordinates, not fake ones
            try:
                sample = next(iter(eggs))
                if sample[0] >= 0 and sample[1] >= 0:
                    return eggs
            except (StopIteration, TypeError, IndexError):
                pass

        # If we can't get valid egg positions, return empty set
        # Don't use fake coordinates like (-1, i) - they break the BFS logic!
        return set()

    def _is_corner(self, loc: Tuple[int, int]) -> bool:
        x, y = loc
        return (x == 0 or x == self.map_size - 1) and (y == 0 or y == self.map_size - 1)

    def _dist_to_nearest_valid_egg_spot(self, board: board_module.Board) -> int:
        """
        BFS to find distance to nearest EMPTY square with correct parity (valid egg spot).
        Returns distance, or max distance if no spots reachable.
        Helps prevent 'plain move' idling - rewards moving closer to scoring opportunities.
        """
        start_node = board.chicken_player.get_location()
        my_parity = board.chicken_player.even_chicken

        queue = deque([(start_node, 0)])
        visited = {start_node}

        # Get all items on board
        my_items = self._get_items_from_board(board, board.chicken_player)
        enemy_items = self._get_items_from_board(board, board.chicken_enemy)

        # Build lethal zones around enemy turds
        lethal_zones = set()
        enemy_turds = board.turds_enemy
        for tx, ty in enemy_turds:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lx, ly = tx + dx, ty + dy
                    if 0 <= lx < self.map_size and 0 <= ly < self.map_size:
                        lethal_zones.add((lx, ly))

        while queue:
            curr, dist = queue.popleft()
            cx, cy = curr

            # Check if this is a valid egg spot
            square_parity = (cx + cy) % 2
            if square_parity == my_parity:
                # Must be EMPTY (not occupied by anyone)
                if curr not in my_items and curr not in enemy_items:
                    return dist

            # Explore neighbors
            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                nxt = loc_after_direction(curr, direction)

                if nxt in visited:
                    continue

                if not board.is_valid_cell(nxt):
                    continue

                # Can't path through enemy items or lethal zones
                if nxt in enemy_items or nxt in lethal_zones:
                    continue

                # Can path through our own items
                visited.add(nxt)
                queue.append((nxt, dist + 1))

        # No valid egg spot reachable - return max distance
        return self.map_size * 2
    
    def debug_evaluate_position(self, board: board_module.Board, trapdoor_tracker=None, move_name="Current") -> float:
        """
        Duplicate of evaluate_position but prints detailed scoring logs.
        """
        score = 0.0
        logs = [f"--- DEBUG EVALUATION: {move_name} ---"]

        # --- 1. SAFETY ---
        safety_score = 0.0
        my_loc = board.chicken_player.get_location()
        if trapdoor_tracker:
            if my_loc in trapdoor_tracker.known_trapdoors:
                 safety_score -= 1_000_000.0
                 logs.append("  [FATAL] Known Trapdoor!")
            else:
                danger = trapdoor_tracker.get_danger_score(my_loc)
                # THRESHOLD UPDATED: 0.25 to align with Agent's risk tolerance
                if danger > 0.25:
                    penalty = danger * 10000.0
                    safety_score -= penalty
                    logs.append(f"  [DANGER] Level: {danger:.4f} -> Penalty: -{penalty:.1f}")
        score += safety_score

        # --- 2. MATERIAL ---
        my_eggs = board.eggs_player
        enemy_eggs = board.eggs_enemy
        
        my_mat = sum(600.0 if self._is_corner(e) else 100.0 for e in my_eggs)
        en_mat = sum(600.0 if self._is_corner(e) else 100.0 for e in enemy_eggs)
        
        material_score = my_mat - en_mat
        logs.append(f"  [MATERIAL] Me: {my_mat} vs Enemy: {en_mat} -> Score: {material_score}")
        score += material_score

        # --- 3. REACHABILITY ANALYSIS ---
        my_potential, my_territory = self._analyze_reachability(board, is_me=True, trapdoor_tracker=trapdoor_tracker)
        en_potential, en_territory = self._analyze_reachability(board, is_me=False, trapdoor_tracker=None)
        
        pot_score = (my_potential - en_potential) * 50.0
        terr_score = (my_territory - en_territory) * 10.0
        
        logs.append(f"  [POTENTIAL] Me: {my_potential:.2f} vs Enemy: {en_potential:.2f} -> Score: {pot_score:.2f}")
        logs.append(f"  [TERRITORY] Me: {my_territory:.2f} vs Enemy: {en_territory:.2f} -> Score: {terr_score:.2f}")
        
        score += pot_score
        score += terr_score

        # --- 4. CORNER POSITIONING ---
        if self._is_corner(my_loc) and board.chicken_player.can_lay_egg(my_loc):
             score += 50.0
             logs.append("  [BONUS] Standing in Corner")

        # --- 5. DISTANCE BONUS ---
        if my_potential > 0:
            dist = self._dist_to_nearest_valid_egg_spot(board)
            max_dist = self.map_size * 2
            dist_bonus = (max_dist - dist) * 3.0
            logs.append(f"  [DISTANCE] To nearest egg spot: {dist} -> Bonus: +{dist_bonus:.1f}")
            score += dist_bonus

        # --- 6. TURD USAGE PENALTY ---
        turds_used = 5 - board.chicken_player.get_turds_left()
        enemy_turds_used = 5 - board.chicken_enemy.get_turds_left()
        turd_penalty = (turds_used - enemy_turds_used) * 200.0
        if turd_penalty != 0:
            logs.append(f"  [TURD PENALTY] Me: {turds_used} used, Enemy: {enemy_turds_used} used -> Penalty: -{turd_penalty:.1f}")
        score -= turd_penalty

        logs.append(f"  TOTAL SCORE: {score:.2f}")
        print("\n".join(logs))
        return score