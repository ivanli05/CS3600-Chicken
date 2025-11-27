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
    ) -> float:
        """
        Calculates the score of the current board state.
        Higher is better for the active player.
        """
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
                # Heavy penalty for risking it
                score -= danger * 10000.0

        # --- 2. MATERIAL (Current Eggs on Board) ---
        # We value eggs already laid. Corners are worth 3x in game,
        # but we weight them heavily here to encourage keeping them.
        my_eggs = board.eggs_player
        enemy_eggs = board.eggs_enemy
        
        my_material_score = 0
        for egg_loc in my_eggs:
            if self._is_corner(egg_loc):
                my_material_score += 600.0  # Huge bonus for corner eggs
            else:
                my_material_score += 100.0
        
        enemy_material_score = 0
        for egg_loc in enemy_eggs:
            if self._is_corner(egg_loc):
                enemy_material_score += 600.0
            else:
                enemy_material_score += 100.0
                
        score += (my_material_score - enemy_material_score)

        # --- 3. REACHABILITY ANALYSIS (BFS Depth 20) ---
        # Calculates "Potential" (future eggs) and "Territory" (movement freedom)
        my_potential, my_territory = self._analyze_reachability(board, is_me=True, trapdoor_tracker=trapdoor_tracker)
        en_potential, en_territory = self._analyze_reachability(board, is_me=False, trapdoor_tracker=None) # Don't assume enemy knows trapdoors

        # Potential is very important (future scoring opportunities)
        score += (my_potential - en_potential) * 50.0
        
        # Territory is a tie-breaker (avoid getting boxed in)
        score += (my_territory - en_territory) * 10.0

        # --- 4. CORNER POSITIONING ---
        # Bonus just for standing in a corner (if we can lay there)
        if self._is_corner(my_loc) and board.chicken_player.can_lay_egg(my_loc):
             score += 50.0

        return score

    def quick_evaluate_move(
        self,
        move: Tuple[Direction, MoveType],
        board: board_module.Board,
        trapdoor_tracker=None,
    ) -> float:
        """
        Used for move ordering. Fast heuristic without BFS.
        """
        direction, move_type = move
        my_loc = board.chicken_player.get_location()
        new_loc = loc_after_direction(my_loc, direction)
        score = 0.0

        # FIX: Check if we can ACTUALLY lay an egg here before rewarding it
        if move_type == MoveType.EGG:
            if board.chicken_player.can_lay_egg(my_loc):
                score += 1000.0 
                if self._is_corner(new_loc):
                    score += 2000.0 
            else:
                # If we try to lay an egg where we can't, it's just a plain move.
                # Penalize slightly to stop wasting search time on invalid move types.
                score -= 50.0 
        
        elif move_type == MoveType.TURD:
            # (Keep your existing Turd logic here)
            pass 

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
            potential: Weighted count of valid egg spots reachable.
            territory: Weighted count of total squares reachable.
        """
        limit = 20

        if is_me:
            start_node = board.chicken_player.get_location()
            chicken_obj = board.chicken_player
            # FIX: Build items by scanning the board grid, not from chicken attributes
            my_items = self._get_items_from_board(board, board.chicken_player)
            blocker_items = self._get_items_from_board(board, board.chicken_enemy)
        else:
            start_node = board.chicken_enemy.get_location()
            chicken_obj = board.chicken_enemy
            my_items = self._get_items_from_board(board, board.chicken_enemy)
            blocker_items = self._get_items_from_board(board, board.chicken_player)

        # BFS Structures
        queue = deque([(start_node, 0)])
        visited = {start_node}
        
        potential_score = 0.0
        territory_score = 0.0

        while queue:
            curr, dist = queue.popleft()
            
            if dist >= limit:
                continue

            # Decay factor: rewards reaching spots SOONER.
            decay = 1 ** dist 

            for direction in [Direction.UP, Direction.DOWN, Direction.LEFT, Direction.RIGHT]:
                nxt = loc_after_direction(curr, direction)
                
                if nxt in visited:
                    continue
                
                # Validity Checks
                if not board.is_valid_cell(nxt):
                    continue
                
                is_blocked = False

                # FIX 1: Explicitly check against the KNOWN blocker items.
                # This guarantees that if I lay a Turd, the Enemy BFS sees it as a wall.
                if nxt in blocker_items:
                    is_blocked = True
                
                # FIX 2: Also check generic board blocks (redundancy)
                elif board.is_cell_blocked(nxt):
                    # If it's blocked, it's only passable if it's strictly MY item
                    if nxt not in my_items:
                        is_blocked = True
                
                # Trapdoor avoidance (only for me)
                if is_me and trapdoor_tracker and nxt in trapdoor_tracker.known_trapdoors:
                    is_blocked = True

                if not is_blocked:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
                    
                    # 1. TERRITORY SCORE
                    territory_score += 1.0 * decay

                    # 2. POTENTIAL SCORE (Egg laying opportunities)
                    if chicken_obj.can_lay_egg(nxt):
                        # Ensure strictly empty for laying
                        if not board.is_cell_blocked(nxt) and nxt not in blocker_items:
                            val = 1.0
                            if self._is_corner(nxt):
                                val = 3.0 
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

        logs.append(f"  TOTAL SCORE: {score:.2f}")
        print("\n".join(logs))
        return score