from collections.abc import Callable
from typing import List, Tuple, Optional, Dict, Set

import math
import random
import time
import numpy as np

# Import specific game modules based on the environment
from game import *
from game.enums import Direction, MoveType, Result
from game.game_map import prob_hear, prob_feel

# Type aliases
Board = board.Board

class PlayerAgent:
    """
    Aggressive 'Harvester' Agent.
    
    Improvements:
    1. Egg Potential: Values territory based on HOW MANY eggs we can lay there.
    2. Distance Bias: Penalizes 'Plain' moves that don't get closer to a valid egg spot.
    3. Aggressive Ordering: Prioritizes eggs above almost everything.
    """

    # --- Config ---
    MAX_DEPTH = 10
    RISK_THRESHOLD_PRUNE = 0.30
    
    # --- Dynamic Weights ---
    # We increased Egg weights and added "Potential" (Future Eggs)
    BASE_W_EGG = 150.0       # High value on immediate score
    BASE_W_POTENTIAL = 25.0  # Value of an empty square we can reach & lay on
    BASE_W_SPACE = 5.0       # Lower value on raw space (safety), focus on potential
    BASE_W_TURD = 40.0
    
    def __init__(self, initial_board: Board, time_left: Callable):
        self.map_size = initial_board.game_map.MAP_SIZE
        self.timer = time_left
        
        # Determine our parity (Even or Odd) for egg laying
        # We need this to calculate "Egg Potential"
        start_loc = initial_board.chicken_player.get_location()
        # If we are Player A (White), we likely have Even parity. 
        # But let's trust the Chicken object's internal parity if we could see it.
        # Since we can't easily see .even_chicken in the Board wrapper without digging,
        # we infer it: Player A (White) usually starts on Black (Odd sum)? 
        # Wait, rules say: "Chicken A starts on a randomly chosen black square... Chicken A can lay an egg on [Even Sum]"
        # So we just check the rule: Player A = Even Sums, Player B = Odd Sums.
        # However, checking 'is_player_a' or turns is safer.
        # Board.is_as_turn tells us if it is A's turn. 
        # We will dynamically check parity in the evaluation loop.
        
        # Bayesian Tracking
        self.trapdoor_beliefs = [None, None]
        self.initial_trapdoor_prior = [None, None]
        self._init_trapdoor_beliefs()

        self.killer_moves = {} 
        self._nodes_searched = 0

    # ========================= 1. PROBABILISTIC LOGIC ========================= #
    # (Same as before, no changes needed here)

    def _init_trapdoor_beliefs(self):
        dim = self.map_size
        unnormalized = np.zeros((dim, dim), dtype=float)
        if dim > 4: unnormalized[2:dim-2, 2:dim-2] = 1.0
        if dim > 6: unnormalized[3:dim-3, 3:dim-3] = 2.0

        for parity in (0, 1):
            prior = np.zeros((dim, dim), dtype=float)
            for i in range(dim):
                for j in range(dim):
                    if (i + j) % 2 == parity:
                        prior[i, j] = unnormalized[i, j]
            total = prior.sum()
            if total > 0: prior /= total
            self.trapdoor_beliefs[parity] = prior
            self.initial_trapdoor_prior[parity] = prior.copy()

    def _update_trapdoor_beliefs(self, b: Board, sensor_data: List[Tuple[bool, bool]]):
        loc = b.chicken_player.get_location()
        lx, ly = loc
        dim = self.map_size

        for parity in (0, 1):
            did_hear, did_feel = sensor_data[parity]
            prior = self.trapdoor_beliefs[parity]
            if np.max(prior) > 0.99: continue

            posterior = np.zeros_like(prior)
            for i in range(dim):
                for j in range(dim):
                    if (i + j) % 2 != parity: continue
                    if prior[i, j] < 0.001: continue 

                    dx, dy = abs(lx - i), abs(ly - j)
                    p_h = prob_hear(dx, dy)
                    p_f = prob_feel(dx, dy)
                    
                    p_obs = (p_h if did_hear else 1-p_h) * (p_f if did_feel else 1-p_f)
                    posterior[i, j] = prior[i, j] * p_obs

            total = posterior.sum()
            if total <= 0:
                posterior = self.initial_trapdoor_prior[parity].copy()
            else:
                posterior /= total
            self.trapdoor_beliefs[parity] = posterior

        if hasattr(b, "found_trapdoors"):
            for tx, ty in b.found_trapdoors:
                p = (tx + ty) % 2
                self.trapdoor_beliefs[p] = np.zeros((dim, dim))
                self.trapdoor_beliefs[p][tx, ty] = 1.0

    def get_risk(self, pos: Tuple[int, int]) -> float:
        x, y = pos
        parity = (x + y) % 2
        return self.trapdoor_beliefs[parity][x, y]

    # ========================= 2. MAIN LOOP ========================= #

    def play(self, current_board: Board, sensor_data, time_left: Callable):
        self.timer = time_left
        self._nodes_searched = 0
        self.killer_moves = {} 
        
        self._update_trapdoor_beliefs(current_board, sensor_data)
        
        turns_rem = current_board.turns_left_player
        total_time = self.timer()
        
        # More aggressive timing to ensure we find the kill path
        if turns_rem > 35: alloc_time = 0.8
        elif turns_rem < 8: alloc_time = 0.5
        else: alloc_time = min(total_time / turns_rem * 1.5, 3.5)
        
        end_time = time.time() + alloc_time - 0.1

        best_move = (Direction.UP, MoveType.PLAIN)
        
        valid_moves = current_board.get_valid_moves()
        safe_moves = [m for m in valid_moves if not self._is_suicide(current_board, m)]
        if not safe_moves: safe_moves = valid_moves 
        if not safe_moves: return best_move 
        
        safe_moves = self._order_moves(current_board, safe_moves, 0)
        best_move = safe_moves[0]

        for depth in range(1, self.MAX_DEPTH + 1):
            if time.time() > end_time: break
            
            try:
                val, move = self._minimax(current_board, depth, -math.inf, math.inf, True, end_time)
                if move:
                    best_move = move
            except TimeoutError:
                break
        
        return best_move

    def _is_suicide(self, b: Board, move) -> bool:
        d, mt = move
        loc = b.chicken_player.get_location()
        dx, dy = 0, 0
        if d == Direction.UP: dx, dy = 0, -1
        elif d == Direction.DOWN: dx, dy = 0, 1
        elif d == Direction.LEFT: dx, dy = -1, 0
        elif d == Direction.RIGHT: dx, dy = 1, 0
        
        nx, ny = loc[0] + dx, loc[1] + dy
        
        if 0 <= nx < self.map_size and 0 <= ny < self.map_size:
            if self.get_risk((nx, ny)) > self.RISK_THRESHOLD_PRUNE:
                return True
        return False

    # ========================= 3. SEARCH ENGINE ========================= #

    def _minimax(self, b: Board, depth: int, alpha: float, beta: float, 
                 is_max: bool, end_time: float) -> Tuple[float, Optional[Tuple]]:
        
        self._nodes_searched += 1
        if self._nodes_searched % 100 == 0:
            if time.time() > end_time: raise TimeoutError

        if depth == 0 or b.is_game_over():
            return self._evaluate(b), None

        if is_max:
            max_val = -math.inf
            best_move = None
            
            moves = b.get_valid_moves()
            moves = [m for m in moves if not self._is_suicide(b, m)]
            if not moves: return -10000.0, None 
            
            moves = self._order_moves(b, moves, depth)

            for move in moves:
                child = b.forecast_move(move[0], move[1])
                if not child: continue
                
                val, _ = self._minimax(child, depth-1, alpha, beta, False, end_time)
                
                if val > max_val:
                    max_val = val
                    best_move = move
                
                alpha = max(alpha, val)
                if beta <= alpha:
                    self.killer_moves[depth] = move
                    break
            return max_val, best_move

        else: 
            min_val = math.inf
            moves = b.get_valid_moves(enemy=True)
            if not moves: return 10000.0, None 

            for move in moves:
                d, mt = move
                child = b.get_copy()
                child.reverse_perspective()
                if child.apply_move(d, mt, check_ok=True):
                    child.reverse_perspective()
                    val, _ = self._minimax(child, depth-1, alpha, beta, True, end_time)
                    
                    if val < min_val: min_val = val
                    beta = min(beta, val)
                    if beta <= alpha: break
            return min_val, None

    def _order_moves(self, b: Board, moves: List, depth: int) -> List:
        killer = self.killer_moves.get(depth)
        
        def score_move(m):
            d, mt = m
            score = 0
            if m == killer: score += 1000
            if mt == MoveType.EGG: score += 100  # Aggressively prefer eggs
            if mt == MoveType.TURD:
                # Only value turds if they are aggressive (near enemy)
                p_loc = b.chicken_player.get_location()
                e_loc = b.chicken_enemy.get_location()
                dist = abs(p_loc[0]-e_loc[0]) + abs(p_loc[1]-e_loc[1])
                if dist <= 3: score += 60 
                else: score -= 10 
            return score
            
        moves.sort(key=score_move, reverse=True)
        return moves

    # ========================= 4. IMPROVED EVALUATION ========================= #

    def _evaluate(self, b: Board) -> float:
        if b.is_game_over():
            winner = b.get_winner()
            if winner == Result.PLAYER: return 100000.0
            if winner == Result.ENEMY: return -100000.0
            return 0.0

        p_chk = b.chicken_player
        e_chk = b.chicken_enemy
        
        turns_rem = b.turns_left_player
        is_endgame = turns_rem < 10
        
        # 1. ACTUAL SCORE (Weighted highest)
        # We increase the weight of eggs to prevent "idling" with high territory
        egg_diff = p_chk.get_eggs_laid() - e_chk.get_eggs_laid()
        w_egg = self.BASE_W_EGG * (2.0 if is_endgame else 1.0) 
        
        # 2. ANALYSIS: Territory AND Potential
        # We need to know our own parity to count potential eggs
        # Player A (White) = Even (Sum % 2 == 0)
        # Player B (Black) = Odd (Sum % 2 == 1)
        # We can infer parity by checking if we can lay an egg at an Even spot
        # or just relying on the heuristic that we want to move to empty spots we can reach.
        
        # Determine Parity:
        # If I am at (0,0), sum is 0. If I can lay egg there, I am Even.
        # Since we might not be at a valid spot, we check `chicken.even_chicken` if possible
        # but simpler is to pass the parity into the analysis function.
        # Note: Chicken object has `.even_chicken` attribute (0 or 1).
        
        p_space, p_potential = self._bfs_analysis(b, True)
        e_space, e_potential = self._bfs_analysis(b, False)
        
        # Potential is "Reachable squares that are EMPTY and match my PARITY"
        # This drives the bot to fill the territory it created.
        potential_diff = p_potential - e_potential
        space_diff = p_space - e_space

        # 3. DISTANCE BIAS (To cure "Plain Move" idling)
        # If we didn't lay an egg this turn, are we standing closer to a valid spot?
        # We calculate distance to nearest valid empty square
        dist_bonus = 0
        if p_potential > 0:
            dist = self._dist_to_nearest_valid(b)
            # Closer is better. Max distance on board is ~14. 
            dist_bonus = (14 - dist) * 2.0

        # Risk Calculation
        loc = p_chk.get_location()
        risk = self.get_risk(loc)
        risk_penalty = risk * 2000.0 

        score = (w_egg * egg_diff) + \
                (self.BASE_W_POTENTIAL * potential_diff) + \
                (self.BASE_W_SPACE * space_diff) + \
                dist_bonus - \
                risk_penalty
        
        return score

    def _bfs_analysis(self, b: Board, is_player: bool) -> Tuple[int, int]:
        """
        Returns (Reachable Count, Potential Egg Spots).
        Potential Egg Spots = Reachable squares that are EMPTY and match PARITY.
        """
        if is_player:
            agent = b.chicken_player
            start_node = agent.get_location()
            enemy_eggs = b.eggs_enemy
            enemy_turds = b.turds_enemy
            my_eggs = b.eggs_player
            my_turds = b.turds_player
        else:
            agent = b.chicken_enemy
            start_node = agent.get_location()
            enemy_eggs = b.eggs_player
            enemy_turds = b.turds_player
            my_eggs = b.eggs_enemy
            my_turds = b.turds_enemy

        # Access parity directly from the Chicken object
        # The attribute is .even_chicken (0 or 1)
        parity = agent.even_chicken 

        queue = [start_node]
        visited = {start_node}
        
        space_count = 0
        potential_count = 0
        
        lethal_zones = set()
        for (tx, ty) in enemy_turds:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lethal_zones.add((tx + dx, ty + dy))
        
        while queue:
            cx, cy = queue.pop(0)
            space_count += 1
            
            # Check Potential (Is this a valid place to score?)
            # Must be: Correct Parity, Not already having my egg/turd, Not having enemy egg/turd
            if (cx + cy) % 2 == parity:
                if (cx, cy) not in my_eggs and (cx, cy) not in my_turds and \
                   (cx, cy) not in enemy_eggs and (cx, cy) not in enemy_turds:
                    potential_count += 1
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size): continue
                if (nx, ny) in visited: continue
                if (nx, ny) in enemy_eggs: continue
                if (nx, ny) in lethal_zones: continue
                
                visited.add((nx, ny))
                queue.append((nx, ny))
                
        return space_count, potential_count

    def _dist_to_nearest_valid(self, b: Board) -> int:
        """BFS to find distance to nearest empty valid egg spot."""
        start_node = b.chicken_player.get_location()
        parity = b.chicken_player.even_chicken
        
        queue = [(start_node, 0)]
        visited = {start_node}
        
        # We can't step on enemy stuff
        enemy_eggs = b.eggs_enemy
        enemy_turds = b.turds_enemy
        my_eggs = b.eggs_player
        my_turds = b.turds_player
        
        lethal_zones = set()
        for (tx, ty) in enemy_turds:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    lethal_zones.add((tx + dx, ty + dy))

        while queue:
            (cx, cy), dist = queue.pop(0)
            
            # Check if target
            if (cx + cy) % 2 == parity:
                 if (cx, cy) not in my_eggs and (cx, cy) not in my_turds and \
                   (cx, cy) not in enemy_eggs and (cx, cy) not in enemy_turds:
                    return dist

            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                
                if not (0 <= nx < self.map_size and 0 <= ny < self.map_size): continue
                if (nx, ny) in visited: continue
                if (nx, ny) in enemy_eggs: continue
                if (nx, ny) in lethal_zones: continue
                
                visited.add((nx, ny))
                queue.append(((nx, ny), dist + 1))
        
        return 20 # Max fallback if no spots reachable