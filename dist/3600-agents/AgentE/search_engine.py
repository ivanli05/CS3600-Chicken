# """
# Search Engine for AgentB (CLEAN)
# """

# from typing import Tuple, Optional, List, Callable, Set
# import copy
# from game.enums import Direction, MoveType, loc_after_direction
# import game.board as board_module
# from .heuristics import MoveEvaluator

# class SearchEngine:
#     def __init__(
#         self,
#         evaluator: MoveEvaluator,
#         max_depth: int = 9,
#         time_limit: float = 0.8
#     ):
#         self.evaluator = evaluator
#         self.max_depth = max_depth
#         self.time_limit = time_limit

#     def search(
#         self,
#         board: board_module.Board,
#         time_left: Callable,
#         trapdoor_tracker=None,
#         root_moves: List[Tuple[Direction, MoveType]] = None
#     ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
#         """
#         Main search entry point.
#         """
#         available_time = time_left()
#         search_time = min(available_time * self.time_limit, 2.0)

#         depth = self.max_depth
#         if available_time > 60: depth += 1

#         score, best_move = self._minimax(
#             board=board,
#             depth=depth,
#             alpha=float('-inf'),
#             beta=float('inf'),
#             maximizing=True,
#             time_left=search_time,
#             trapdoor_tracker=trapdoor_tracker,
#             root_moves=root_moves
#         )

#         return score, best_move

#     def _minimax(
#         self,
#         board: board_module.Board,
#         depth: int,
#         alpha: float,
#         beta: float,
#         maximizing: bool,
#         time_left: float,
#         trapdoor_tracker=None,
#         root_moves=None
#     ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        
#         # 1. Terminal Node Check (Depth or Time)
#         if depth == 0 or time_left < 0.01:
#             score = self.evaluator.evaluate_position(board, trapdoor_tracker)
#             # FIX 1: If it's the enemy's turn (not maximizing), the board is flipped.
#             # The evaluator returns (Enemy - Me). We want (Me - Enemy).
#             # So we must negate the score.
#             if not maximizing:
#                 score = -score
#             return score, None

#         # 2. Game Over Check
#         if board.is_game_over():
#             # FIX 2: Same logic for terminal state. 
#             score = self._evaluate_terminal(board)
#             if not maximizing:
#                 score = -score
#             return score, None

#         # 3. Get Moves
#         if root_moves is not None and maximizing:
#             valid_moves = root_moves
#         else:
#             valid_moves = board.get_valid_moves()
#             if trapdoor_tracker:
#                  # Note: This filtering is good, but make sure it doesn't filter ALL moves
#                  # causing an empty list panic.
#                  safe_moves = [
#                      m for m in valid_moves 
#                      if loc_after_direction(board.chicken_player.get_location(), m[0]) 
#                      not in trapdoor_tracker.known_trapdoors
#                  ]
#                  # Fallback: if all moves are trapdoors, we must pick one (suicide is forced)
#                  if safe_moves:
#                      valid_moves = safe_moves

#         if not valid_moves:
#             # If we have no moves, we lose. 
#             # If maximizing (My Turn), return -inf. If minimizing (Enemy Turn), return +inf.
#             return (-100000.0 if maximizing else 100000.0), None

#         # 4. Move Ordering
#         ordered_moves = self._order_moves(valid_moves, board, trapdoor_tracker)
        
#         best_move = ordered_moves[0] # Default to first move in case loop fails
        
#         # 5. Search Loop
#         for move in ordered_moves:
#             try:
#                 # REWRITE: Update Board Logic
#                 # 1. We use forecast_move without 'check_ok=False'.
#                 #    Enabling checks ensures internal sets (eggs/turds) are 
#                 #    perfectly synced with the grid, preventing "Phantom Blocker" bugs.
#                 forecast = board.forecast_move(move[0], move[1])
                
#                 if forecast is None: continue
                
#                 # 2. Flip board for the next recursion (Enemy Perspective)
#                 forecast.reverse_perspective()
                
#                 # 3. Pass synced board to heuristics (via recursion)
#                 score, _ = self._minimax(
#                     forecast, 
#                     depth - 1, 
#                     alpha, 
#                     beta, 
#                     not maximizing, # Switch turns
#                     time_left - 0.05, # Slightly more realistic cost per node
#                     trapdoor_tracker, 
#                     None
#                 )
                
#                 if maximizing:
#                     if score > alpha:
#                         alpha = score
#                         best_move = move
#                     if alpha >= beta:
#                         break # Beta Cutoff
#                 else:
#                     if score < beta:
#                         beta = score
#                         best_move = move
#                     if beta <= alpha:
#                         break # Alpha Cutoff
                    
#             except Exception:
#                 continue

#         return (alpha if maximizing else beta), best_move
    
#     def _order_moves(self, moves, board, trapdoor_tracker):
#         """
#         Heuristic to sort moves so Alpha-Beta pruning cuts off branches earlier.
#         """
#         scores = []
#         for move in moves:
#             # Uses the fast heuristic from your evaluator
#             score = self.evaluator.quick_evaluate_move(move, board, trapdoor_tracker)
#             scores.append((score, move))
        
#         # Sort descending: we want to check the best-looking moves first
#         scores.sort(key=lambda x: x[0], reverse=True)
#         return [m for s, m in scores]

#     def _evaluate_terminal(self, board: board_module.Board) -> float:
#         # Returns +Big if current player wins, -Big if current player loses
#         # The calling function handles the negation if it's the enemy's turn.
#         my_eggs = board.chicken_player.get_eggs_laid()
#         enemy_eggs = board.chicken_enemy.get_eggs_laid()
        
#         if my_eggs > enemy_eggs: return 100000.0
#         elif enemy_eggs > my_eggs: return -100000.0
#         return 0.0 # Draw




"""
Search Engine with Advanced Time Management
Features:
1. Iterative Deepening - progressively search deeper until time runs out
2. Adaptive Time Allocation - use more time in critical game phases
3. Transposition Table - cache evaluated positions to avoid re-computation
4. Smart depth limits based on remaining time
"""

from typing import Tuple, Optional, List, Callable, Dict
import time
from game.enums import Direction, MoveType, loc_after_direction
import game.board as board_module
from .heuristics import MoveEvaluator

class TranspositionTable:
    """Cache for board positions to avoid re-evaluating same state"""
    def __init__(self, max_size: int = 100000):
        self.table: Dict[int, Tuple[float, int, Optional[Tuple[Direction, MoveType]]]] = {}
        self.max_size = max_size

    def get(self, board_hash: int, depth: int) -> Optional[Tuple[float, Optional[Tuple[Direction, MoveType]]]]:
        """Returns (score, best_move) if cached at sufficient depth"""
        if board_hash in self.table:
            cached_score, cached_depth, cached_move = self.table[board_hash]
            if cached_depth >= depth:
                return cached_score, cached_move
        return None

    def store(self, board_hash: int, depth: int, score: float, best_move: Optional[Tuple[Direction, MoveType]]):
        """Store position evaluation"""
        if len(self.table) >= self.max_size:
            # Simple eviction: remove random entry
            self.table.pop(next(iter(self.table)))
        self.table[board_hash] = (score, depth, best_move)

    def clear(self):
        """Clear the table (call at start of each turn)"""
        self.table.clear()

class SearchEngine:
    def __init__(
        self,
        evaluator: MoveEvaluator,
        max_depth: int = 20,  # Increased to 20 for deeper search
        time_limit: float = 0.8
    ):
        self.evaluator = evaluator
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.tt = TranspositionTable()
        self.nodes_searched = 0
        self.tt_hits = 0
        self.turn_count = 0  # Track turns for equal time allocation

    def _get_board_hash(self, board: board_module.Board) -> int:
        """Simple hash of board state for transposition table"""
        try:
            # Hash based on: player location, enemy location, eggs, turds
            h = hash((
                board.chicken_player.get_location(),
                board.chicken_enemy.get_location(),
                frozenset(board.eggs_player),
                frozenset(board.eggs_enemy),
                frozenset(board.turds_player),
                frozenset(board.turds_enemy),
            ))
            return h
        except:
            # Fallback to location-only hash if something fails
            return hash((
                board.chicken_player.get_location(),
                board.chicken_enemy.get_location(),
            ))

    def search(
        self,
        board: board_module.Board,
        time_left: Callable,
        trapdoor_tracker=None,
        root_moves: List[Tuple[Direction, MoveType]] = None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:
        """
        Main search with iterative deepening and equal time allocation.
        """
        self.tt.clear()  # Clear cache for new turn
        self.nodes_searched = 0
        self.tt_hits = 0
        self.turn_count += 1

        total_time = time_left()

        # Equal time allocation - divide remaining time by estimated moves left
        allocated_time = self._calculate_equal_time_allocation(total_time)

        # Cap maximum time per move to avoid timeout
        allocated_time = min(allocated_time, 30.0)

        print(f"[SEARCH] Turn {self.turn_count}, Total time: {total_time:.1f}s, Allocated: {allocated_time:.1f}s")

        start_time = time.time()
        deadline = start_time + allocated_time

        best_move = None
        best_score = float('-inf')

        # Iterative deepening: start at depth 1, go up to max_depth
        for depth in range(1, self.max_depth + 1):
            remaining = deadline - time.time()

            if remaining < 0.1:  # Need at least 0.1s to start a new iteration
                print(f"[SEARCH] Time limit reached at depth {depth-1}")
                break

            print(f"[SEARCH] Starting depth {depth} (remaining: {remaining:.2f}s)...")

            try:
                score, move = self._minimax(
                    board=board,
                    depth=depth,
                    alpha=float('-inf'),
                    beta=float('inf'),
                    maximizing=True,
                    deadline=deadline,
                    trapdoor_tracker=trapdoor_tracker,
                    root_moves=root_moves
                )

                # Only update if we got a valid result
                if move is not None:
                    best_move = move
                    best_score = score
                    elapsed = time.time() - start_time
                    print(f"[SEARCH] Depth {depth} complete: score={score:.1f}, nodes={self.nodes_searched}, tt_hits={self.tt_hits}, time={elapsed:.2f}s")
                else:
                    print(f"[SEARCH] Depth {depth} returned no move, keeping previous result")

            except TimeoutError:
                print(f"[SEARCH] Timeout during depth {depth}, using previous best")
                break

        total_elapsed = time.time() - start_time
        print(f"[SEARCH] Final: depth={depth-1}, score={best_score:.1f}, time={total_elapsed:.2f}s")

        return best_score, best_move

    def _calculate_equal_time_allocation(self, total_time: float) -> float:
        """
        Calculate equal time allocation per move.
        Assumes approximately 60 total moves per game.
        Adjusts estimate based on current turn count.
        """
        # Estimate total moves in a game (conservative estimate)
        estimated_total_moves = 60

        # Calculate moves remaining
        moves_remaining = max(1, estimated_total_moves - self.turn_count)

        # Allocate equal time per remaining move
        time_per_move = total_time / moves_remaining

        # Add small safety buffer (use 95% to avoid timeout)
        time_per_move *= 0.95

        return time_per_move

    def _minimax(
        self,
        board: board_module.Board,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        deadline: float,
        trapdoor_tracker=None,
        root_moves=None
    ) -> Tuple[float, Optional[Tuple[Direction, MoveType]]]:

        self.nodes_searched += 1

        # Check time limit
        if time.time() > deadline:
            raise TimeoutError("Search time limit exceeded")

        # Check transposition table (only for non-root nodes)
        if root_moves is None:
            board_hash = self._get_board_hash(board)
            cached = self.tt.get(board_hash, depth)
            if cached is not None:
                self.tt_hits += 1
                return cached

        # 1. Terminal Node Check (Depth)
        if depth == 0:
            score = self.evaluator.evaluate_position(board, trapdoor_tracker)
            if not maximizing:
                score = -score
            return score, None

        # 2. Game Over Check
        if board.is_game_over():
            score = self._evaluate_terminal(board)
            if not maximizing:
                score = -score
            return score, None

        # 3. Get Moves
        if root_moves is not None and maximizing:
            valid_moves = root_moves
        else:
            valid_moves = board.get_valid_moves()
            if trapdoor_tracker:
                 safe_moves = [
                     m for m in valid_moves
                     if loc_after_direction(board.chicken_player.get_location(), m[0])
                     not in trapdoor_tracker.known_trapdoors
                 ]
                 if safe_moves:
                     valid_moves = safe_moves

        if not valid_moves:
            return (-100000.0 if maximizing else 100000.0), None

        # 4. Move Ordering
        ordered_moves = self._order_moves(valid_moves, board, trapdoor_tracker)

        best_move = ordered_moves[0]

        # 5. Search Loop
        for move in ordered_moves:
            try:
                forecast = board.forecast_move(move[0], move[1])

                if forecast is None:
                    continue

                forecast.reverse_perspective()

                score, _ = self._minimax(
                    forecast,
                    depth - 1,
                    alpha,
                    beta,
                    not maximizing,
                    deadline,
                    trapdoor_tracker,
                    None
                )

                if maximizing:
                    if score > alpha:
                        alpha = score
                        best_move = move
                    if alpha >= beta:
                        break  # Beta Cutoff
                else:
                    if score < beta:
                        beta = score
                        best_move = move
                    if beta <= alpha:
                        break  # Alpha Cutoff

            except TimeoutError:
                # Propagate timeout up the call stack
                raise
            except Exception:
                continue

        final_score = alpha if maximizing else beta

        # Store in transposition table (only for non-root nodes)
        if root_moves is None:
            self.tt.store(board_hash, depth, final_score, best_move)

        return final_score, best_move

    def _order_moves(self, moves, board, trapdoor_tracker):
        """
        Heuristic to sort moves so Alpha-Beta pruning cuts off branches earlier.
        """
        scores = []
        for move in moves:
            score = self.evaluator.quick_evaluate_move(move, board, trapdoor_tracker)
            scores.append((score, move))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [m for s, m in scores]

    def _evaluate_terminal(self, board: board_module.Board) -> float:
        """Evaluate terminal game state"""
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()

        if my_eggs > enemy_eggs:
            return 100000.0
        elif enemy_eggs > my_eggs:
            return -100000.0
        return 0.0
