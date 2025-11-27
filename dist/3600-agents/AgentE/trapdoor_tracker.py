"""
Trapdoor Probability Tracker for AgentB (FIXED)

Major Fix: Corrected Coordinate System.
- Game uses (x, y) -> (Column, Row)
- Numpy uses [row, col] -> [y, x]
"""

from typing import Tuple, List, Set
import numpy as np

class TrapdoorTracker:
    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        self.prob_white = self._initialize_prior(color='white')
        self.prob_black = self._initialize_prior(color='black')
        self.observation_count = np.zeros((map_size, map_size))
        self.known_trapdoors: Set[Tuple[int, int]] = set()

    def _initialize_prior(self, color: str) -> np.ndarray:
        prob = np.zeros((self.map_size, self.map_size))
        for r in range(self.map_size):      # r = row (y)
            for c in range(self.map_size):  # c = col (x)
                # Check parity (x + y) % 2
                if (c + r) % 2 == (0 if color == 'white' else 1):
                    dist_from_edge = min(r, c, self.map_size - 1 - r, self.map_size - 1 - c)
                    weight = max(0, dist_from_edge - 1)
                    prob[r, c] = weight
        total = prob.sum()
        return prob / total if total > 0 else prob

    def update_beliefs(self, my_loc: Tuple[int, int], sensor_data: List[Tuple[bool, bool]]):
        # my_loc is (x, y) -> (Col, Row)
        x, y = my_loc 
        heard_white, felt_white = sensor_data[0]
        heard_black, felt_black = sensor_data[1]

        self._bayesian_update(x, y, heard_white, felt_white, self.prob_white)
        self._bayesian_update(x, y, heard_black, felt_black, self.prob_black)

        # Track observations (Access as [row, col] -> [y, x])
        self.observation_count[y, x] += 1

        # We survived this square, so it is NOT a trapdoor
        self.prob_white[y, x] = 0.0
        self.prob_black[y, x] = 0.0

        self._normalize(self.prob_white)
        self._normalize(self.prob_black)

    def _normalize(self, grid: np.ndarray):
        total = grid.sum()
        if total > 0: grid[:] = grid / total

    def _bayesian_update(self, x: int, y: int, heard: bool, felt: bool, prob_grid: np.ndarray):
        """
        x: Player Column
        y: Player Row
        prob_grid: Numpy array [Row, Col]
        """
        likelihood = np.ones_like(prob_grid, dtype=float)

        # Iterate through CANDIDATE trapdoors
        for r in range(self.map_size):      # r = Candidate Row
            for c in range(self.map_size):  # c = Candidate Col
                
                # FIXED: Compare x (Player Col) with c (Cand Col)
                # FIXED: Compare y (Player Row) with r (Cand Row)
                dist_type = self._get_distance_type(x, y, c, r)

                if dist_type == 'adjacent':
                    p_hear, p_feel = 0.50, 0.30
                elif dist_type == 'diagonal':
                    p_hear, p_feel = 0.25, 0.15
                elif dist_type == 'zone3':
                    p_hear, p_feel = 0.10, 0.00
                else:
                    p_hear, p_feel = 0.0, 0.0

                p_obs_hear = p_hear if heard else (1.0 - p_hear)
                p_obs_feel = p_feel if felt else (1.0 - p_feel)
                
                likelihood[r, c] = p_obs_hear * p_obs_feel

        prob_grid[:] = likelihood * prob_grid
        self._normalize(prob_grid)

    def _get_distance_type(self, x1: int, y1: int, x2: int, y2: int) -> str:
        """Helper: logic remains the same, but inputs must be consistent (Col, Row, Col, Row)"""
        dx, dy = abs(x1 - x2), abs(y1 - y2)
        
        if dx == 0 and dy == 0: return 'adjacent'
        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1): return 'adjacent'
        if dx == 1 and dy == 1: return 'diagonal'
        # Knight's move or straight 2
        if (dx == 2 and dy == 0) or (dx == 0 and dy == 2): return 'zone3'
        if (dx == 1 and dy == 2) or (dx == 2 and dy == 1): return 'zone3'
        return 'far'

    def get_danger_score(self, loc: Tuple[int, int]) -> float:
        # loc is (x, y) -> (Col, Row)
        x, y = loc
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return 0.0
        # Numpy Access: [Row, Col] -> [y, x]
        return self.prob_white[y, x] + self.prob_black[y, x]

    def get_most_likely_trapdoors(self, n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        candidates = []
        for r in range(self.map_size):      # Row
            for c in range(self.map_size):  # Col
                # Return tuple as (x, y) -> (c, r)
                p_w = self.prob_white[r, c]
                if p_w > 0.01: candidates.append(((c, r), p_w))
                
                p_b = self.prob_black[r, c]
                if p_b > 0.01: candidates.append(((c, r), p_b))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def mark_trapdoor_found(self, loc: Tuple[int, int]):
        self.known_trapdoors.add(loc)
        x, y = loc # (Col, Row)
        
        # Access [Row, Col] -> [y, x]
        if (x + y) % 2 == 0:
            self.prob_white[:] = 0.0
            self.prob_white[y, x] = 1.0
        else:
            self.prob_black[:] = 0.0
            self.prob_black[y, x] = 1.0
            
    def get_summary(self) -> str:
        likely = self.get_most_likely_trapdoors(3)
        summary = "Trapdoor beliefs:\n"
        for (x, y), prob in likely:
            color = "white" if (x + y) % 2 == 0 else "black"
            summary += f"  ({x},{y}) [{color}]: {prob:.1%}\n"
        return summary