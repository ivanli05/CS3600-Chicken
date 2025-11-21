"""
Trapdoor Probability Tracker for AgentC

This module tracks and updates probability distributions for trapdoor locations
using enhanced Bayesian inference based on sensor readings (hearing/feeling).
"""

from typing import Tuple, List, Set
import numpy as np


class TrapdoorTracker:
    """
    Tracks trapdoor locations using Bayesian probability inference.
    
    Enhanced version with better prior distributions and more accurate
    likelihood calculations based on sensor feedback.
    """

    def __init__(self, map_size: int = 8):
        self.map_size = map_size

        # Two probability grids (white/black trapdoor)
        self.prob_white = self._initialize_prior(color='white')
        self.prob_black = self._initialize_prior(color='black')

        # Track observations per square for better inference
        self.observation_count = np.zeros((map_size, map_size))
        
        # Track sensor history for more accurate inference
        self.sensor_history: List[Tuple[Tuple[int, int], List[Tuple[bool, bool]]]] = []

        # Known trapdoors (confirmed by stepping on them)
        self.known_trapdoors: Set[Tuple[int, int]] = set()

    def _initialize_prior(self, color: str) -> np.ndarray:
        """
        Initialize prior probability distribution based on assignment rules.
        
        Enhanced prior: Trapdoors are weighted toward center:
        - Edge squares: weight 0
        - 1 layer in: weight 0
        - 2 layers in: weight 1
        - 3 layers in (center): weight 2
        """
        prob = np.zeros((self.map_size, self.map_size))

        for i in range(self.map_size):
            for j in range(self.map_size):
                # Only valid squares for this color
                if (i + j) % 2 == (0 if color == 'white' else 1):
                    # Weight by distance from edge
                    dist_from_edge = min(i, j, self.map_size - 1 - i, self.map_size - 1 - j)
                    weight = max(0, dist_from_edge - 1)  # 0, 0, 1, 2 for layers
                    prob[i, j] = weight

        # Normalize to probability distribution
        total = prob.sum()
        return prob / total if total > 0 else prob

    def update_beliefs(
        self,
        my_loc: Tuple[int, int],
        sensor_data: List[Tuple[bool, bool]]
    ):
        """
        Update probability beliefs based on sensor readings using Bayesian inference.
        
        Args:
            my_loc: Current location (x, y)
            sensor_data: [(heard_white, felt_white), (heard_black, felt_black)]
        """
        x, y = my_loc
        heard_white, felt_white = sensor_data[0]
        heard_black, felt_black = sensor_data[1]

        # Store sensor history for better inference
        self.sensor_history.append((my_loc, sensor_data))
        if len(self.sensor_history) > 50:  # Keep last 50 observations
            self.sensor_history.pop(0)

        # Update for white trapdoor
        self._bayesian_update(x, y, heard_white, felt_white, self.prob_white)

        # Update for black trapdoor
        self._bayesian_update(x, y, heard_black, felt_black, self.prob_black)

        # Track that we observed this square
        self.observation_count[x, y] += 1

    def _bayesian_update(
        self,
        x: int,
        y: int,
        heard: bool,
        felt: bool,
        prob_grid: np.ndarray
    ):
        """
        Bayesian update: P(trap at T | observation) ∝ P(observation | trap at T) * P(trap at T)
        
        Enhanced likelihood calculation with more accurate sensor probabilities.
        
        Sensor probabilities from assignment:
        - Adjacent (edge-sharing): 50% hear, 30% feel
        - Diagonal: 25% hear, 15% feel
        - 2-away (edge-sharing): 10% hear, 0% feel
        """
        likelihood = np.ones_like(prob_grid, dtype=float)

        for i in range(self.map_size):
            for j in range(self.map_size):
                dist_type = self._get_distance_type(x, y, i, j)

                # Get sensor probabilities for this distance
                if dist_type == 'adjacent':
                    p_hear, p_feel = 0.50, 0.30
                elif dist_type == 'diagonal':
                    p_hear, p_feel = 0.25, 0.15
                elif dist_type == 'two_away':
                    p_hear, p_feel = 0.10, 0.00
                else:
                    p_hear, p_feel = 0.0, 0.0

                # Calculate likelihood: P(observation | trap at (i,j))
                if felt:
                    # We felt it - high probability it's close (adjacent or diagonal)
                    if dist_type == 'adjacent':
                        likelihood[i, j] = p_feel
                    elif dist_type == 'diagonal':
                        likelihood[i, j] = p_feel * 0.8  # Slightly lower for diagonal
                    else:
                        likelihood[i, j] = 0.01  # Very unlikely if far away
                elif heard:
                    # We heard but didn't feel - could be diagonal, 2-away, or adjacent
                    likelihood[i, j] = p_hear * (1 - p_feel)
                else:
                    # Negative evidence: didn't hear or feel
                    # This makes nearby squares LESS likely
                    likelihood[i, j] = (1 - p_hear) * (1 - p_feel)

        # Posterior ∝ Likelihood × Prior
        prob_grid[:] = likelihood * prob_grid

        # Normalize to maintain probability distribution
        total = prob_grid.sum()
        if total > 0:
            prob_grid[:] = prob_grid / total
        else:
            # If all probabilities are zero, reinitialize with uniform prior
            prob_grid[:] = self._initialize_prior('white' if prob_grid is self.prob_white else 'black')

    def _get_distance_type(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int
    ) -> str:
        """
        Determine the type of distance relationship between two squares.
        
        Returns: 'adjacent', 'diagonal', 'two_away', or 'far'
        """
        dx, dy = abs(x1 - x2), abs(y1 - y2)

        # Same square
        if dx == 0 and dy == 0:
            return 'adjacent'  # We'd definitely know if we're on it

        # Adjacent (edge-sharing)
        if (dx == 1 and dy == 0) or (dx == 0 and dy == 1):
            return 'adjacent'

        # Diagonal
        if dx == 1 and dy == 1:
            return 'diagonal'

        # Two-away (knight's move distance, but only cardinal)
        if (dx == 2 and dy == 0) or (dx == 0 and dy == 2):
            return 'two_away'

        # Too far to sense
        return 'far'

    def get_danger_score(self, loc: Tuple[int, int]) -> float:
        """
        Get the combined trapdoor danger score for a location.
        
        Returns: Probability (0.0 to 1.0) that there's a trapdoor at this location
        """
        x, y = loc
        if not (0 <= x < self.map_size and 0 <= y < self.map_size):
            return 0.0

        # Combined probability from both trapdoors
        return self.prob_white[x, y] + self.prob_black[x, y]

    def get_most_likely_trapdoors(self, n: int = 5) -> List[Tuple[Tuple[int, int], float]]:
        """
        Get the N most likely trapdoor locations.
        
        Returns: List of ((x, y), probability) tuples
        """
        candidates = []

        # Check white squares
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i + j) % 2 == 0:
                    prob = self.prob_white[i, j]
                    if prob > 0.01:  # Filter noise
                        candidates.append(((i, j), prob))

        # Check black squares
        for i in range(self.map_size):
            for j in range(self.map_size):
                if (i + j) % 2 == 1:
                    prob = self.prob_black[i, j]
                    if prob > 0.01:
                        candidates.append(((i, j), prob))

        # Sort by probability
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:n]

    def mark_trapdoor_found(self, loc: Tuple[int, int]):
        """Mark a trapdoor as confirmed (agent stepped on it)"""
        self.known_trapdoors.add(loc)

        # Set probability to 1.0 at this location
        x, y = loc
        if (x + y) % 2 == 0:
            self.prob_white[:] = 0.0
            self.prob_white[x, y] = 1.0
        else:
            self.prob_black[:] = 0.0
            self.prob_black[x, y] = 1.0

    def get_summary(self) -> str:
        """Get a summary of current trapdoor beliefs"""
        likely = self.get_most_likely_trapdoors(3)
        summary = "Trapdoor beliefs:\n"
        for (x, y), prob in likely:
            color = "white" if (x + y) % 2 == 0 else "black"
            summary += f"  ({x},{y}) [{color}]: {prob:.1%}\n"
        return summary

