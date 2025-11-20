"""
Comprehensive Feature Extraction for Neural Network Training

This module extracts a complete 128-dimensional feature vector from board positions.
Features are designed to capture all strategic aspects of the game.

Feature Layout (128 total):
- [0-63]:   Trapdoor probability map (8x8 grid)
- [64-71]:  Position features
- [72-79]:  Material features (eggs, turds)
- [80-87]:  Mobility & tempo features
- [88-103]: Spatial control features
- [104-119]: Tactical proximity features
- [120-127]: Strategic features
"""

import numpy as np
from typing import Tuple, Optional
import game.board as board_module
from game.enums import Direction, loc_after_direction


class FeatureExtractor:
    """Extracts comprehensive features from board positions"""

    def __init__(self, map_size: int = 8):
        self.map_size = map_size
        self.feature_size = 128

    def extract_features(
        self,
        board: board_module.Board,
        trapdoor_tracker=None
    ) -> np.ndarray:
        """
        Extract complete 128-dimensional feature vector.

        Args:
            board: Current board state
            trapdoor_tracker: Optional trapdoor probability tracker

        Returns:
            128-dimensional feature vector (float32)
        """
        features = np.zeros(self.feature_size, dtype=np.float32)

        try:
            my_loc = board.chicken_player.get_location()
            enemy_loc = board.chicken_enemy.get_location()

            # [0-63] Trapdoor probability map (MOST IMPORTANT!)
            if trapdoor_tracker is not None:
                idx = 0
                for y in range(self.map_size):
                    for x in range(self.map_size):
                        location = (x, y)
                        # Get probability this square is a trapdoor
                        prob = trapdoor_tracker.get_danger_score(location)
                        # Mark known trapdoors as 1.0
                        if location in trapdoor_tracker.known_trapdoors:
                            prob = 1.0
                        features[idx] = prob
                        idx += 1

            # [64-71] Position features
            features[64] = my_loc[0] / self.map_size  # My X
            features[65] = my_loc[1] / self.map_size  # My Y
            features[66] = enemy_loc[0] / self.map_size  # Enemy X
            features[67] = enemy_loc[1] / self.map_size  # Enemy Y

            # Distance between chickens
            distance = abs(my_loc[0] - enemy_loc[0]) + abs(my_loc[1] - enemy_loc[1])
            features[68] = distance / (2 * self.map_size)  # Normalize by max distance

            # Distance to center
            center = self.map_size / 2.0
            my_center_dist = abs(my_loc[0] - center) + abs(my_loc[1] - center)
            enemy_center_dist = abs(enemy_loc[0] - center) + abs(enemy_loc[1] - center)
            features[69] = my_center_dist / self.map_size
            features[70] = enemy_center_dist / self.map_size

            # Relative position (angle-like feature)
            dx = enemy_loc[0] - my_loc[0]
            dy = enemy_loc[1] - my_loc[1]
            if distance > 0:
                features[71] = (dx + dy) / (2 * distance)  # Normalized direction

            # [72-79] Material features
            my_eggs = board.chicken_player.get_eggs_laid()
            enemy_eggs = board.chicken_enemy.get_eggs_laid()
            features[72] = my_eggs / 40.0  # Max possible eggs ~40
            features[73] = enemy_eggs / 40.0
            features[74] = (my_eggs - enemy_eggs) / 20.0  # Normalized difference

            my_turds = board.chicken_player.get_turds_left()
            enemy_turds = board.chicken_enemy.get_turds_left()
            features[75] = my_turds / 5.0  # Max 5 turds
            features[76] = enemy_turds / 5.0
            features[77] = (my_turds - enemy_turds) / 5.0

            # [80-87] Mobility & tempo features
            my_moves = len(board.get_valid_moves())
            board.reverse_perspective()
            enemy_moves = len(board.get_valid_moves())
            board.reverse_perspective()

            features[80] = my_moves / 8.0  # Max ~8 directions
            features[81] = enemy_moves / 8.0
            features[82] = (my_moves - enemy_moves) / 8.0

            # Turn information
            turn_count = getattr(board, 'turn_count', 0)
            features[83] = turn_count / 80.0  # Max 80 turns total
            features[84] = board.turns_left_player / 40.0

            # Game phase (early: 0, mid: 0.5, late: 1)
            game_phase = turn_count / 80.0
            features[86] = game_phase

            # [88-103] Spatial control features
            self._extract_spatial_features(board, features)

            # [104-119] Tactical proximity features
            self._extract_tactical_features(board, features)

            # [120-127] Strategic features
            self._extract_strategic_features(board, features)

            return features

        except Exception as e:
            # On error, return zero vector (safe fallback)
            return np.zeros(self.feature_size, dtype=np.float32)

    def _extract_spatial_features(self, board: board_module.Board, features: np.ndarray):
        """Extract spatial control features [88-103]"""
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        # [88-91] Corner control (who can lay eggs in which corners)
        corners = [
            (0, 0), (0, self.map_size - 1),
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]

        for i, corner in enumerate(corners):
            can_i_lay = board.chicken_player.can_lay_egg(corner)
            can_enemy_lay = board.chicken_enemy.can_lay_egg(corner)
            # 0 = neither, 0.5 = enemy, 1.0 = me
            if can_i_lay:
                features[88 + i] = 1.0
            elif can_enemy_lay:
                features[88 + i] = 0.5

        # [92-95] Center control (4 center squares)
        center = self.map_size // 2
        center_squares = [
            (center - 1, center - 1), (center - 1, center),
            (center, center - 1), (center, center)
        ]

        for i, square in enumerate(center_squares):
            my_dist = abs(square[0] - my_loc[0]) + abs(square[1] - my_loc[1])
            enemy_dist = abs(square[0] - enemy_loc[0]) + abs(square[1] - enemy_loc[1])
            # Closer = better control (normalized)
            control = 0.5 + (enemy_dist - my_dist) / (2 * self.map_size)
            features[92 + i] = np.clip(control, 0, 1)

        # [96-99] Egg spread by quadrant
        mid = self.map_size // 2
        quadrants = [(0, 0), (0, 1), (1, 0), (1, 1)]  # TL, TR, BL, BR

        for i, (qx, qy) in enumerate(quadrants):
            my_eggs_in_quad = 0
            enemy_eggs_in_quad = 0

            for egg in board.eggs_player:
                in_quad = (egg[0] < mid if qx == 0 else egg[0] >= mid) and \
                          (egg[1] < mid if qy == 0 else egg[1] >= mid)
                if in_quad:
                    my_eggs_in_quad += 1

            for egg in board.eggs_enemy:
                in_quad = (egg[0] < mid if qx == 0 else egg[0] >= mid) and \
                          (egg[1] < mid if qy == 0 else egg[1] >= mid)
                if in_quad:
                    enemy_eggs_in_quad += 1

            # Normalized egg count in quadrant
            features[96 + i] = (my_eggs_in_quad - enemy_eggs_in_quad) / 10.0

        # [100-103] Territory control by quadrant (based on position)
        for i, (qx, qy) in enumerate(quadrants):
            my_in_quad = (my_loc[0] < mid if qx == 0 else my_loc[0] >= mid) and \
                         (my_loc[1] < mid if qy == 0 else my_loc[1] >= mid)
            enemy_in_quad = (enemy_loc[0] < mid if qx == 0 else enemy_loc[0] >= mid) and \
                            (enemy_loc[1] < mid if qy == 0 else enemy_loc[1] >= mid)

            if my_in_quad:
                features[100 + i] = 1.0
            elif enemy_in_quad:
                features[100 + i] = -1.0

    def _extract_tactical_features(self, board: board_module.Board, features: np.ndarray):
        """Extract tactical proximity features [104-119]"""
        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()

        # [104-107] Nearest own egg in each direction
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right

        for i, (dx, dy) in enumerate(directions):
            min_dist = self.map_size * 2  # Max distance
            for egg in board.eggs_player:
                if (dx != 0 and np.sign(egg[0] - my_loc[0]) == dx) or \
                   (dy != 0 and np.sign(egg[1] - my_loc[1]) == dy):
                    dist = abs(egg[0] - my_loc[0]) + abs(egg[1] - my_loc[1])
                    min_dist = min(min_dist, dist)
            features[104 + i] = min_dist / (2 * self.map_size)

        # [108-111] Nearest enemy egg in each direction
        for i, (dx, dy) in enumerate(directions):
            min_dist = self.map_size * 2
            for egg in board.eggs_enemy:
                if (dx != 0 and np.sign(egg[0] - my_loc[0]) == dx) or \
                   (dy != 0 and np.sign(egg[1] - my_loc[1]) == dy):
                    dist = abs(egg[0] - my_loc[0]) + abs(egg[1] - my_loc[1])
                    min_dist = min(min_dist, dist)
            features[108 + i] = min_dist / (2 * self.map_size)

        # [112-115] Nearest own turd in each direction
        for i, (dx, dy) in enumerate(directions):
            min_dist = self.map_size * 2
            for turd in board.turds_player:
                if (dx != 0 and np.sign(turd[0] - my_loc[0]) == dx) or \
                   (dy != 0 and np.sign(turd[1] - my_loc[1]) == dy):
                    dist = abs(turd[0] - my_loc[0]) + abs(turd[1] - my_loc[1])
                    min_dist = min(min_dist, dist)
            features[112 + i] = min_dist / (2 * self.map_size)

        # [116-119] Nearest enemy turd in each direction
        for i, (dx, dy) in enumerate(directions):
            min_dist = self.map_size * 2
            for turd in board.turds_enemy:
                if (dx != 0 and np.sign(turd[0] - my_loc[0]) == dx) or \
                   (dy != 0 and np.sign(turd[1] - my_loc[1]) == dy):
                    dist = abs(turd[0] - my_loc[0]) + abs(turd[1] - my_loc[1])
                    min_dist = min(min_dist, dist)
            features[116 + i] = min_dist / (2 * self.map_size)

    def _extract_strategic_features(self, board: board_module.Board, features: np.ndarray):
        """Extract strategic features [120-127]"""

        # [120-121] Eggs in corners (valuable!)
        corners = [
            (0, 0), (0, self.map_size - 1),
            (self.map_size - 1, 0), (self.map_size - 1, self.map_size - 1)
        ]

        my_corner_eggs = sum(1 for egg in board.eggs_player if egg in corners)
        enemy_corner_eggs = sum(1 for egg in board.eggs_enemy if egg in corners)
        features[120] = my_corner_eggs / 4.0
        features[121] = enemy_corner_eggs / 4.0

        # [122-123] Egg clustering (eggs near other eggs)
        my_cluster_count = 0
        for egg in board.eggs_player:
            for other_egg in board.eggs_player:
                if egg != other_egg:
                    dist = abs(egg[0] - other_egg[0]) + abs(egg[1] - other_egg[1])
                    if dist <= 2:  # Close eggs
                        my_cluster_count += 1
                        break

        enemy_cluster_count = 0
        for egg in board.eggs_enemy:
            for other_egg in board.eggs_enemy:
                if egg != other_egg:
                    dist = abs(egg[0] - other_egg[0]) + abs(egg[1] - other_egg[1])
                    if dist <= 2:
                        enemy_cluster_count += 1
                        break

        features[122] = my_cluster_count / 20.0  # Normalize
        features[123] = enemy_cluster_count / 20.0

        # [124-125] Available egg-laying squares
        my_available = 0
        enemy_available = 0

        for x in range(self.map_size):
            for y in range(self.map_size):
                loc = (x, y)
                # Check if I can lay egg here
                if board.chicken_player.can_lay_egg(loc):
                    if loc not in board.eggs_player and loc not in board.eggs_enemy:
                        if loc not in board.turds_player and loc not in board.turds_enemy:
                            my_available += 1

                # Check if enemy can lay egg here
                if board.chicken_enemy.can_lay_egg(loc):
                    if loc not in board.eggs_player and loc not in board.eggs_enemy:
                        if loc not in board.turds_player and loc not in board.turds_enemy:
                            enemy_available += 1

        features[124] = my_available / 32.0  # Max ~32 squares per color
        features[125] = enemy_available / 32.0

        # [126-127] Reserved for future use

    def get_feature_names(self) -> list:
        """Get human-readable names for all features"""
        names = []

        # Trapdoor map
        for y in range(8):
            for x in range(8):
                names.append(f"trapdoor_prob_{x}_{y}")

        # Position
        names.extend(["my_x", "my_y", "enemy_x", "enemy_y", "distance",
                     "my_center_dist", "enemy_center_dist", "relative_pos"])

        # Material
        names.extend(["my_eggs", "enemy_eggs", "egg_diff",
                     "my_turds", "enemy_turds", "turd_diff", "reserved_78", "reserved_79"])

        # Mobility & tempo
        names.extend(["my_moves", "enemy_moves", "mobility_diff",
                     "turn_count", "my_turns_left", "reserved_85", "game_phase", "reserved_87"])

        # Spatial control
        names.extend(["corner_tl", "corner_tr", "corner_bl", "corner_br",
                     "center_tl", "center_tr", "center_bl", "center_br",
                     "eggs_quad_tl", "eggs_quad_tr", "eggs_quad_bl", "eggs_quad_br",
                     "terr_quad_tl", "terr_quad_tr", "terr_quad_bl", "terr_quad_br"])

        # Tactical
        names.extend(["my_egg_up", "my_egg_down", "my_egg_left", "my_egg_right",
                     "enemy_egg_up", "enemy_egg_down", "enemy_egg_left", "enemy_egg_right",
                     "my_turd_up", "my_turd_down", "my_turd_left", "my_turd_right",
                     "enemy_turd_up", "enemy_turd_down", "enemy_turd_left", "enemy_turd_right"])

        # Strategic
        names.extend(["my_corner_eggs", "enemy_corner_eggs",
                     "my_clustering", "enemy_clustering",
                     "my_available_squares", "enemy_available_squares",
                     "reserved_126", "reserved_127"])

        return names
