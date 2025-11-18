"""
Training Data Generation for AgentB

Generates diverse game positions with evaluations for training the neural network.
"""

import sys
import os
import random
import json
import numpy as np
from typing import List, Dict, Tuple

# Add parent directories to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from game import Board
from game.enums import Direction, MoveType
from agent import PlayerAgent


class TrainingDataGenerator:
    """
    Generates training data by:
    1. Playing random games to get diverse positions
    2. Evaluating positions with deep search
    3. Augmenting with random trapdoor locations
    """

    def __init__(self, depth_for_labels: int = 6):
        self.depth_for_labels = depth_for_labels
        self.positions_collected = []

    def generate_random_position(self, min_moves: int = 5, max_moves: int = 35) -> Board:
        """
        Generate a random but legal game position.

        This creates diverse training data by:
        - Starting from initial position
        - Playing random moves
        - Varying trapdoor locations
        """
        board = Board()
        num_moves = random.randint(min_moves, max_moves)

        current_perspective_is_a = True

        for _ in range(num_moves):
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break

            # Weighted random: prefer egg moves for more interesting positions
            move_weights = []
            for move in valid_moves:
                if move[1] == MoveType.EGG:
                    move_weights.append(3.0)  # 3x more likely
                elif move[1] == MoveType.TURD:
                    move_weights.append(1.5)
                else:
                    move_weights.append(1.0)

            # Normalize weights
            total = sum(move_weights)
            move_probs = [w / total for w in move_weights]

            # Choose move
            move = random.choices(valid_moves, weights=move_probs, k=1)[0]

            # Apply move
            board = board.forecast_move(move[0], move[1], check_ok=False)
            if board is None:
                return None

            # Switch perspective
            board.reverse_perspective()
            current_perspective_is_a = not current_perspective_is_a

        return board

    def evaluate_position_deep(
        self,
        board: Board,
        agent: PlayerAgent
    ) -> float:
        """
        Evaluate position using deep search.

        This is our "ground truth" - we use a deep search to get
        an accurate evaluation, then train the NN to predict this.
        """
        try:
            # Use deep search (6+ ply)
            score, _ = agent.search_engine._minimax(
                board=board,
                depth=self.depth_for_labels,
                alpha=float('-inf'),
                beta=float('inf'),
                maximizing=True,
                time_left=30.0,  # Allow plenty of time
                trapdoor_tracker=agent.trapdoor_tracker
            )

            # Normalize to [-1, 1] range for tanh output
            # Score is roughly in range [-10000, 10000]
            normalized_score = np.tanh(score / 1000.0)

            return normalized_score

        except Exception as e:
            print(f"Error evaluating position: {e}")
            return 0.0

    def extract_features_from_board(
        self,
        board: Board,
        agent: PlayerAgent
    ) -> np.ndarray:
        """
        Extract features from board state.

        Features include:
        - Position information
        - Egg counts
        - Turd counts
        - Mobility
        - Trapdoor probabilities (if available)
        """
        features = np.zeros(128)  # Expanded feature set

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        map_size = board.game_map.MAP_SIZE

        # Basic position features (0-7)
        features[0] = my_loc[0] / map_size
        features[1] = my_loc[1] / map_size
        features[2] = enemy_loc[0] / map_size
        features[3] = enemy_loc[1] / map_size

        # Distance features (4-7)
        dx = abs(my_loc[0] - enemy_loc[0]) / map_size
        dy = abs(my_loc[1] - enemy_loc[1]) / map_size
        features[4] = dx
        features[5] = dy
        features[6] = (dx + dy) / 2  # Manhattan
        features[7] = np.sqrt(dx**2 + dy**2)  # Euclidean

        # Egg counts (8-11)
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        features[8] = my_eggs / 40.0
        features[9] = enemy_eggs / 40.0
        features[10] = (my_eggs - enemy_eggs) / 40.0  # Difference
        features[11] = my_eggs / max(enemy_eggs, 1)  # Ratio

        # Turd counts (12-15)
        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()
        features[12] = my_turds / 5.0
        features[13] = enemy_turds / 5.0
        features[14] = (my_turds - enemy_turds) / 5.0
        features[15] = len(board.turds_player) / 5.0  # Placed turds

        # Board control features (16-31) - quadrant analysis
        quadrants = [
            (0, map_size//2, 0, map_size//2),
            (map_size//2, map_size, 0, map_size//2),
            (0, map_size//2, map_size//2, map_size),
            (map_size//2, map_size, map_size//2, map_size)
        ]

        for i, (x1, x2, y1, y2) in enumerate(quadrants):
            my_eggs_quad = sum(1 for e in board.eggs_player if x1 <= e[0] < x2 and y1 <= e[1] < y2)
            enemy_eggs_quad = sum(1 for e in board.eggs_enemy if x1 <= e[0] < x2 and y1 <= e[1] < y2)
            my_turds_quad = sum(1 for t in board.turds_player if x1 <= t[0] < x2 and y1 <= t[1] < y2)
            enemy_turds_quad = sum(1 for t in board.turds_enemy if x1 <= t[0] < x2 and y1 <= t[1] < y2)

            features[16 + i*4] = my_eggs_quad / 10.0
            features[17 + i*4] = enemy_eggs_quad / 10.0
            features[18 + i*4] = my_turds_quad / 5.0
            features[19 + i*4] = enemy_turds_quad / 5.0

        # Mobility features (32-35)
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()

        features[32] = my_moves / 8.0
        features[33] = enemy_moves / 8.0
        features[34] = (my_moves - enemy_moves) / 8.0
        features[35] = my_moves / max(enemy_moves, 1)

        # Time/phase features (36-39)
        features[36] = board.turn_count / 80.0
        features[37] = board.turns_left_player / 40.0
        features[38] = board.turns_left_enemy / 40.0
        features[39] = min(board.turns_left_player, board.turns_left_enemy) / 40.0

        # Positional features (40-50)
        # Center control
        center = map_size / 2.0
        my_center_dist = (abs(my_loc[0] - center) + abs(my_loc[1] - center)) / map_size
        enemy_center_dist = (abs(enemy_loc[0] - center) + abs(enemy_loc[1] - center)) / map_size
        features[40] = my_center_dist
        features[41] = enemy_center_dist
        features[42] = enemy_center_dist - my_center_dist

        # Edge proximity
        my_edge_dist = min(my_loc[0], my_loc[1], map_size-1-my_loc[0], map_size-1-my_loc[1]) / map_size
        enemy_edge_dist = min(enemy_loc[0], enemy_loc[1], map_size-1-enemy_loc[0], map_size-1-enemy_loc[1]) / map_size
        features[43] = my_edge_dist
        features[44] = enemy_edge_dist

        # Corner features
        corners = [(0, 0), (0, map_size-1), (map_size-1, 0), (map_size-1, map_size-1)]
        my_corner_dist = min(abs(my_loc[0]-c[0]) + abs(my_loc[1]-c[1]) for c in corners) / map_size
        enemy_corner_dist = min(abs(enemy_loc[0]-c[0]) + abs(enemy_loc[1]-c[1]) for c in corners) / map_size
        features[45] = my_corner_dist
        features[46] = enemy_corner_dist

        # Egg clustering (eggs near other eggs)
        my_egg_neighbors = sum(
            1 for e1 in board.eggs_player
            for e2 in board.eggs_player
            if e1 != e2 and abs(e1[0]-e2[0]) + abs(e1[1]-e2[1]) <= 2
        ) / max(my_eggs, 1)
        enemy_egg_neighbors = sum(
            1 for e1 in board.eggs_enemy
            for e2 in board.eggs_enemy
            if e1 != e2 and abs(e1[0]-e2[0]) + abs(e1[1]-e2[1]) <= 2
        ) / max(enemy_eggs, 1)
        features[47] = my_egg_neighbors / 10.0
        features[48] = enemy_egg_neighbors / 10.0

        # Trapdoor features (50-60) - if available
        if hasattr(agent, 'trapdoor_tracker') and agent.trapdoor_tracker:
            # Danger at current position
            features[50] = agent.trapdoor_tracker.get_danger_score(my_loc)
            features[51] = agent.trapdoor_tracker.get_danger_score(enemy_loc)

            # Danger in surrounding squares
            nearby_danger = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nearby = (my_loc[0] + dx, my_loc[1] + dy)
                    if 0 <= nearby[0] < map_size and 0 <= nearby[1] < map_size:
                        nearby_danger.append(agent.trapdoor_tracker.get_danger_score(nearby))

            features[52] = np.mean(nearby_danger) if nearby_danger else 0.0
            features[53] = np.max(nearby_danger) if nearby_danger else 0.0

        return features.astype(np.float32)

    def generate_dataset(
        self,
        num_positions: int = 10000,
        output_file: str = 'training_data.json'
    ):
        """
        Generate complete training dataset.
        """
        print(f"Generating {num_positions} training positions...")
        print(f"Using depth {self.depth_for_labels} for labels\n")

        # Initialize agent (without NN, using heuristics only)
        dummy_board = Board()
        agent = PlayerAgent(dummy_board, lambda: 300.0)

        dataset = []

        for i in range(num_positions):
            if i % 100 == 0:
                print(f"Progress: {i}/{num_positions} ({100*i/num_positions:.1f}%)")

            # Generate random position
            board = self.generate_random_position()
            if board is None or board.is_game_over():
                continue

            # Extract features
            features = self.extract_features_from_board(board, agent)

            # Get ground truth evaluation from deep search
            score = self.evaluate_position_deep(board, agent)

            # Store
            dataset.append({
                'features': features.tolist(),
                'score': float(score),
                'metadata': {
                    'turn': board.turn_count,
                    'eggs_diff': board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid()
                }
            })

        print(f"\n✓ Generated {len(dataset)} positions")
        print(f"Saving to {output_file}...")

        with open(output_file, 'w') as f:
            json.dump(dataset, f)

        print(f"✓ Dataset saved!")

        return dataset


if __name__ == '__main__':
    generator = TrainingDataGenerator(depth_for_labels=6)
    dataset = generator.generate_dataset(num_positions=5000)

    # Print statistics
    scores = [d['score'] for d in dataset]
    print(f"\nDataset Statistics:")
    print(f"  Positions: {len(dataset)}")
    print(f"  Score mean: {np.mean(scores):.3f}")
    print(f"  Score std: {np.std(scores):.3f}")
    print(f"  Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
