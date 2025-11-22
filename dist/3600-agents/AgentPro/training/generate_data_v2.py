"""
Generate 30k training positions with depth-9 minimax labels
Improved version with better anti-repetition heuristics
"""

import sys
import os
import json
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Add paths to import game modules
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)
engine_dir = os.path.join(os.path.dirname(os.path.dirname(agentpro_dir)), 'engine')

sys.path.insert(0, agentpro_dir)
sys.path.insert(0, engine_dir)

from game.board import Board
from game.enums import Direction, MoveType
from heuristics import MoveEvaluator
from feature_extractor import FeatureExtractor


class PositionGenerator:
    """Generate diverse game positions with minimax evaluations"""

    def __init__(self, depth=9):
        self.depth = depth
        self.evaluator = MoveEvaluator()
        self.feature_extractor = FeatureExtractor()

    def generate_random_position(self, min_moves=8, max_moves=30, seed=None):
        """
        Generate a random position by playing random moves.
        Returns (board, features, minimax_score)
        """
        if seed is not None:
            np.random.seed(seed)

        # Import game modules
        from game.game_map import GameMap
        from game.trapdoor_manager import TrapdoorManager

        # Create board with game map
        game_map = GameMap()
        trapdoor_manager = TrapdoorManager(game_map)
        board = Board(game_map)

        # Initialize chickens at spawn positions
        spawns = trapdoor_manager.choose_spawns()
        board.chicken_player.start(spawns[0], 0)
        board.chicken_enemy.start(spawns[1], 1)

        num_moves = np.random.randint(min_moves, max_moves + 1)

        # Play random moves with weighted selection (favor EGG moves)
        for _ in range(num_moves):
            if board.is_game_over():
                break

            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break

            # Weight moves: 60% EGG, 30% PLAIN, 10% TURD
            move_weights = []
            for move in valid_moves:
                if move[1] == MoveType.EGG:
                    move_weights.append(0.6)
                elif move[1] == MoveType.PLAIN:
                    move_weights.append(0.3)
                else:  # TURD
                    move_weights.append(0.1)

            # Normalize weights
            move_weights = np.array(move_weights)
            move_weights = move_weights / move_weights.sum()

            # Select move
            move_idx = np.random.choice(len(valid_moves), p=move_weights)
            move = valid_moves[move_idx]

            # Apply move using forecast (safer than play_move)
            board = board.forecast_move(move[0], move[1], check_ok=False)
            if board is None:
                return None  # Invalid move, skip this position

            # Switch perspective for next player
            board.reverse_perspective()

        # Skip if game is over
        if board.is_game_over():
            return None

        # Extract features
        features = self.feature_extractor.extract_features(board, trapdoor_tracker=None)

        # Get minimax evaluation
        score = self.minimax(board, self.depth, float('-inf'), float('inf'), True)

        # Normalize score to [-2, +2] range (roughly)
        # Typical scores range from -3000 to +3000
        normalized_score = np.clip(score / 1500.0, -2.0, 2.0)

        return {
            'features': features.tolist(),
            'score': float(normalized_score),
            'raw_score': float(score),
            'moves_played': num_moves
        }

    def minimax(self, board, depth, alpha, beta, maximizing):
        """
        Minimax search with alpha-beta pruning.
        Returns evaluation score.
        """
        if depth == 0 or board.is_game_over():
            return self.evaluator.evaluate_position(board)

        valid_moves = board.get_valid_moves()
        if not valid_moves:
            return self.evaluator.evaluate_position(board)

        if maximizing:
            max_eval = float('-inf')
            for move in valid_moves:
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                forecast.reverse_perspective()
                eval_score = self.minimax(forecast, depth - 1, alpha, beta, False)
                forecast.reverse_perspective()

                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)

                if beta <= alpha:
                    break  # Beta cutoff

            return max_eval
        else:
            min_eval = float('inf')
            for move in valid_moves:
                forecast = board.forecast_move(move[0], move[1], check_ok=False)
                if forecast is None:
                    continue

                forecast.reverse_perspective()
                eval_score = self.minimax(forecast, depth - 1, alpha, beta, True)
                forecast.reverse_perspective()

                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)

                if beta <= alpha:
                    break  # Alpha cutoff

            return min_eval


def generate_position_worker(args):
    """Worker function for parallel processing"""
    worker_id, num_positions, depth, min_moves, max_moves = args

    generator = PositionGenerator(depth=depth)
    positions = []

    base_seed = worker_id * 1000000

    for i in range(num_positions):
        seed = base_seed + i
        position = generator.generate_random_position(
            min_moves=min_moves,
            max_moves=max_moves,
            seed=seed
        )

        if position is not None:
            positions.append(position)

        # Progress reporting (every 100 positions)
        if (i + 1) % 100 == 0:
            print(f"Worker {worker_id}: {i + 1}/{num_positions} positions generated")

    return positions


def main():
    print("="*70)
    print("Training Data Generation v2 - 30k Positions @ Depth 9")
    print("Improved anti-repetition heuristics")
    print("="*70)

    # Configuration
    TARGET_POSITIONS = 20000
    DEPTH = 9
    MIN_MOVES = 8
    MAX_MOVES = 30
    NUM_WORKERS = 32
    OUTPUT_FILE = 'training_data_v2.json'

    print(f"\nConfiguration:")
    print(f"  Target positions: {TARGET_POSITIONS:,}")
    print(f"  Minimax depth: {DEPTH}")
    print(f"  Move range: {MIN_MOVES}-{MAX_MOVES}")
    print(f"  Workers: {NUM_WORKERS}")
    print(f"  Output: {OUTPUT_FILE}")

    # Calculate positions per worker
    positions_per_worker = TARGET_POSITIONS // NUM_WORKERS

    print(f"\nStarting parallel generation...")
    print(f"Each worker will generate ~{positions_per_worker} positions")
    print(f"Estimated time: ~7.5 hours (at 0.74 pos/s per worker)")
    print()

    start_time = datetime.now()

    # Create worker arguments
    worker_args = [
        (worker_id, positions_per_worker, DEPTH, MIN_MOVES, MAX_MOVES)
        for worker_id in range(NUM_WORKERS)
    ]

    # Run parallel generation
    all_positions = []
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(generate_position_worker, worker_args)
        for worker_positions in results:
            all_positions.extend(worker_positions)

    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"  Total positions: {len(all_positions):,}")
    print(f"  Time elapsed: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"  Rate: {len(all_positions)/elapsed:.2f} positions/second")
    print(f"{'='*70}")

    # Statistics
    scores = [p['score'] for p in all_positions]
    turns = [p['moves_played'] for p in all_positions]

    print(f"\nTurn distribution:")
    print(f"  Mean: {np.mean(turns):.1f}")
    print(f"  Std:  {np.std(turns):.1f}")
    print(f"  Min:  {min(turns)}")
    print(f"  Max:  {max(turns)}")

    print(f"\nScore distribution:")
    score_ranges = [
        ("Very negative (< -1.0)   ", lambda s: s < -1.0),
        ("Negative (-1.0 to -0.3)  ", lambda s: -1.0 <= s < -0.3),
        ("Slight negative (-0.3 to -0.1)", lambda s: -0.3 <= s < -0.1),
        ("Balanced (-0.1 to +0.1)  ", lambda s: -0.1 <= s <= 0.1),
        ("Slight positive (+0.1 to +0.3)", lambda s: 0.1 < s <= 0.3),
        ("Positive (+0.3 to +1.0)  ", lambda s: 0.3 < s <= 1.0),
        ("Very positive (> +1.0)   ", lambda s: s > 1.0),
    ]

    for label, condition in score_ranges:
        count = sum(1 for s in scores if condition(s))
        pct = count / len(scores) * 100
        bar = "█" * int(pct / 2)
        print(f"  {label}: {count:6d} ({pct:5.1f}%) {bar}")

    print(f"{'='*70}")

    # Save to JSON
    print(f"\nSaving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_positions, f)

    # Get file size
    file_size = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✓ Saved {len(all_positions):,} positions ({file_size:.1f} MB)")

    print(f"\n{'='*70}")
    print(f"Training data is ready!")
    print(f"Next: python train_on_gpu.py --config config_v2.yaml")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
