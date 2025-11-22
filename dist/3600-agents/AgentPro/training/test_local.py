"""
Quick local test to verify data generation works
Generates just 10 positions to test the pipeline
"""

import sys
import os

# Add paths
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)
engine_dir = os.path.join(os.path.dirname(os.path.dirname(agentpro_dir)), 'engine')

sys.path.insert(0, agentpro_dir)
sys.path.insert(0, engine_dir)

import numpy as np
from game.board import Board
from game.game_map import GameMap
from game.trapdoor_manager import TrapdoorManager
from game.enums import MoveType
from heuristics import MoveEvaluator
from feature_extractor import FeatureExtractor


def generate_one_position(seed=42):
    """Generate a single test position"""
    np.random.seed(seed)

    # Create board
    game_map = GameMap()
    trapdoor_manager = TrapdoorManager(game_map)
    board = Board(game_map)

    # Initialize chickens
    spawns = trapdoor_manager.choose_spawns()
    board.chicken_player.start(spawns[0], 0)
    board.chicken_enemy.start(spawns[1], 1)

    # Play 15 random moves
    num_moves = 15
    for i in range(num_moves):
        valid_moves = board.get_valid_moves()
        if not valid_moves or board.is_game_over():
            break

        # Weighted random move (favor eggs)
        move_weights = []
        for move in valid_moves:
            if move[1] == MoveType.EGG:
                move_weights.append(0.6)
            elif move[1] == MoveType.PLAIN:
                move_weights.append(0.3)
            else:
                move_weights.append(0.1)

        move_weights = np.array(move_weights)
        move_weights = move_weights / move_weights.sum()

        move_idx = np.random.choice(len(valid_moves), p=move_weights)
        move = valid_moves[move_idx]

        # Apply move
        board = board.forecast_move(move[0], move[1], check_ok=False)
        if board is None:
            print(f"Move {i+1} failed, stopping")
            return None

        board.reverse_perspective()

    if board.is_game_over():
        print("Game ended early")
        return None

    # Extract features
    feature_extractor = FeatureExtractor()
    features = feature_extractor.extract_features(board, trapdoor_tracker=None)

    # Quick evaluation (no deep minimax for test)
    evaluator = MoveEvaluator()
    score = evaluator.evaluate_position(board)

    # Normalize
    normalized_score = np.clip(score / 1500.0, -2.0, 2.0)

    return {
        'features': features.tolist(),
        'score': float(normalized_score),
        'raw_score': float(score),
        'moves_played': num_moves
    }


def main():
    print("="*60)
    print("Local Test - Generate 10 Positions")
    print("="*60)

    positions = []

    for i in range(10):
        print(f"\nGenerating position {i+1}/10...")

        position = generate_one_position(seed=42 + i)

        if position:
            positions.append(position)
            print(f"  ✓ Score: {position['score']:.3f} (raw: {position['raw_score']:.1f})")
            print(f"  ✓ Features: {len(position['features'])} dimensions")
        else:
            print(f"  ✗ Failed")

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Generated: {len(positions)}/10 positions")
    print(f"  Success rate: {len(positions)/10*100:.0f}%")

    if positions:
        scores = [p['score'] for p in positions]
        print(f"\nScore distribution:")
        print(f"  Mean: {np.mean(scores):.3f}")
        print(f"  Std:  {np.std(scores):.3f}")
        print(f"  Min:  {np.min(scores):.3f}")
        print(f"  Max:  {np.max(scores):.3f}")

    print(f"{'='*60}")

    if len(positions) >= 8:
        print("\n✓ Test PASSED! Data generation is working.")
        print("  You can now run: sbatch generate_data_v2_job.sbatch")
    else:
        print("\n✗ Test FAILED! Fix issues before running on PACE.")

    print()


if __name__ == '__main__':
    main()
