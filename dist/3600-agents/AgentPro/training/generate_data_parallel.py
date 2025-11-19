"""
Parallel Data Generation for AgentB Training

Generates training data using multiprocessing for speed.
Uses deep search (depth 6) as ground truth labels.
"""

import sys
import os
import time
import json
import yaml
import random
import argparse
from multiprocessing import Pool, Manager, cpu_count
from pathlib import Path

# Add paths to import game engine and agent modules
# From training/ we need to go up to dist/ then into engine/
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)  # AgentPro/
agents_dir = os.path.dirname(agentpro_dir)     # 3600-agents/
dist_dir = os.path.dirname(agents_dir)         # dist/
engine_dir = os.path.join(dist_dir, 'engine')  # dist/engine/

sys.path.insert(0, engine_dir)      # For 'from game import Board'
sys.path.insert(0, agents_dir)      # For 'from AgentPro.agent import PlayerAgent'

import numpy as np
from game import Board
from game.enums import Direction, MoveType
from AgentPro.agent import PlayerAgent


def generate_random_position(min_moves=5, max_moves=35):
    """
    Generate a random but legal game position.
    """
    try:
        board = Board()
        num_moves = random.randint(min_moves, max_moves)

        for _ in range(num_moves):
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break

            # Weighted random: prefer egg moves
            move_weights = []
            for move in valid_moves:
                if move[1] == MoveType.EGG:
                    move_weights.append(3.0)
                elif move[1] == MoveType.TURD:
                    move_weights.append(1.5)
                else:
                    move_weights.append(1.0)

            total = sum(move_weights)
            move_probs = [w / total for w in move_weights]

            move = random.choices(valid_moves, weights=move_probs, k=1)[0]

            board = board.forecast_move(move[0], move[1], check_ok=False)
            if board is None:
                return None

            board.reverse_perspective()

        if board.is_game_over():
            return None

        return board

    except Exception as e:
        return None


def evaluate_with_search(board, depth=6):
    """
    Evaluate position using deep search.
    This is our "ground truth" label.
    """
    try:
        # Create temporary agent
        agent = PlayerAgent(board, lambda: 300.0)

        # Deep search
        score, _ = agent.search_engine._minimax(
            board=board,
            depth=depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing=True,
            time_left=30.0,
            trapdoor_tracker=agent.trapdoor_tracker
        )

        # Normalize to [-1, 1]
        normalized_score = np.tanh(score / 1000.0)

        return normalized_score

    except Exception as e:
        return 0.0


def extract_features(board, agent):
    """
    Extract feature vector from board position.
    (Uses same feature extraction as agent)
    """
    try:
        features = np.zeros(128, dtype=np.float32)

        my_loc = board.chicken_player.get_location()
        enemy_loc = board.chicken_enemy.get_location()
        map_size = 8

        # Basic features
        features[0] = my_loc[0] / map_size
        features[1] = my_loc[1] / map_size
        features[2] = enemy_loc[0] / map_size
        features[3] = enemy_loc[1] / map_size

        # Egg counts
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        features[8] = my_eggs / 40.0
        features[9] = enemy_eggs / 40.0
        features[10] = (my_eggs - enemy_eggs) / 40.0

        # Turd counts
        my_turds = board.chicken_player.get_turds_left()
        enemy_turds = board.chicken_enemy.get_turds_left()
        features[12] = my_turds / 5.0
        features[13] = enemy_turds / 5.0

        # Mobility
        my_moves = len(board.get_valid_moves())
        board.reverse_perspective()
        enemy_moves = len(board.get_valid_moves())
        board.reverse_perspective()

        features[32] = my_moves / 8.0
        features[33] = enemy_moves / 8.0

        # Turn information
        features[36] = board.turn_count / 80.0
        features[37] = board.turns_left_player / 40.0

        # Center distance
        center = map_size / 2.0
        my_center_dist = (abs(my_loc[0] - center) + abs(my_loc[1] - center)) / map_size
        enemy_center_dist = (abs(enemy_loc[0] - center) + abs(enemy_loc[1] - center)) / map_size
        features[40] = my_center_dist
        features[41] = enemy_center_dist

        return features

    except Exception as e:
        return np.zeros(128, dtype=np.float32)


def generate_single_position(args):
    """
    Generate one training example.
    Called by worker processes.
    """
    worker_id, position_id, depth, min_moves, max_moves = args

    try:
        # Generate random position
        board = generate_random_position(min_moves, max_moves)
        if board is None:
            return None

        # Create agent for feature extraction
        agent = PlayerAgent(board, lambda: 300.0)

        # Extract features
        features = extract_features(board, agent)

        # Get ground truth from deep search
        score = evaluate_with_search(board, depth)

        return {
            'features': features.tolist(),
            'score': float(score),
            'metadata': {
                'turn': board.turn_count,
                'eggs_diff': board.chicken_player.get_eggs_laid() - board.chicken_enemy.get_eggs_laid(),
                'worker_id': worker_id
            }
        }

    except Exception as e:
        print(f"Error in worker {worker_id}: {e}")
        return None


def generate_dataset(config, output_file='training_data.json'):
    """
    Generate complete dataset using multiprocessing.
    """
    num_positions = config['data_generation']['num_positions']
    depth = config['data_generation']['depth_for_labels']
    num_processes = config['data_generation']['num_processes']
    min_moves = config['data_generation']['min_moves']
    max_moves = config['data_generation']['max_moves']

    print(f"{'='*60}")
    print(f"Parallel Data Generation")
    print(f"{'='*60}")
    print(f"Target positions: {num_positions}")
    print(f"Search depth: {depth}")
    print(f"Parallel workers: {num_processes}")
    print(f"Move range: {min_moves}-{max_moves}")
    print(f"{'='*60}\n")

    # Prepare arguments for workers
    args_list = []
    for i in range(num_positions):
        worker_id = i % num_processes
        args_list.append((worker_id, i, depth, min_moves, max_moves))

    # Generate in parallel
    start_time = time.time()
    dataset = []

    print("Starting parallel generation...")
    with Pool(processes=num_processes) as pool:
        # Process in chunks to show progress
        chunk_size = max(1, num_positions // 20)  # 5% chunks

        for i in range(0, len(args_list), chunk_size):
            chunk = args_list[i:i+chunk_size]

            results = pool.map(generate_single_position, chunk)

            # Filter out None results
            valid_results = [r for r in results if r is not None]
            dataset.extend(valid_results)

            # Progress
            progress = len(dataset) / num_positions * 100
            elapsed = time.time() - start_time
            rate = len(dataset) / elapsed if elapsed > 0 else 0
            eta = (num_positions - len(dataset)) / rate if rate > 0 else 0

            print(f"Progress: {len(dataset)}/{num_positions} ({progress:.1f}%) | "
                  f"Rate: {rate:.1f} pos/s | ETA: {eta/60:.1f} min")

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    print(f"Generated: {len(dataset)} positions")
    print(f"Time: {elapsed_time/60:.1f} minutes")
    print(f"Rate: {len(dataset)/elapsed_time:.1f} positions/second")
    print(f"{'='*60}\n")

    # Calculate statistics
    scores = [d['score'] for d in dataset]
    print(f"Dataset Statistics:")
    print(f"  Score mean: {np.mean(scores):.3f}")
    print(f"  Score std: {np.std(scores):.3f}")
    print(f"  Score range: [{np.min(scores):.3f}, {np.max(scores):.3f}]")
    print()

    # Save
    print(f"Saving to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(dataset, f)

    print(f"✓ Dataset saved!")

    return dataset


def main():
    parser = argparse.ArgumentParser(description='Generate training data in parallel')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--output', type=str, default='training_data.json',
                        help='Output file')
    parser.add_argument('--num-positions', type=int, default=None,
                        help='Override number of positions')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override number of workers')

    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override if specified
    if args.num_positions:
        config['data_generation']['num_positions'] = args.num_positions

    if args.num_workers:
        config['data_generation']['num_processes'] = args.num_workers

    # Generate
    generate_dataset(config, args.output)


if __name__ == '__main__':
    main()
