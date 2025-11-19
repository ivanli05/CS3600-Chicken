"""
Local Testing Version - Data Generation for AgentPro

Simplified version for testing on your laptop:
- Generates only 100 positions (instead of 50,000)
- Uses only 2 workers (instead of 32)
- Smaller search depth (4 instead of 6)
- Progress updates every 10 positions

Run: python generate_data_local.py
"""

import sys
import os
import time
import json
import random
from multiprocessing import Pool

# Add paths to import game engine and agent modules
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)  # AgentPro/
agents_dir = os.path.dirname(agentpro_dir)     # 3600-agents/
dist_dir = os.path.dirname(agents_dir)         # dist/
engine_dir = os.path.join(dist_dir, 'engine')  # dist/engine/

sys.path.insert(0, engine_dir)      # For 'from game.board import Board'
sys.path.insert(0, agents_dir)      # For 'from AgentPro.agent import PlayerAgent'

import numpy as np


def _setup_worker_paths():
    """
    Set up import paths for worker processes.
    This is needed because multiprocessing workers don't inherit sys.path modifications.
    """
    import sys
    import os

    # Calculate paths relative to this file
    training_dir = os.path.dirname(os.path.abspath(__file__))
    agentpro_dir = os.path.dirname(training_dir)
    agents_dir = os.path.dirname(agentpro_dir)
    dist_dir = os.path.dirname(agents_dir)
    engine_dir = os.path.join(dist_dir, 'engine')

    # Add to sys.path if not already there
    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)


def generate_single_position(args):
    """
    Generate one training example.
    Called by worker processes.
    """
    # CRITICAL: Set up paths in worker process
    _setup_worker_paths()

    # Now import modules (must be after path setup)
    from game.board import Board
    from game.game_map import GameMap
    from game.trapdoor_manager import TrapdoorManager
    from game.enums import MoveType
    from AgentPro.agent import PlayerAgent
    import numpy as np
    import random

    worker_id, position_id, depth, min_moves, max_moves = args

    try:
        # Generate random position
        game_map = GameMap()
        trapdoor_manager = TrapdoorManager(game_map)
        board = Board(game_map)

        # Initialize chicken starting positions
        spawns = trapdoor_manager.choose_spawns()
        board.chicken_player.start(spawns[0], 0)  # Player A (even chicken)
        board.chicken_enemy.start(spawns[1], 1)   # Player B (odd chicken)

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

        # Create agent for evaluation
        agent = PlayerAgent(board, lambda: 300.0)

        # Extract features
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

        # Evaluate with deep search
        score, _ = agent.search_engine._minimax(
            board=board,
            depth=depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing=True,
            time_left=30.0,
            trapdoor_tracker=agent.trapdoor_tracker
        )
        normalized_score = np.tanh(score / 1000.0)

        return {
            'features': features.tolist(),
            'score': float(normalized_score),
            'metadata': {
                'turn': board.turn_count,
                'eggs_diff': my_eggs - enemy_eggs,
                'worker_id': worker_id
            }
        }

    except Exception as e:
        # Print detailed error for debugging
        import traceback
        error_msg = f"Worker {worker_id} error on position {position_id}: {e}\n{traceback.format_exc()}"
        print(error_msg, flush=True)
        return None


def main():
    # Local testing configuration
    num_positions = 100      # Small dataset for testing
    depth = 4                # Shallower search for speed
    num_processes = 2        # Fewer workers for laptop
    min_moves = 5
    max_moves = 20           # Shorter games for speed
    output_file = 'training_data_local.json'

    print(f"{'='*60}")
    print(f"Local Testing - Data Generation")
    print(f"{'='*60}")
    print(f"Target positions: {num_positions}")
    print(f"Search depth: {depth}")
    print(f"Parallel workers: {num_processes}")
    print(f"Move range: {min_moves}-{max_moves}")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    # Prepare arguments for workers
    args_list = []
    for i in range(num_positions):
        worker_id = i % num_processes
        args_list.append((worker_id, i, depth, min_moves, max_moves))

    # Generate in parallel
    start_time = time.time()
    dataset = []

    print("Starting generation...\n")
    with Pool(processes=num_processes) as pool:
        # Process in chunks to show progress
        chunk_size = 10  # Update every 10 positions

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
                  f"Rate: {rate:.1f} pos/s | ETA: {eta:.0f}s")

    elapsed_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Generation complete!")
    print(f"{'='*60}")
    print(f"Generated: {len(dataset)} positions")
    print(f"Time: {elapsed_time:.1f} seconds")
    print(f"Rate: {len(dataset)/elapsed_time:.1f} positions/second")
    print(f"{'='*60}\n")

    # Calculate statistics
    if len(dataset) > 0:
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
        print(f"\nYou can now test training with:")
        print(f"  python train_on_gpu.py --config config_local.yaml")
    else:
        print("\n⚠ WARNING: No valid positions were generated!")
        print("Check the error messages above.")

    return dataset


if __name__ == '__main__':
    main()
