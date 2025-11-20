"""
ULTIMATE Training Data Generation for AgentPro Neural Network

This generates high-quality training data with:
1. Proper value calculation (no broken tanh compression)
2. Complete 128-feature extraction
3. Superior heuristics from AgentProNoNets
4. Diverse, realistic positions
5. Deep minimax evaluation

Run: python generate_data_ultimate.py

For PACE: sbatch generate_data_ultimate_job.sbatch
"""

import sys
import os
import time
import json
import random
from multiprocessing import Pool
from pathlib import Path

# Setup paths
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)
agents_dir = os.path.dirname(agentpro_dir)
dist_dir = os.path.dirname(agents_dir)
engine_dir = os.path.join(dist_dir, 'engine')

sys.path.insert(0, engine_dir)
sys.path.insert(0, agents_dir)

import numpy as np


def _setup_worker_paths():
    """Set up paths in worker processes"""
    import sys
    import os

    training_dir = os.path.dirname(os.path.abspath(__file__))
    agentpro_dir = os.path.dirname(training_dir)
    agents_dir = os.path.dirname(agentpro_dir)
    dist_dir = os.path.dirname(agents_dir)
    engine_dir = os.path.join(dist_dir, 'engine')

    if engine_dir not in sys.path:
        sys.path.insert(0, engine_dir)
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)


def generate_position(args):
    """
    Generate ONE training sample (position + features + value).

    This is called by worker processes in parallel.
    """
    _setup_worker_paths()

    from game.board import Board
    from game.game_map import GameMap
    from game.trapdoor_manager import TrapdoorManager
    from game.enums import MoveType
    from AgentPro.agent import PlayerAgent
    from AgentPro.feature_extractor import FeatureExtractor
    import numpy as np
    import random

    worker_id, position_id, config = args

    try:
        # Unpack configuration
        depth_for_labels = config['depth']
        min_moves = config['min_moves']
        max_moves = config['max_moves']
        move_variety = config.get('move_variety', 'weighted')  # weighted, uniform, or smart

        # Generate random game position
        game_map = GameMap()
        trapdoor_manager = TrapdoorManager(game_map)
        board = Board(game_map)

        # Initialize chickens
        spawns = trapdoor_manager.choose_spawns()
        board.chicken_player.start(spawns[0], 0)
        board.chicken_enemy.start(spawns[1], 1)

        # Play random moves to get to a mid-game position
        num_moves = random.randint(min_moves, max_moves)

        for move_num in range(num_moves):
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break

            # Choose move based on strategy
            if move_variety == 'weighted':
                # Prefer egg moves (more realistic)
                move_weights = []
                for move in valid_moves:
                    if move[1] == MoveType.EGG:
                        move_weights.append(3.0)
                    elif move[1] == MoveType.TURD:
                        move_weights.append(1.0)
                    else:
                        move_weights.append(2.0)  # Plain moves common too
                total = sum(move_weights)
                move_probs = [w / total for w in move_weights]
                move = random.choices(valid_moves, weights=move_probs, k=1)[0]

            elif move_variety == 'smart':
                # Use light heuristic guidance (more realistic positions)
                # Create temp agent for quick evaluation
                temp_agent = PlayerAgent(board, lambda: 300.0)
                move_scores = [
                    (temp_agent.move_evaluator.quick_evaluate_move(
                        m, board, temp_agent.trapdoor_tracker
                    ), m)
                    for m in valid_moves
                ]
                # Add randomness - pick from top 50%
                move_scores.sort(reverse=True)
                top_half = max(1, len(move_scores) // 2)
                move = random.choice([m for _, m in move_scores[:top_half]])

            else:  # uniform
                move = random.choice(valid_moves)

            # Make the move
            board = board.forecast_move(move[0], move[1], check_ok=False)
            if board is None:
                return None

            board.reverse_perspective()

        # Skip if game is over
        if board.is_game_over():
            return None

        # Skip very early positions (not interesting)
        if num_moves < min_moves:
            return None

        # Create agent for evaluation and feature extraction
        agent = PlayerAgent(board, lambda: 300.0)

        # Extract features (128 dimensions)
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(board, agent.trapdoor_tracker)

        # Evaluate position with DEEP minimax search
        # This is our "ground truth" - what should the network learn
        score = evaluate_position_deep(board, agent, depth_for_labels)

        # CRITICAL: Proper value normalization!
        # Don't use tanh compression - preserve relative magnitudes
        normalized_score = normalize_score(score, board)

        # Metadata for debugging
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        turn_count = getattr(board, 'turn_count', num_moves)

        return {
            'features': features.tolist(),
            'score': float(normalized_score),
            'metadata': {
                'raw_score': float(score),
                'turn': turn_count,
                'my_eggs': my_eggs,
                'enemy_eggs': enemy_eggs,
                'eggs_diff': my_eggs - enemy_eggs,
                'worker_id': worker_id,
                'depth_evaluated': depth_for_labels
            }
        }

    except Exception:
        # Silently skip failed positions
        return None


def evaluate_position_deep(board, agent, depth=8):
    """
    Evaluate position using DEEP minimax with IMPROVED heuristics.

    This is the ground truth label for training.
    """
    try:
        # Use agent's search engine with improved heuristics
        score, _ = agent.search_engine._minimax(
            board=board,
            depth=depth,
            alpha=float('-inf'),
            beta=float('inf'),
            maximizing=True,
            time_left=60.0,  # Generous time for data generation
            trapdoor_tracker=agent.trapdoor_tracker
        )

        return score

    except Exception:
        # Fallback: use simple material count
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        return (my_eggs - enemy_eggs) * 300.0  # Egg value from improved heuristics


def normalize_score(score, board):
    """
    Properly normalize score for training.

    CRITICAL FIX: Don't use tanh compression!
    Preserve relative magnitudes while keeping values trainable.

    Score ranges (with improved heuristics):
    - Material: ±300 per egg → ±3000 for 10-egg lead
    - Mobility: ±5 per move → ±40
    - Positional: ±200
    - Trapdoors: Can add/subtract thousands

    Typical range: [-4000, +4000] for extreme positions
    Normal range: [-1500, +1500]
    """
    try:
        my_eggs = board.chicken_player.get_eggs_laid()
        enemy_eggs = board.chicken_enemy.get_eggs_laid()
        egg_diff = my_eggs - enemy_eggs

        # For extreme material advantages, use sign directly
        if abs(egg_diff) >= 10:
            # Huge material lead - almost certainly winning/losing
            return np.clip(np.sign(egg_diff) * 1.5, -2.0, 2.0)

        # For normal positions, scale by reasonable factor
        # Map typical range [-1500, +1500] to roughly [-1.0, +1.0]
        # But allow outliers up to ±2.0
        normalized = score / 1500.0

        # Clip to prevent extreme outliers
        normalized = np.clip(normalized, -2.5, 2.5)

        return normalized

    except Exception:
        # Fallback: use tanh but with better scaling
        return np.tanh(score / 800.0)


def main():
    """Generate training data"""

    # Configuration
    config = {
        'num_positions': 350000,  # 350k positions - optimized for 8hr PACE window
        'depth': 7,              # Depth 7 for high quality labels (~5.6 hrs on PACE)
        'min_moves': 8,          # Skip very early positions
        'max_moves': 30,         # Don't go to endgame (boring)
        'move_variety': 'weighted',  # weighted, uniform, or smart
        'num_processes': 32,     # Match SBATCH CPU allocation
        'output_file': 'training_data_ultimate.json'
    }

    print(f"{'='*70}")
    print(f"ULTIMATE Training Data Generation")
    print(f"{'='*70}")
    print(f"Target positions: {config['num_positions']:,}")
    print(f"Search depth for labels: {config['depth']}")
    print(f"Move range: {config['min_moves']}-{config['max_moves']}")
    print(f"Move variety: {config['move_variety']}")
    print(f"Parallel workers: {config['num_processes']}")
    print(f"Output: {config['output_file']}")
    print(f"{'='*70}\n")

    print("⚠ This will take several hours. Consider running on PACE with GPU job.")
    print()

    # Prepare worker arguments
    args_list = []
    for i in range(config['num_positions']):
        worker_id = i % config['num_processes']
        args_list.append((worker_id, i, config))

    # Generate data
    start_time = time.time()
    dataset = []

    print("Starting generation...\n")

    with Pool(processes=config['num_processes']) as pool:
        # Process in chunks for progress updates
        chunk_size = 100

        for i in range(0, len(args_list), chunk_size):
            chunk = args_list[i:i+chunk_size]
            results = pool.map(generate_position, chunk)

            # Filter out None results
            valid_results = [r for r in results if r is not None]
            dataset.extend(valid_results)

            # Progress update
            if len(dataset) % 500 == 0 or i + chunk_size >= len(args_list):
                elapsed = time.time() - start_time
                progress = len(dataset) / config['num_positions'] * 100
                rate = len(dataset) / elapsed if elapsed > 0 else 0
                eta = (config['num_positions'] - len(dataset)) / rate if rate > 0 else 0

                print(f"Progress: {len(dataset):,}/{config['num_positions']:,} ({progress:.1f}%) | "
                      f"Rate: {rate:.1f} pos/s | ETA: {eta/60:.1f}min | "
                      f"Elapsed: {elapsed/60:.1f}min")

    elapsed_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Generation complete!")
    print(f"Positions generated: {len(dataset):,}")
    print(f"Success rate: {len(dataset)/config['num_positions']*100:.1f}%")
    print(f"Total time: {elapsed_time/60:.1f} minutes ({elapsed_time/3600:.2f} hours)")
    print(f"Average: {len(dataset)/elapsed_time:.2f} positions/second")
    print(f"{'='*70}\n")

    # Analyze dataset
    analyze_dataset(dataset)

    # Save dataset
    print(f"Saving to {config['output_file']}...")
    output_path = Path(config['output_file'])
    with open(output_path, 'w') as f:
        json.dump(dataset, f)

    file_size = output_path.stat().st_size / (1024 * 1024)
    print(f"✓ Saved {len(dataset):,} positions ({file_size:.1f} MB)")

    print(f"\n{'='*70}")
    print(f"Training data is ready!")
    print(f"Next: python train_on_gpu.py --config config_ultimate.yaml")
    print(f"{'='*70}\n")


def analyze_dataset(dataset):
    """Analyze generated dataset for quality"""
    print(f"\n{'='*70}")
    print("Dataset Analysis")
    print(f"{'='*70}")

    scores = [d['score'] for d in dataset]
    raw_scores = [d['metadata']['raw_score'] for d in dataset]
    eggs_diffs = [d['metadata']['eggs_diff'] for d in dataset]
    turns = [d['metadata']['turn'] for d in dataset]

    print(f"Normalized scores:")
    print(f"  Mean: {np.mean(scores):.4f}")
    print(f"  Std:  {np.std(scores):.4f}")
    print(f"  Min:  {np.min(scores):.4f}")
    print(f"  Max:  {np.max(scores):.4f}")
    print(f"  25%:  {np.percentile(scores, 25):.4f}")
    print(f"  50%:  {np.percentile(scores, 50):.4f}")
    print(f"  75%:  {np.percentile(scores, 75):.4f}")

    print(f"\nRaw scores (heuristic):")
    print(f"  Mean: {np.mean(raw_scores):.1f}")
    print(f"  Std:  {np.std(raw_scores):.1f}")
    print(f"  Min:  {np.min(raw_scores):.1f}")
    print(f"  Max:  {np.max(raw_scores):.1f}")

    print(f"\nEgg differences:")
    print(f"  Mean: {np.mean(eggs_diffs):.2f}")
    print(f"  Std:  {np.std(eggs_diffs):.2f}")
    print(f"  Min:  {np.min(eggs_diffs)}")
    print(f"  Max:  {np.max(eggs_diffs)}")

    print(f"\nTurn distribution:")
    print(f"  Mean: {np.mean(turns):.1f}")
    print(f"  Std:  {np.std(turns):.1f}")
    print(f"  Min:  {np.min(turns)}")
    print(f"  Max:  {np.max(turns)}")

    # Check score distribution
    score_histogram = {
        'Very negative (< -1.0)': sum(1 for s in scores if s < -1.0),
        'Negative (-1.0 to -0.3)': sum(1 for s in scores if -1.0 <= s < -0.3),
        'Slight negative (-0.3 to -0.1)': sum(1 for s in scores if -0.3 <= s < -0.1),
        'Balanced (-0.1 to +0.1)': sum(1 for s in scores if -0.1 <= s <= 0.1),
        'Slight positive (+0.1 to +0.3)': sum(1 for s in scores if 0.1 < s <= 0.3),
        'Positive (+0.3 to +1.0)': sum(1 for s in scores if 0.3 < s <= 1.0),
        'Very positive (> +1.0)': sum(1 for s in scores if s > 1.0),
    }

    print(f"\nScore distribution:")
    for label, count in score_histogram.items():
        pct = count / len(scores) * 100
        bar = '█' * int(pct / 2)
        print(f"  {label:25s}: {count:6,} ({pct:5.1f}%) {bar}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
