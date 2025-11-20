"""
Quick local test for data generation - generates just 10 positions to verify it works.

Run: python test_generation.py
"""

import sys
import os
import time
import json
from pathlib import Path

# Setup paths
training_dir = os.path.dirname(os.path.abspath(__file__))
agentpro_dir = os.path.dirname(training_dir)
agents_dir = os.path.dirname(agentpro_dir)
dist_dir = os.path.dirname(agents_dir)
engine_dir = os.path.join(dist_dir, 'engine')

sys.path.insert(0, engine_dir)
sys.path.insert(0, agents_dir)

# Import the generation function
from generate_data_ultimate import generate_position

def main():
    """Quick test - generate 10 positions"""

    print("="*70)
    print("LOCAL TEST - Generating 10 positions")
    print("="*70)
    print("This will help verify:")
    print("  1. No verbose output spam")
    print("  2. Generation actually works")
    print("  3. Speed per position")
    print("="*70)
    print()

    config = {
        'depth': 6,
        'min_moves': 8,
        'max_moves': 30,
        'move_variety': 'weighted'
    }

    positions_to_test = 10
    successful = 0
    failed = 0

    print(f"Generating {positions_to_test} test positions (depth={config['depth']})...")
    print()

    start_time = time.time()

    for i in range(positions_to_test):
        pos_start = time.time()

        # Generate one position
        args = (0, i, config)
        result = generate_position(args)

        pos_time = time.time() - pos_start

        if result is not None:
            successful += 1
            status = "✓"
            score = result['score']
            raw_score = result['metadata']['raw_score']
            eggs_diff = result['metadata']['eggs_diff']
            print(f"  {status} Position {i+1:2d}: {pos_time:.2f}s | Score: {score:+.3f} (raw: {raw_score:+6.0f}) | Eggs: {eggs_diff:+2d}")
        else:
            failed += 1
            print(f"  ✗ Position {i+1:2d}: {pos_time:.2f}s | FAILED")

    elapsed = time.time() - start_time

    print()
    print("="*70)
    print(f"Test complete!")
    print(f"  Successful: {successful}/{positions_to_test}")
    print(f"  Failed: {failed}/{positions_to_test}")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  Average: {elapsed/positions_to_test:.2f}s per position")
    print(f"  Rate: {positions_to_test/elapsed:.2f} positions/sec")
    print("="*70)
    print()

    if successful > 0:
        # Estimate time for full run
        avg_time = elapsed / successful
        full_time_100k = avg_time * 100000 / 3600  # hours

        print("Estimates for full PACE run:")
        print(f"  100k positions @ depth 6: ~{full_time_100k:.1f} hours (single core)")
        print(f"  With 32 cores: ~{full_time_100k/32:.1f} hours")
        print()

        if full_time_100k / 32 <= 8:
            print("✓ Should fit in 8-hour PACE window!")
        else:
            print(f"⚠ May exceed 8 hours - consider reducing to {int(100000 * 8 / (full_time_100k / 32))}k positions")

    print()
    print("If this looks good, submit to PACE with:")
    print("  cd dist/3600-agents/AgentPro/training")
    print("  sbatch generate_data_ultimate_job.sbatch")
    print()


if __name__ == '__main__':
    main()
