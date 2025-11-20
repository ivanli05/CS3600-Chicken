"""
Benchmark script to accurately measure data generation time.

Tests different depths to help you choose the right configuration.

Run: python benchmark_generation.py
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

from generate_data_ultimate import generate_position


def benchmark_depth(depth, num_samples=50):
    """Benchmark a specific depth"""
    print(f"\n{'='*70}")
    print(f"Benchmarking DEPTH {depth}")
    print(f"{'='*70}")
    print(f"Generating {num_samples} positions to get accurate average...")
    print()

    config = {
        'depth': depth,
        'min_moves': 8,
        'max_moves': 30,
        'move_variety': 'weighted'
    }

    times = []
    successful = 0
    failed = 0

    # Skip first 3 for initialization overhead
    print("Warming up (3 positions, not counted)...")
    for i in range(3):
        args = (0, i, config)
        generate_position(args)

    print(f"Benchmarking {num_samples} positions...")
    for i in range(num_samples):
        start = time.time()
        args = (0, i, config)
        result = generate_position(args)
        elapsed = time.time() - start

        if result is not None:
            successful += 1
            times.append(elapsed)
            if (i + 1) % 10 == 0:
                avg_so_far = sum(times) / len(times)
                print(f"  {i+1}/{num_samples}: {elapsed:.2f}s (avg so far: {avg_so_far:.2f}s)")
        else:
            failed += 1

    if successful == 0:
        print("ERROR: All positions failed!")
        return None

    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    median_time = sorted(times)[len(times) // 2]

    print()
    print("Results:")
    print(f"  Successful: {successful}/{num_samples}")
    print(f"  Failed: {failed}/{num_samples}")
    print(f"  Average time: {avg_time:.3f}s per position")
    print(f"  Median time: {median_time:.3f}s")
    print(f"  Min time: {min_time:.3f}s")
    print(f"  Max time: {max_time:.3f}s")

    return avg_time


def main():
    """Run benchmarks and show estimates"""

    print("="*70)
    print("DATA GENERATION BENCHMARK")
    print("="*70)
    print()
    print("This will test depths 6, 7, and 8 to help you choose.")
    print("Each test generates 50 positions (after 3 warmup).")
    print("This will take ~2-5 minutes total.")
    print()
    input("Press ENTER to start...")

    results = {}

    # Test depth 6
    results[6] = benchmark_depth(6, num_samples=50)

    # Test depth 7
    results[7] = benchmark_depth(7, num_samples=50)

    # Test depth 8 (fewer samples since it's slower)
    print(f"\n{'='*70}")
    print("Depth 8 is MUCH slower - testing with only 20 samples...")
    print(f"{'='*70}")
    results[8] = benchmark_depth(8, num_samples=20)

    # Summary
    print()
    print("="*70)
    print("SUMMARY & RECOMMENDATIONS")
    print("="*70)
    print()

    configs = [
        (100000, 6),
        (150000, 6),
        (200000, 6),
        (100000, 7),
        (150000, 7),
        (200000, 7),
        (50000, 8),
        (100000, 8),
        (150000, 8),
    ]

    print(f"{'Positions':<12} {'Depth':<8} {'Single Core':<15} {'32 Cores':<15} {'Recommendation'}")
    print("-" * 70)

    for positions, depth in configs:
        if results[depth] is None:
            continue

        single_core_hours = (positions * results[depth]) / 3600
        parallel_hours = single_core_hours / 32

        # Recommendation
        if parallel_hours <= 4:
            rec = "✓ Fast"
        elif parallel_hours <= 8:
            rec = "✓ Good fit (8hr)"
        elif parallel_hours <= 12:
            rec = "⚠ Needs 12hr"
        elif parallel_hours <= 24:
            rec = "⚠ Needs 24hr"
        else:
            rec = "✗ Too slow"

        print(f"{positions:<12,} {depth:<8} {single_core_hours:<14.1f}h {parallel_hours:<14.1f}h {rec}")

    print()
    print("="*70)
    print("DEPTH QUALITY vs SPEED:")
    print("  Depth 6: Good labels, fastest")
    print("  Depth 7: Better labels, ~3-4x slower")
    print("  Depth 8: Best labels, ~10-15x slower")
    print()
    print("RECOMMENDED CONFIGS:")

    # Find best configs
    for time_limit, label in [(8, "8-hour"), (12, "12-hour"), (24, "24-hour")]:
        print(f"\n  If you have {label} window:")
        best_found = False
        for positions, depth in reversed(configs):
            if results[depth] is None:
                continue
            parallel_hours = (positions * results[depth]) / 3600 / 32
            if parallel_hours <= time_limit * 0.9:  # Use 90% of time as buffer
                if not best_found:
                    print(f"    → {positions:,} positions @ depth {depth} ({parallel_hours:.1f} hours)")
                    best_found = True
                    break

    print()
    print("To update generate_data_ultimate.py, edit lines 257-264:")
    print("  'num_positions': <your choice>")
    print("  'depth': <your choice>")
    print("="*70)


if __name__ == '__main__':
    main()
