"""
Training Monitor for AgentB

Displays real-time training progress, plots loss curves,
and estimates completion time.

Usage:
    python monitor_training.py
    python monitor_training.py --log-file logs/train_12345.out
"""

import argparse
import re
import time
from pathlib import Path
from datetime import datetime, timedelta

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Install with: pip install matplotlib")


def parse_log_file(log_file):
    """
    Parse training log file and extract metrics.
    """
    if not Path(log_file).exists():
        return None

    epochs = []
    train_losses = []
    val_losses = []
    learning_rates = []

    # Pattern: Epoch 45/200 (1.2s) | Train: 0.0342 | Val: 0.0389 | LR: 0.000500
    pattern = r"Epoch (\d+)/(\d+) \([\d.]+s\) \| Train: ([\d.]+) \| Val: ([\d.]+) \| LR: ([\d.]+)"

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                epoch = int(match.group(1))
                max_epochs = int(match.group(2))
                train_loss = float(match.group(3))
                val_loss = float(match.group(4))
                lr = float(match.group(5))

                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                learning_rates.append(lr)

    if not epochs:
        return None

    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'learning_rates': learning_rates,
        'max_epochs': max_epochs,
        'current_epoch': epochs[-1] if epochs else 0
    }


def plot_training_progress(data, output_file='training_progress.png'):
    """
    Plot training and validation loss curves.
    """
    if not MATPLOTLIB_AVAILABLE:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Loss curves
    ax1.plot(data['epochs'], data['train_losses'], label='Train Loss', linewidth=2)
    ax1.plot(data['epochs'], data['val_losses'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Progress', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Learning rate
    ax2.plot(data['epochs'], data['learning_rates'], color='orange', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Learning Rate', fontsize=12)
    ax2.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Plot saved: {output_file}")


def estimate_completion(data):
    """
    Estimate training completion time.
    """
    if len(data['epochs']) < 2:
        return None

    current_epoch = data['current_epoch']
    max_epochs = data['max_epochs']
    remaining_epochs = max_epochs - current_epoch

    # Calculate average time per epoch (from recent epochs)
    recent_epochs = min(10, len(data['epochs']))
    # This is a rough estimate; actual timing would need timestamps

    return remaining_epochs


def print_summary(data):
    """
    Print training summary.
    """
    print(f"\n{'='*60}")
    print(f"Training Summary")
    print(f"{'='*60}")
    print(f"Current Epoch: {data['current_epoch']}/{data['max_epochs']}")
    print(f"Progress: {data['current_epoch']/data['max_epochs']*100:.1f}%")
    print()

    print(f"Latest Metrics:")
    print(f"  Train Loss: {data['train_losses'][-1]:.4f}")
    print(f"  Val Loss:   {data['val_losses'][-1]:.4f}")
    print(f"  LR:         {data['learning_rates'][-1]:.6f}")
    print()

    # Best validation loss
    best_val_loss = min(data['val_losses'])
    best_epoch = data['epochs'][data['val_losses'].index(best_val_loss)]
    print(f"Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
    print()

    # Improvement
    if len(data['val_losses']) >= 10:
        recent_avg = sum(data['val_losses'][-10:]) / 10
        early_avg = sum(data['val_losses'][:10]) / 10
        improvement = (early_avg - recent_avg) / early_avg * 100
        print(f"Improvement: {improvement:+.1f}% (last 10 epochs vs first 10)")

    print(f"{'='*60}\n")


def monitor_live(log_file=None, interval=30):
    """
    Monitor training in real-time.
    """
    if log_file is None:
        # Find most recent log file
        log_dir = Path('logs')
        if not log_dir.exists():
            print("No logs directory found!")
            return

        log_files = list(log_dir.glob('train_*.out'))
        if not log_files:
            print("No training log files found!")
            return

        log_file = max(log_files, key=lambda p: p.stat().st_mtime)

    print(f"Monitoring: {log_file}")
    print(f"Update interval: {interval} seconds")
    print(f"Press Ctrl+C to stop\n")

    try:
        while True:
            data = parse_log_file(log_file)

            if data:
                print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Update:")
                print_summary(data)

                if MATPLOTLIB_AVAILABLE:
                    plot_training_progress(data)
                    print("  Updated: training_progress.png")
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for data...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor AgentB training')
    parser.add_argument('--log-file', type=str, default=None,
                        help='Training log file to monitor')
    parser.add_argument('--interval', type=int, default=30,
                        help='Update interval in seconds')
    parser.add_argument('--once', action='store_true',
                        help='Run once and exit (no live monitoring)')

    args = parser.parse_args()

    if args.once:
        # Single snapshot
        if args.log_file is None:
            log_dir = Path('logs')
            log_files = list(log_dir.glob('train_*.out'))
            if not log_files:
                print("No log files found!")
                return
            args.log_file = max(log_files, key=lambda p: p.stat().st_mtime)

        data = parse_log_file(args.log_file)
        if data:
            print_summary(data)
            if MATPLOTLIB_AVAILABLE:
                plot_training_progress(data)
        else:
            print("No training data found in log file")
    else:
        # Live monitoring
        monitor_live(args.log_file, args.interval)


if __name__ == '__main__':
    main()
