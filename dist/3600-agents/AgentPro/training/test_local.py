#!/usr/bin/env python3
"""
Local Test Runner for AgentPro Training Pipeline

This script automates the entire local testing process:
1. Checks Python version
2. Generates test data (100 positions)
3. Runs training (20 epochs on CPU)
4. Verifies outputs

Run: python test_local.py
"""

import sys
import subprocess
import time
from pathlib import Path


def print_banner(text):
    """Print a nice banner"""
    width = 60
    print(f"\n{'='*width}")
    print(f"  {text}")
    print(f"{'='*width}\n")


def check_python_version():
    """Check Python version is 3.10+"""
    print_banner("Step 1: Checking Python Version")

    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("\n❌ ERROR: Python 3.10+ is required")
        print("   The game code uses 'match' statements which require Python 3.10+")
        print("\n   Install Python 3.10+:")
        print("   - macOS: brew install python@3.10")
        print("   - Linux: sudo apt install python3.10")
        print("   - Windows: Download from python.org")
        return False

    print("✓ Python version OK")
    return True


def check_dependencies():
    """Check required dependencies are installed"""
    print_banner("Step 2: Checking Dependencies")

    required = {
        'torch': 'PyTorch',
        'numpy': 'NumPy',
        'yaml': 'PyYAML'
    }

    missing = []

    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name} installed")
        except ImportError:
            print(f"❌ {name} NOT installed")
            missing.append(name)

    if missing:
        print(f"\n❌ Missing dependencies: {', '.join(missing)}")
        print("\nInstall with:")
        print("  pip install torch numpy pyyaml")
        print("\nFor CPU-only PyTorch (faster install):")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
        return False

    # Note about optional tensorboard (not required for local testing)
    try:
        __import__('tensorboard')
        print("✓ TensorBoard installed (optional)")
    except ImportError:
        print("ℹ TensorBoard not installed (optional, not needed for local testing)")

    return True


def generate_test_data():
    """Run data generation"""
    print_banner("Step 3: Generating Test Data (100 positions)")

    print("This will take ~2-5 minutes on a modern laptop...")
    print("Using 2 workers, search depth 4\n")

    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, 'generate_data_local.py'],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Data generation completed in {elapsed:.1f} seconds")

        # Check if file was created
        if Path('training_data_local.json').exists():
            size = Path('training_data_local.json').stat().st_size
            print(f"✓ Created training_data_local.json ({size/1024:.1f} KB)")
            return True
        else:
            print("❌ training_data_local.json was not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Data generation failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def run_training():
    """Run training"""
    print_banner("Step 4: Running Training (20 epochs on CPU)")

    print("This will take ~5-10 minutes...")
    print("Training on CPU (GPU training happens on PACE)\n")

    start_time = time.time()

    try:
        subprocess.run(
            [sys.executable, 'train_on_gpu.py', '--config', 'config_local.yaml'],
            check=True,
            capture_output=False,
            text=True
        )

        elapsed = time.time() - start_time
        print(f"\n✓ Training completed in {elapsed/60:.1f} minutes")

        # Check if model was created
        if Path('best_evaluator.pth').exists():
            size = Path('best_evaluator.pth').stat().st_size
            print(f"✓ Created best_evaluator.pth ({size/1024:.1f} KB)")
            return True
        else:
            print("❌ best_evaluator.pth was not created")
            return False

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return False


def verify_outputs():
    """Verify all expected files exist"""
    print_banner("Step 5: Verifying Outputs")

    expected_files = {
        'training_data_local.json': 'Training data',
        'best_evaluator.pth': 'Trained model'
    }

    all_good = True

    for filename, description in expected_files.items():
        if Path(filename).exists():
            size = Path(filename).stat().st_size
            print(f"✓ {description}: {filename} ({size/1024:.1f} KB)")
        else:
            print(f"❌ {description}: {filename} NOT FOUND")
            all_good = False

    return all_good


def print_summary(success):
    """Print final summary"""
    print_banner("Test Summary")

    if success:
        print("🎉 SUCCESS! All local tests passed!")
        print("\nWhat was tested:")
        print("  ✓ Import paths work correctly")
        print("  ✓ Multiprocessing workers can find modules")
        print("  ✓ Board/Chicken initialization works")
        print("  ✓ Feature extraction works")
        print("  ✓ Minimax evaluation works")
        print("  ✓ Data is saved correctly")
        print("  ✓ Training script can load and train on data")
        print("\n✅ Your pipeline is ready for PACE!")
        print("\nNext steps:")
        print("  1. Commit your changes: git add . && git commit -m 'Ready for PACE'")
        print("  2. Push to GitHub: git push")
        print("  3. Follow PACE_INSTRUCTIONS.md to deploy to PACE")
        print("\nClean up test files:")
        print("  rm training_data_local.json best_evaluator.pth")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nCommon issues:")
        print("  - Python version < 3.10: Install Python 3.10+")
        print("  - Missing dependencies: pip install torch numpy pyyaml")
        print("  - Import errors: Make sure you're in the training/ directory")
        print("\nFor detailed troubleshooting, see LOCAL_TEST_INSTRUCTIONS.md")


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("  AgentPro Local Testing Pipeline")
    print("="*60)
    print("\nThis will test your training pipeline locally before")
    print("submitting to PACE. It takes ~10-15 minutes total.\n")

    input("Press Enter to start...")

    # Step 1: Check Python version
    if not check_python_version():
        print_summary(False)
        return 1

    # Step 2: Check dependencies
    if not check_dependencies():
        print_summary(False)
        return 1

    # Step 3: Generate data
    if not generate_test_data():
        print_summary(False)
        return 1

    # Step 4: Run training
    if not run_training():
        print_summary(False)
        return 1

    # Step 5: Verify outputs
    if not verify_outputs():
        print_summary(False)
        return 1

    # Success!
    print_summary(True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
