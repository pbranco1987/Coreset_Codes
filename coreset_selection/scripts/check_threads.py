#!/usr/bin/env python
"""
Quick script to verify thread configuration for parallel execution.

Run this BEFORE starting your experiments to verify thread limits are set correctly.

Usage:
    # Check what would happen with 10 parallel experiments
    python -m coreset_selection.scripts.check_threads --parallel-experiments 10
    
    # Check with explicit thread count
    CORESET_NUM_THREADS=20 python -m coreset_selection.scripts.check_threads
"""

import os
import sys


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check thread configuration")
    parser.add_argument("--parallel-experiments", type=int, default=None,
                        help="Number of parallel experiments")
    args = parser.parse_args()
    
    # Set env var if specified
    if args.parallel_experiments:
        os.environ["CORESET_PARALLEL_EXPERIMENTS"] = str(args.parallel_experiments)
    
    # Import after setting env vars
    from ..utils.torch_perf import configure_torch_threads, available_cpu_count
    
    n_cpus = available_cpu_count()
    print(f"Detected CPUs: {n_cpus}")
    print(f"CORESET_NUM_THREADS: {os.environ.get('CORESET_NUM_THREADS', 'not set')}")
    print(f"CORESET_PARALLEL_EXPERIMENTS: {os.environ.get('CORESET_PARALLEL_EXPERIMENTS', 'not set')}")
    print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print()
    
    n_threads, _ = configure_torch_threads(verbose=True)
    
    print()
    print(f"Configured threads: {n_threads}")
    print(f"OMP_NUM_THREADS after config: {os.environ.get('OMP_NUM_THREADS')}")
    
    try:
        import torch
        print(f"PyTorch num_threads: {torch.get_num_threads()}")
        print(f"PyTorch num_interop_threads: {torch.get_num_interop_threads()}")
    except ImportError:
        print("PyTorch not installed")
    
    print()
    if args.parallel_experiments:
        expected = max(1, int(n_cpus * 0.95) // args.parallel_experiments)
        print(f"For {args.parallel_experiments} parallel experiments on {n_cpus} cores:")
        print(f"  Expected threads per experiment: ~{expected}")
        print(f"  Actual configured: {n_threads}")
        if n_threads == expected or abs(n_threads - expected) <= 1:
            print("  ✓ Configuration looks correct!")
        else:
            print("  ⚠ Configuration may not be optimal")


if __name__ == "__main__":
    main()
