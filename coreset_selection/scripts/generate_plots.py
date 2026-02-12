#!/usr/bin/env python
"""
Generate plots and tables from experiment results.

Usage:
    python -m coreset_selection.scripts.generate_plots
    python -m coreset_selection.scripts.generate_plots --runs-dir runs_out --output-dir artifacts_out
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots and tables from experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Generate all plots (uses default directories)
    python -m coreset_selection.scripts.generate_plots

    # Specify custom directories
    python -m coreset_selection.scripts.generate_plots --runs-dir my_runs --output-dir my_plots

    # Only generate figures (skip tables)
    python -m coreset_selection.scripts.generate_plots --figures-only
""",
    )
    parser.add_argument(
        "--runs-dir",
        default="runs_out",
        help="Directory containing experiment results (default: runs_out)",
    )
    parser.add_argument(
        "--cache-dir",
        default="replicate_cache",
        help="Directory containing replicate caches (default: replicate_cache)",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts_out",
        help="Output directory for plots and tables (default: artifacts_out)",
    )
    parser.add_argument(
        "--figures-only",
        action="store_true",
        help="Only generate figures, skip tables",
    )
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help="Only generate tables, skip figures",
    )

    args = parser.parse_args()

    # Check if runs directory exists
    if not os.path.exists(args.runs_dir):
        print(f"Error: Runs directory '{args.runs_dir}' does not exist.")
        print(f"Make sure you have run some experiments first.")
        sys.exit(1)

    # Import here to avoid slow startup if just checking --help
    from ..artifacts.generator import ManuscriptArtifactGenerator

    print(f"=" * 60)
    print(f"GENERATING PLOTS AND TABLES")
    print(f"=" * 60)
    print(f"  Runs directory:   {args.runs_dir}")
    print(f"  Cache directory:  {args.cache_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"=" * 60)

    try:
        generator = ManuscriptArtifactGenerator(
            runs_root=args.runs_dir,
            cache_root=args.cache_dir,
            out_dir=args.output_dir,
        )

        generated = generator.generate_all()

        print(f"\n[SUCCESS] Generated artifacts:")
        print(f"  Figures: {len(generated.get('figures', []))}")
        for fig in generated.get("figures", []):
            print(f"    - {fig}")
        print(f"  Tables:  {len(generated.get('tables', []))}")
        for tbl in generated.get("tables", []):
            print(f"    - {tbl}")

        print(f"\nOutput saved to: {args.output_dir}/")
        return True

    except Exception as e:
        print(f"\n[ERROR] Failed to generate artifacts: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
