#!/usr/bin/env python3
"""
Build Dataset Script
Fetches data from Polygon and builds features/labels for training
"""
import argparse
import os
import sys

from data_loader import load_and_prepare_data
from features import FeatureEngineer
from labels import LabelGenerator, analyze_label_distribution


def main():
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument("--start-year", type=int, default=2004, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument("--output-dir", type=str, default="data", help="Output directory")
    parser.add_argument("--api-key", type=str, default=None, help="Polygon API key")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY environment variable not set")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SPY DAILY RANGE PLAN - DATASET BUILDER")
    print("=" * 60)
    print(f"Period: {args.start_year} - {args.end_year}")
    print(f"Output: {args.output_dir}")
    print("=" * 60)

    # Load and prepare data
    print("\n[1/4] Loading raw data from Polygon...")
    sessions, daily = load_and_prepare_data(
        api_key=api_key,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_dir=args.output_dir
    )

    # Build features
    print("\n[2/4] Building feature matrix...")
    engineer = FeatureEngineer()
    features_df = engineer.build_feature_matrix(sessions, daily)
    print(f"  Features shape: {features_df.shape}")

    # Build labels
    print("\n[3/4] Generating labels...")
    generator = LabelGenerator()
    labels_df, levels_df = generator.build_label_dataframe(sessions, daily)
    print(f"  Labels shape: {labels_df.shape}")

    # Analyze label distribution
    print("\n[4/4] Analyzing label distribution...")
    stats = analyze_label_distribution(labels_df)
    print("\n  Touch rates:")
    for target in ['touch_t1_long', 'touch_t2_long', 'touch_t3_long', 'touch_sl_long']:
        rate = stats.get(f'{target}_rate', 0)
        print(f"    {target}: {rate*100:.1f}%")

    # Save datasets
    features_path = os.path.join(args.output_dir, "features.parquet")
    labels_path = os.path.join(args.output_dir, "labels.parquet")
    levels_path = os.path.join(args.output_dir, "levels.parquet")
    daily_path = os.path.join(args.output_dir, "daily.parquet")

    features_df.to_parquet(features_path)
    labels_df.to_parquet(labels_path)
    levels_df.to_parquet(levels_path)
    daily.to_parquet(daily_path)

    print(f"\n{'=' * 60}")
    print("DATASET BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Features: {features_path}")
    print(f"Labels: {labels_path}")
    print(f"Levels: {levels_path}")
    print(f"Daily: {daily_path}")
    print(f"\nTotal sessions: {len(features_df)}")


if __name__ == "__main__":
    main()
