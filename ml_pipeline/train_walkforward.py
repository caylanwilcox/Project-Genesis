#!/usr/bin/env python3
"""
Walk-Forward Training Script
Trains models using expanding window validation
"""
import argparse
import os
import sys
import pandas as pd

from config import WalkForwardConfig, ALL_FEATURES
from train import WalkForwardTrainer
from calibrate import build_calibration_pipeline
from evaluate import run_full_evaluation


def main():
    parser = argparse.ArgumentParser(description="Train models with walk-forward validation")
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Model output directory")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Evaluation output directory")
    parser.add_argument("--test-years", type=str, default="2015-2024", help="Test years range (e.g., 2015-2024)")
    args = parser.parse_args()

    # Parse test years
    start_year, end_year = map(int, args.test_years.split("-"))
    test_years = list(range(start_year, end_year + 1))

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SPY DAILY RANGE PLAN - WALK-FORWARD TRAINING")
    print("=" * 60)
    print(f"Test years: {test_years}")
    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.model_dir}")
    print("=" * 60)

    # Load prepared data
    print("\n[1/5] Loading prepared datasets...")
    features_path = os.path.join(args.data_dir, "features.parquet")
    labels_path = os.path.join(args.data_dir, "labels.parquet")

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print("Error: Dataset not found. Run build_dataset.py first.")
        sys.exit(1)

    features_df = pd.read_parquet(features_path)
    labels_df = pd.read_parquet(labels_path)

    print(f"  Features: {features_df.shape}")
    print(f"  Labels: {labels_df.shape}")

    # Configure walk-forward
    wf_config = WalkForwardConfig()
    wf_config.test_years = test_years

    # Train models
    print("\n[2/5] Training models (walk-forward)...")
    trainer = WalkForwardTrainer(wf_config=wf_config)
    results = trainer.train_all_models(
        features_df,
        labels_df,
        output_dir=args.model_dir
    )

    # Calibrate probabilities
    print("\n[3/5] Calibrating probabilities...")
    calibrator = build_calibration_pipeline(
        results['walk_forward_results'],
        output_dir=args.model_dir
    )

    # Evaluate
    print("\n[4/5] Evaluating models...")
    eval_results = run_full_evaluation(
        results['walk_forward_results'],
        output_dir=args.output_dir
    )

    # Summary
    print("\n[5/5] Generating summary...")
    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Binary models trained: {len(results['binary_models'])}")
    print(f"Multiclass models trained: {len(results['multiclass_models'])}")
    print(f"Walk-forward years: {len(results['walk_forward_results'])}")
    print(f"\nModels saved to: {args.model_dir}")
    print(f"Evaluation saved to: {args.output_dir}")

    # Print key metrics
    print(f"\n{'=' * 60}")
    print("KEY METRICS (Aggregate)")
    print(f"{'=' * 60}")

    for target, metrics in eval_results['walk_forward_metrics'].items():
        agg = metrics.aggregate_metrics
        print(f"{target}: AUC={agg.auc_roc:.3f}, Brier={agg.brier:.3f}, ECE={agg.ece:.3f}")


if __name__ == "__main__":
    main()
