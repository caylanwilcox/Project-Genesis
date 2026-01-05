#!/usr/bin/env python3
"""
Today's Inference Script
Generates daily range plan for current trading session
"""
import argparse
import os
import sys
import json
from datetime import datetime

from inference import PlanGenerator, infer_today


def main():
    parser = argparse.ArgumentParser(description="Generate today's daily range plan")
    parser.add_argument("--model-dir", type=str, default="models", help="Model directory")
    parser.add_argument("--output", type=str, default=None, help="Output file (JSON)")
    parser.add_argument("--api-key", type=str, default=None, help="Polygon API key")
    parser.add_argument("--demo", action="store_true", help="Run with demo/mock data")
    args = parser.parse_args()

    if args.demo:
        # Demo mode with mock data
        print("=" * 60)
        print("SPY DAILY RANGE PLAN - DEMO MODE")
        print("=" * 60)

        from config import ALL_FEATURES
        import numpy as np

        # Create mock features
        np.random.seed(42)
        features = {feat: np.random.random() * 2 - 1 for feat in ALL_FEATURES}
        features['atr_14'] = 3.5
        features['atr_5'] = 4.0
        features['atr_ratio_5_14'] = 1.14
        features['rsi_14'] = 55
        features['day_of_week'] = datetime.now().weekday()
        features['month'] = datetime.now().month

        generator = PlanGenerator(model_dir=args.model_dir)
        plan = generator.generate_plan(
            features=features,
            vwap=475.50,
            unit=3.20,
            date=datetime.now().strftime("%Y-%m-%d"),
            symbol="SPY"
        )
    else:
        # Live mode
        api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
        if not api_key:
            print("Error: POLYGON_API_KEY required for live inference")
            print("Use --demo flag to run with mock data")
            sys.exit(1)

        print("=" * 60)
        print("SPY DAILY RANGE PLAN - LIVE INFERENCE")
        print("=" * 60)

        try:
            plan = infer_today(
                model_dir=args.model_dir,
                api_key=api_key
            )
        except Exception as e:
            print(f"Error generating plan: {e}")
            print("Market may be closed or data unavailable.")
            sys.exit(1)

    # Output plan
    print(plan.summary())

    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            f.write(plan.to_json())
        print(f"\nPlan saved to: {args.output}")

    # Also output as JSON to stdout for programmatic use
    print("\n" + "=" * 60)
    print("JSON OUTPUT")
    print("=" * 60)
    print(plan.to_json())


if __name__ == "__main__":
    main()
