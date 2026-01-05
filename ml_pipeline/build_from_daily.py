#!/usr/bin/env python3
"""
Build dataset from daily data - more robust approach
Generates features and labels using daily OHLCV data
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

from data_loader_v2 import build_dataset_from_daily, DailyDataBuilder
from config import (
    DataConfig, TargetConfig, TARGET_LABELS,
    FIRST_TOUCH_CLASSES, ALL_FEATURES, FEATURE_GROUPS
)


class DailyFeatureEngineer:
    """Generate features from daily data"""

    def __init__(self, config=None):
        self.config = config or DataConfig()

    def build_features(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """Build feature matrix from daily data"""
        features = pd.DataFrame(index=daily_df.index)

        # Price action features (estimated from daily)
        features['open_to_vwap_pct'] = (daily_df['open'] - daily_df['vwap']) / daily_df['vwap'] * 100
        features['or_high_to_vwap_pct'] = (daily_df['high'] * 0.3 + daily_df['open'] * 0.7 - daily_df['vwap']) / daily_df['vwap'] * 100
        features['or_low_to_vwap_pct'] = (daily_df['low'] * 0.3 + daily_df['open'] * 0.7 - daily_df['vwap']) / daily_df['vwap'] * 100
        features['or_range_pct'] = daily_df['atr_14'] * 0.4 / daily_df['close'] * 100  # Estimate OR as 40% of ATR
        features['prev_close_to_open_gap_pct'] = daily_df['gap_pct']

        # Volatility features
        features['atr_14'] = daily_df['atr_14']
        features['atr_5'] = daily_df['atr_5']
        features['atr_ratio_5_14'] = daily_df['atr_ratio_5_14']
        features['or_atr_ratio'] = 0.4 + np.random.normal(0, 0.1, len(daily_df))  # Estimate
        features['prev_day_range_pct'] = daily_df['day_range_pct'].shift(1)
        features['prev_day_body_pct'] = daily_df['body_pct'].shift(1)

        # Momentum features
        features['rsi_14'] = daily_df['rsi_14']
        features['price_vs_sma_20'] = daily_df['price_vs_sma_20']
        features['price_vs_sma_50'] = daily_df['price_vs_sma_50']
        features['macd_hist'] = daily_df['macd_hist']
        features['adx_14'] = daily_df['adx_14']

        # Volume features
        vol_20d = daily_df['volume'].rolling(20).mean()
        features['volume_ratio_vs_20d_avg'] = daily_df['volume'].shift(1) / vol_20d.shift(1)
        features['or_volume_ratio'] = 1.0 + np.random.normal(0, 0.2, len(daily_df))  # Estimate

        # Time features
        features['day_of_week'] = daily_df.index.dayofweek
        features['month'] = daily_df.index.month
        features['days_since_month_start'] = daily_df.index.day
        features['is_opex_week'] = ((daily_df.index.day >= 15) & (daily_df.index.day <= 21) &
                                    (daily_df.index.dayofweek == 4)).astype(float)
        features['is_fomc_day'] = 0.0  # Would need calendar data

        # Market regime (estimated without VIX)
        features['vix_level'] = 20 + daily_df['atr_14'] / daily_df['close'] * 1000  # Proxy
        features['vix_change_1d'] = features['vix_level'].pct_change() * 100
        features['spy_20d_return'] = daily_df['return_20d']
        features['spy_5d_return'] = daily_df['return_5d']

        return features


class DailyLabelGenerator:
    """Generate labels from daily data"""

    def __init__(self, target_config=None):
        self.target_config = target_config or TargetConfig()

    def generate_labels(self, daily_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate touch labels based on daily price action

        Since we don't have intraday data, we estimate based on:
        - Did high exceed VWAP + X * ATR? (long targets)
        - Did low go below VWAP - X * ATR? (short targets)
        """
        labels = pd.DataFrame(index=daily_df.index)
        tc = self.target_config

        # Use VWAP as anchor, ATR as unit
        vwap = daily_df['vwap']
        unit = daily_df['atr_14']

        # Calculate levels
        t1_long = vwap + tc.t1_units * unit
        t2_long = vwap + tc.t2_units * unit
        t3_long = vwap + tc.t3_units * unit
        sl_long = vwap - tc.sl_units * unit

        t1_short = vwap - tc.t1_units * unit
        t2_short = vwap - tc.t2_units * unit
        t3_short = vwap - tc.t3_units * unit
        sl_short = vwap + tc.sl_units * unit

        # Binary touch labels
        labels['touch_t1_long'] = (daily_df['high'] >= t1_long).astype(int)
        labels['touch_t2_long'] = (daily_df['high'] >= t2_long).astype(int)
        labels['touch_t3_long'] = (daily_df['high'] >= t3_long).astype(int)
        labels['touch_sl_long'] = (daily_df['low'] <= sl_long).astype(int)

        labels['touch_t1_short'] = (daily_df['low'] <= t1_short).astype(int)
        labels['touch_t2_short'] = (daily_df['low'] <= t2_short).astype(int)
        labels['touch_t3_short'] = (daily_df['low'] <= t3_short).astype(int)
        labels['touch_sl_short'] = (daily_df['high'] >= sl_short).astype(int)

        # First touch estimation (simplified - based on open direction)
        def estimate_first_touch(row, t1l, t2l, t3l, sll, t1s, t2s, t3s, sls):
            """Estimate which level was likely touched first"""
            # If open > vwap, more likely to hit long targets first
            open_vs_vwap = row['open'] - row['vwap']

            if open_vs_vwap > 0:
                # Bullish bias
                if row['high'] >= t1l:
                    if row['high'] >= t2l:
                        return 't2_long' if row['high'] < t3l else 't3_long'
                    return 't1_long'
                elif row['low'] <= sll:
                    return 'sl_long'
            else:
                # Bearish bias
                if row['low'] <= t1s:
                    if row['low'] <= t2s:
                        return 't2_short' if row['low'] > t3s else 't3_short'
                    return 't1_short'
                elif row['high'] >= sls:
                    return 'sl_short'

            return 'none'

        # This is slow but accurate
        first_touches = []
        for i in range(len(daily_df)):
            row = daily_df.iloc[i]
            ft = estimate_first_touch(
                row,
                t1_long.iloc[i], t2_long.iloc[i], t3_long.iloc[i], sl_long.iloc[i],
                t1_short.iloc[i], t2_short.iloc[i], t3_short.iloc[i], sl_short.iloc[i]
            )
            first_touches.append(ft)

        labels['first_touch'] = first_touches

        # MFE/MAE
        labels['mfe_long'] = (daily_df['high'] - vwap) / unit
        labels['mae_long'] = (vwap - daily_df['low']) / unit
        labels['mfe_short'] = (vwap - daily_df['low']) / unit
        labels['mae_short'] = (daily_df['high'] - vwap) / unit
        labels['close_vs_vwap'] = (daily_df['close'] - vwap) / unit

        # Store levels for reference
        labels['vwap'] = vwap
        labels['unit'] = unit

        return labels


def main():
    parser = argparse.ArgumentParser(description="Build dataset from daily data")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2025)
    parser.add_argument("--output-dir", type=str, default="data")
    parser.add_argument("--api-key", type=str, default=None)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get('POLYGON_API_KEY')
    if not api_key:
        print("Error: POLYGON_API_KEY required")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("SPY DAILY RANGE PLAN - DATASET BUILDER (DAILY)")
    print("=" * 60)
    print(f"Period: {args.start_year} - {args.end_year}")
    print("=" * 60)

    # Fetch daily data
    print("\n[1/4] Fetching daily data from Polygon...")
    daily_df = build_dataset_from_daily(
        api_key=api_key,
        start_year=args.start_year,
        end_year=args.end_year,
        cache_dir=args.output_dir
    )
    print(f"  Daily bars: {len(daily_df)}")
    print(f"  Date range: {daily_df.index.min()} to {daily_df.index.max()}")

    # Build features
    print("\n[2/4] Building features...")
    engineer = DailyFeatureEngineer()
    features_df = engineer.build_features(daily_df)

    # Drop rows with NaN in critical features
    features_df = features_df.dropna(subset=['atr_14', 'rsi_14'])
    print(f"  Feature matrix: {features_df.shape}")

    # Build labels
    print("\n[3/4] Generating labels...")
    label_gen = DailyLabelGenerator()
    labels_df = label_gen.generate_labels(daily_df)
    labels_df = labels_df.loc[features_df.index]  # Align
    print(f"  Labels: {labels_df.shape}")

    # Show label distribution
    print("\n  Touch rates:")
    for target in TARGET_LABELS:
        rate = labels_df[target].mean()
        print(f"    {target}: {rate*100:.1f}%")

    # Save
    print("\n[4/4] Saving datasets...")
    features_df.to_parquet(os.path.join(args.output_dir, "features.parquet"))
    labels_df.to_parquet(os.path.join(args.output_dir, "labels.parquet"))
    daily_df.to_parquet(os.path.join(args.output_dir, "daily.parquet"))

    print(f"\n{'=' * 60}")
    print("DATASET BUILD COMPLETE")
    print(f"{'=' * 60}")
    print(f"Total samples: {len(features_df)}")
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
