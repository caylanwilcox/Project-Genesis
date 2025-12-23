"""
Analyze V6 Model - Extract feature importance and weights
"""

import pickle
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

def analyze_model(ticker):
    model_path = os.path.join(MODELS_DIR, f'{ticker.lower()}_intraday_v6.pkl')
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)

    print(f"\n{'='*70}")
    print(f"  {ticker} V6 MODEL ANALYSIS")
    print(f"{'='*70}")

    print(f"\n  Version: {model_data.get('version')}")
    print(f"  Features: {len(model_data['feature_cols'])}")

    # Ensemble weights
    print(f"\n  ENSEMBLE WEIGHTS:")
    print(f"  {'─'*40}")

    print(f"\n  Early Model (9-11 AM):")
    for name, weight in model_data['weights_early'].items():
        print(f"    {name}: {weight:.1%}")

    print(f"\n  Late Model A (12-4 PM) - Target A:")
    for name, weight in model_data['weights_late_a'].items():
        print(f"    {name}: {weight:.1%}")

    print(f"\n  Late Model B (12-4 PM) - Target B:")
    for name, weight in model_data['weights_late_b'].items():
        print(f"    {name}: {weight:.1%}")

    # Feature importance from Random Forest (late model)
    print(f"\n  TOP FEATURES - Late Target A (RF importance):")
    print(f"  {'─'*40}")

    rf_model = model_data['models_late_a']['rf']
    feature_cols = model_data['feature_cols']
    importances = rf_model.feature_importances_

    feature_importance = list(zip(feature_cols, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance[:15]):
        print(f"    {i+1:2}. {feat:<30} {imp:.4f}")

    # Feature importance for Target B
    print(f"\n  TOP FEATURES - Late Target B (RF importance):")
    print(f"  {'─'*40}")

    rf_model_b = model_data['models_late_b']['rf']
    importances_b = rf_model_b.feature_importances_

    feature_importance_b = list(zip(feature_cols, importances_b))
    feature_importance_b.sort(key=lambda x: x[1], reverse=True)

    for i, (feat, imp) in enumerate(feature_importance_b[:15]):
        print(f"    {i+1:2}. {feat:<30} {imp:.4f}")

    # Accuracy stats
    print(f"\n  MODEL ACCURACY:")
    print(f"  {'─'*40}")
    print(f"    Early Target A:  {model_data['acc_early']:.1%}")
    print(f"    Late Target A:   {model_data['acc_late_a']:.1%}")
    print(f"    Late Target B:   {model_data['acc_late_b']:.1%}")

    return model_data, feature_importance, feature_importance_b

def main():
    print("="*70)
    print("  V6 MODEL DEEP ANALYSIS")
    print("="*70)

    all_data = {}
    for ticker in ['SPY', 'QQQ', 'IWM']:
        all_data[ticker] = analyze_model(ticker)

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY - ALL TICKERS")
    print(f"{'='*70}")

    print(f"\n  {'Ticker':<8} {'Early A':>10} {'Late A':>10} {'Late B':>10}")
    print(f"  {'─'*42}")
    for ticker, (data, _, _) in all_data.items():
        print(f"  {ticker:<8} {data['acc_early']:>9.1%} {data['acc_late_a']:>9.1%} {data['acc_late_b']:>9.1%}")

if __name__ == '__main__':
    main()
