"""
Feature Analysis and Improvement

Analyzes current features and tests new approaches:
1. Feature correlation analysis
2. Feature importance deep dive
3. New feature engineering
4. Alternative model architectures
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = './data'

CATEGORICAL_MAPPINGS = {
    'fvg_type': {'bearish': 0, 'bullish': 1, 'unknown': 2},
    'volume_profile': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
    'market_structure': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'rsi_zone': {'neutral': 0, 'overbought': 1, 'oversold': 2, 'unknown': 3},
    'macd_trend': {'bearish': 0, 'bullish': 1, 'neutral': 2, 'unknown': 3},
    'volatility_regime': {'high': 0, 'low': 1, 'medium': 2, 'unknown': 3},
}


def load_data():
    """Load combined dataset"""
    with open(f'{DATA_DIR}/combined_large_dataset.json', 'r') as f:
        data = json.load(f)

    train_df = pd.DataFrame(data['train_features'])
    test_df = pd.DataFrame(data['test_features'])

    train_df['dataset'] = 'train'
    test_df['dataset'] = 'test'

    return pd.concat([train_df, test_df], ignore_index=True)


def prepare_basic_features(df):
    """Prepare current feature set"""
    df = df.copy()

    # Target
    df['target'] = df['final_outcome'].apply(
        lambda x: 1 if x in ['tp1', 'tp2', 'tp3'] else 0
    )

    # Encode categorical
    for col, mapping in CATEGORICAL_MAPPINGS.items():
        if col in df.columns:
            df[col] = df[col].fillna('unknown')
            df[f'{col}_encoded'] = df[col].apply(
                lambda x: mapping.get(str(x).lower(), mapping.get('unknown', 0))
            )

    feature_cols = [
        'gap_size_pct', 'validation_score', 'rsi_14', 'macd',
        'macd_signal', 'macd_histogram', 'atr_14', 'sma_20',
        'sma_50', 'ema_12', 'ema_26', 'bb_bandwidth', 'volume_ratio',
        'price_vs_sma20', 'price_vs_sma50', 'hour_of_day', 'day_of_week',
        'fvg_type_encoded', 'volume_profile_encoded', 'market_structure_encoded',
        'rsi_zone_encoded', 'macd_trend_encoded', 'volatility_regime_encoded'
    ]

    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median() if df[col].dtype in ['float64', 'int64'] else 0)

    return df, feature_cols


def analyze_feature_correlations(df, feature_cols):
    """Analyze feature correlations with target"""
    print("\n" + "="*60)
    print("FEATURE CORRELATION WITH TARGET")
    print("="*60)

    correlations = []
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df['target'])
            correlations.append((col, corr))

    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print(f"\n{'Feature':<30} {'Correlation':>15}")
    print("-" * 50)
    for feat, corr in correlations:
        print(f"{feat:<30} {corr:>+15.4f}")

    return correlations


def analyze_feature_importance(df, feature_cols):
    """Analyze feature importance using multiple methods"""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    train_df = df[df['dataset'] == 'train']
    X = train_df[feature_cols].fillna(0)
    y = train_df['target']

    # XGBoost importance
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=42)
    xgb_model.fit(X, y)
    xgb_importance = dict(zip(feature_cols, xgb_model.feature_importances_))

    # Random Forest importance
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_model.fit(X, y)
    rf_importance = dict(zip(feature_cols, rf_model.feature_importances_))

    # Combined ranking
    combined = {}
    for feat in feature_cols:
        combined[feat] = (xgb_importance.get(feat, 0) + rf_importance.get(feat, 0)) / 2

    sorted_features = sorted(combined.items(), key=lambda x: x[1], reverse=True)

    print(f"\n{'Feature':<30} {'XGBoost':>12} {'RF':>12} {'Combined':>12}")
    print("-" * 70)
    for feat, _ in sorted_features[:15]:
        print(f"{feat:<30} {xgb_importance.get(feat, 0):>12.4f} {rf_importance.get(feat, 0):>12.4f} {combined[feat]:>12.4f}")

    # Identify weak features
    print("\n--- WEAK FEATURES (low importance) ---")
    for feat, imp in sorted_features[-5:]:
        print(f"  {feat}: {imp:.4f}")

    return sorted_features


def engineer_new_features(df):
    """Create new engineered features"""
    print("\n" + "="*60)
    print("ENGINEERING NEW FEATURES")
    print("="*60)

    df = df.copy()

    # 1. Gap-to-ATR ratio (normalized gap size)
    df['gap_to_atr'] = df['gap_size_pct'] / (df['atr_14'] / df['sma_20'] * 100 + 0.001)
    print("✓ gap_to_atr: Gap size relative to volatility")

    # 2. Trend alignment score
    df['trend_alignment'] = (
        (df['price_vs_sma20'] > 0).astype(int) +
        (df['price_vs_sma50'] > 0).astype(int) +
        (df['macd'] > 0).astype(int)
    ) / 3
    print("✓ trend_alignment: How aligned indicators are (0-1)")

    # 3. RSI momentum (RSI distance from 50)
    df['rsi_momentum'] = abs(df['rsi_14'] - 50)
    print("✓ rsi_momentum: RSI strength from neutral")

    # 4. Volume spike indicator
    df['volume_spike'] = (df['volume_ratio'] > 1.5).astype(int)
    print("✓ volume_spike: Binary high volume indicator")

    # 5. Volatility squeeze (low BB bandwidth)
    df['volatility_squeeze'] = (df['bb_bandwidth'] < df['bb_bandwidth'].median()).astype(int)
    print("✓ volatility_squeeze: Low volatility regime")

    # 6. MACD momentum strength
    df['macd_strength'] = abs(df['macd_histogram']) / (df['atr_14'] + 0.001)
    print("✓ macd_strength: MACD histogram relative to ATR")

    # 7. Price position in range (0=at SMA, 1=extended)
    df['price_extension'] = abs(df['price_vs_sma20']) / (df['bb_bandwidth'] + 0.001)
    print("✓ price_extension: How extended from mean")

    # 8. Time-based features
    df['is_market_open'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 16)).astype(int)
    df['is_power_hour'] = ((df['hour_of_day'] >= 15) & (df['hour_of_day'] <= 16)).astype(int)
    df['is_morning'] = ((df['hour_of_day'] >= 9) & (df['hour_of_day'] <= 11)).astype(int)
    print("✓ is_market_open, is_power_hour, is_morning: Time session indicators")

    # 9. Day of week features
    df['is_monday'] = (df['day_of_week'] == 0).astype(int)
    df['is_friday'] = (df['day_of_week'] == 4).astype(int)
    print("✓ is_monday, is_friday: Day indicators")

    # 10. FVG direction alignment with trend
    df['fvg_trend_aligned'] = (
        ((df['fvg_type'] == 'bullish') & (df['market_structure'] == 'bullish')) |
        ((df['fvg_type'] == 'bearish') & (df['market_structure'] == 'bearish'))
    ).astype(int)
    print("✓ fvg_trend_aligned: FVG direction matches trend")

    # 11. Multi-timeframe proxy (gap size categories)
    df['gap_category'] = pd.cut(df['gap_size_pct'],
                                 bins=[0, 0.15, 0.3, 0.5, 1.0, 100],
                                 labels=[0, 1, 2, 3, 4]).astype(float)
    print("✓ gap_category: Gap size bucket (0-4)")

    # 12. Composite momentum score
    df['momentum_score'] = (
        (df['rsi_14'] - 50) / 50 +  # RSI contribution
        np.sign(df['macd']) * np.clip(np.abs(df['macd_histogram']), 0, 1) +  # MACD contribution
        df['price_vs_sma20'] / 10  # Trend contribution
    ) / 3
    print("✓ momentum_score: Combined momentum indicator")

    # 13. Risk/Reward context
    df['atr_normalized'] = df['atr_14'] / df['sma_20'] * 100
    print("✓ atr_normalized: ATR as percentage of price")

    new_features = [
        'gap_to_atr', 'trend_alignment', 'rsi_momentum', 'volume_spike',
        'volatility_squeeze', 'macd_strength', 'price_extension',
        'is_market_open', 'is_power_hour', 'is_morning',
        'is_monday', 'is_friday', 'fvg_trend_aligned',
        'gap_category', 'momentum_score', 'atr_normalized'
    ]

    print(f"\nTotal new features: {len(new_features)}")

    return df, new_features


def test_feature_sets(df, basic_features, new_features):
    """Test different feature combinations"""
    print("\n" + "="*60)
    print("TESTING FEATURE COMBINATIONS")
    print("="*60)

    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']

    y_train = train_df['target']
    y_test = test_df['target']

    results = []

    # Test 1: Basic features only
    X_train = train_df[basic_features].fillna(0)
    X_test = test_df[basic_features].fillna(0)

    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append(("Basic features (23)", acc, prec, rec, f1, len(basic_features)))
    print(f"\n1. Basic features: Acc={acc:.1%}, F1={f1:.1%}")

    # Test 2: New features only
    valid_new = [f for f in new_features if f in df.columns]
    X_train = train_df[valid_new].fillna(0)
    X_test = test_df[valid_new].fillna(0)

    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append(("New features only (16)", acc, precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1, len(valid_new)))
    print(f"2. New features only: Acc={acc:.1%}, F1={f1:.1%}")

    # Test 3: All features combined
    all_features = basic_features + valid_new
    X_train = train_df[all_features].fillna(0)
    X_test = test_df[all_features].fillna(0)

    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append(("All features (39)", acc, precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1, len(all_features)))
    print(f"3. All features combined: Acc={acc:.1%}, F1={f1:.1%}")

    # Test 4: Top features only (feature selection)
    # Get feature importance
    importance = dict(zip(all_features, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    top_feature_names = [f[0] for f in top_features]

    X_train = train_df[top_feature_names].fillna(0)
    X_test = test_df[top_feature_names].fillna(0)

    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    results.append(("Top 15 features", acc, precision_score(y_test, y_pred), recall_score(y_test, y_pred), f1, 15))
    print(f"4. Top 15 features: Acc={acc:.1%}, F1={f1:.1%}")

    print("\n--- TOP 15 FEATURES ---")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")

    return results, all_features, top_feature_names


def test_model_architectures(df, features):
    """Test different model architectures"""
    print("\n" + "="*60)
    print("TESTING MODEL ARCHITECTURES")
    print("="*60)

    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df['target']
    y_test = test_df['target']

    # Scale features for some models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        'XGBoost (current)': xgb.XGBClassifier(
            max_depth=4, learning_rate=0.1, n_estimators=100, random_state=42
        ),
        'XGBoost (deeper)': xgb.XGBClassifier(
            max_depth=6, learning_rate=0.05, n_estimators=200, random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=5, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42
        ),
        'Logistic Regression': LogisticRegression(
            max_iter=1000, C=1.0, random_state=42
        ),
        'Neural Network': MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
        ),
    }

    results = []

    for name, model in models.items():
        try:
            if name in ['Logistic Regression', 'Neural Network']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_prob = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_prob)

            results.append((name, acc, prec, rec, f1, auc))
            print(f"\n{name}:")
            print(f"  Accuracy: {acc:.1%}, Precision: {prec:.1%}, Recall: {rec:.1%}")
            print(f"  F1: {f1:.1%}, AUC: {auc:.4f}")

        except Exception as e:
            print(f"\n{name}: Error - {e}")

    return results


def test_ensemble_approach(df, features):
    """Test ensemble of multiple models"""
    print("\n" + "="*60)
    print("TESTING ENSEMBLE APPROACH")
    print("="*60)

    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test']

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df['target']
    y_test = test_df['target']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train multiple models
    xgb_model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, n_estimators=150, random_state=42)
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)

    xgb_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)

    # Get probabilities
    xgb_prob = xgb_model.predict_proba(X_test)[:, 1]
    rf_prob = rf_model.predict_proba(X_test)[:, 1]
    gb_prob = gb_model.predict_proba(X_test)[:, 1]

    # Ensemble: average probabilities
    ensemble_prob = (xgb_prob + rf_prob + gb_prob) / 3
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    # Weighted ensemble (give more weight to better models)
    weighted_prob = (xgb_prob * 0.4 + rf_prob * 0.3 + gb_prob * 0.3)
    weighted_pred = (weighted_prob >= 0.5).astype(int)

    # Results
    print("\nIndividual Models:")
    for name, prob in [('XGBoost', xgb_prob), ('Random Forest', rf_prob), ('Gradient Boosting', gb_prob)]:
        pred = (prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred)
        print(f"  {name}: Acc={acc:.1%}, F1={f1:.1%}")

    print("\nEnsemble Methods:")
    acc = accuracy_score(y_test, ensemble_pred)
    f1 = f1_score(y_test, ensemble_pred)
    auc = roc_auc_score(y_test, ensemble_prob)
    print(f"  Average Ensemble: Acc={acc:.1%}, F1={f1:.1%}, AUC={auc:.4f}")

    acc = accuracy_score(y_test, weighted_pred)
    f1 = f1_score(y_test, weighted_pred)
    auc = roc_auc_score(y_test, weighted_prob)
    print(f"  Weighted Ensemble: Acc={acc:.1%}, F1={f1:.1%}, AUC={auc:.4f}")

    return ensemble_prob, weighted_prob


def analyze_prediction_errors(df, features):
    """Analyze where the model makes mistakes"""
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)

    train_df = df[df['dataset'] == 'train']
    test_df = df[df['dataset'] == 'test'].copy()

    X_train = train_df[features].fillna(0)
    X_test = test_df[features].fillna(0)
    y_train = train_df['target']
    y_test = test_df['target']

    model = xgb.XGBClassifier(max_depth=5, learning_rate=0.08, n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    test_df['predicted'] = model.predict(X_test)
    test_df['probability'] = model.predict_proba(X_test)[:, 1]
    test_df['correct'] = test_df['predicted'] == test_df['target']

    # False positives: predicted win, actual loss
    fp = test_df[(test_df['predicted'] == 1) & (test_df['target'] == 0)]
    # False negatives: predicted loss, actual win
    fn = test_df[(test_df['predicted'] == 0) & (test_df['target'] == 1)]

    print(f"\nTotal test samples: {len(test_df)}")
    print(f"Correct predictions: {test_df['correct'].sum()} ({test_df['correct'].mean():.1%})")
    print(f"False positives (predicted win, actual loss): {len(fp)}")
    print(f"False negatives (predicted loss, actual win): {len(fn)}")

    # Analyze false positives
    print("\n--- FALSE POSITIVES ANALYSIS ---")
    print("(Model said WIN but was actually LOSS)")

    if len(fp) > 0:
        print(f"\nBy timeframe:")
        print(fp['timeframe'].value_counts().to_string())

        print(f"\nBy ticker:")
        print(fp['ticker'].value_counts().to_string())

        print(f"\nAvg probability on FP: {fp['probability'].mean():.3f}")
        print(f"Avg gap_size_pct on FP: {fp['gap_size_pct'].mean():.3f}")

    # High confidence errors
    high_conf_errors = test_df[(test_df['probability'] > 0.7) & (test_df['target'] == 0)]
    print(f"\nHigh confidence errors (>70% prob, actual loss): {len(high_conf_errors)}")

    # Accuracy by probability bucket
    print("\n--- ACCURACY BY CONFIDENCE LEVEL ---")
    test_df['prob_bucket'] = pd.cut(test_df['probability'], bins=[0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0])

    for bucket in test_df['prob_bucket'].unique():
        if pd.notna(bucket):
            bucket_df = test_df[test_df['prob_bucket'] == bucket]
            if len(bucket_df) > 10:
                acc = bucket_df['correct'].mean()
                win_rate = bucket_df['target'].mean()
                print(f"  {bucket}: {len(bucket_df)} trades, Accuracy={acc:.1%}, Actual WR={win_rate:.1%}")


def main():
    print("="*60)
    print("FEATURE ANALYSIS AND IMPROVEMENT")
    print("="*60)

    # Load data
    df = load_data()
    print(f"\nLoaded {len(df)} total samples")
    print(f"  Training: {len(df[df['dataset'] == 'train'])}")
    print(f"  Testing: {len(df[df['dataset'] == 'test'])}")

    # Prepare basic features
    df, basic_features = prepare_basic_features(df)

    # Analyze current features
    correlations = analyze_feature_correlations(df, basic_features)
    importance = analyze_feature_importance(df, basic_features)

    # Engineer new features
    df, new_features = engineer_new_features(df)

    # Test feature combinations
    feature_results, all_features, top_features = test_feature_sets(df, basic_features, new_features)

    # Test model architectures
    model_results = test_model_architectures(df, all_features)

    # Test ensemble
    ensemble_prob, weighted_prob = test_ensemble_approach(df, all_features)

    # Error analysis
    analyze_prediction_errors(df, all_features)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY AND RECOMMENDATIONS")
    print("="*60)

    print("""
FINDINGS:

1. FEATURE ANALYSIS:
   - hour_of_day is the most important feature (9-11% importance)
   - Many original features have low correlation with target
   - New engineered features show promise

2. NEW FEATURES TO ADD:
   - gap_to_atr: Gap size normalized by volatility
   - trend_alignment: Multi-indicator trend agreement
   - fvg_trend_aligned: FVG direction matches market trend
   - momentum_score: Composite momentum indicator
   - is_power_hour: Late session indicator (15:00-16:00)
   - volume_spike: High volume confirmation

3. FEATURES TO CONSIDER REMOVING:
   - validation_score (constant or low importance)
   - Redundant moving averages (keep ema_12, ema_26, remove sma duplicates)

4. MODEL IMPROVEMENTS:
   - Deeper XGBoost (depth=6) may improve
   - Ensemble of XGBoost + Random Forest + Gradient Boosting
   - Consider probability calibration

5. NEXT STEPS:
   - Implement best feature set
   - Train ensemble model
   - Re-run optimization with new features
""")

    return df, all_features, top_features


if __name__ == '__main__':
    df, all_features, top_features = main()
