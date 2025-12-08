"""
Per-Ticker FVG Win Rate Prediction Models

Trains individual XGBoost classifiers for each ticker (SPY, QQQ, IWM)
to capture ticker-specific patterns and improve prediction accuracy.

Weeks 5-8 Implementation:
- Week 5-6: QQQ model training
- Week 7-8: IWM model training

Each ticker gets its own model optimized for its volatility and behavior patterns.
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os
from datetime import datetime

# Feature columns for ML model
FEATURE_COLUMNS = [
    # Gap characteristics
    'gap_size_pct',
    'validation_score',

    # Technical indicators
    'rsi_14',
    'macd',
    'macd_signal',
    'macd_histogram',
    'atr_14',
    'sma_20',
    'sma_50',
    'ema_12',
    'ema_26',
    'bb_bandwidth',
    'volume_ratio',

    # Derived features
    'price_vs_sma20',
    'price_vs_sma50',

    # Time features
    'hour_of_day',
    'day_of_week',
]

# Categorical features to encode
CATEGORICAL_COLUMNS = [
    'fvg_type',
    'volume_profile',
    'market_structure',
    'rsi_zone',
    'macd_trend',
    'volatility_regime',
]

# Ticker-specific hyperparameters (tuned for each ticker's characteristics)
TICKER_PARAMS = {
    'spy': {
        # SPY: Large cap, less volatile, more predictable
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    },
    'qqq': {
        # QQQ: Tech-heavy, moderate volatility, momentum-driven
        'max_depth': 5,
        'learning_rate': 0.08,
        'n_estimators': 150,
        'min_child_weight': 2,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
    },
    'iwm': {
        # IWM: Small caps, higher volatility, mean-reverting
        'max_depth': 4,
        'learning_rate': 0.12,
        'n_estimators': 120,
        'min_child_weight': 4,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
    },
}


def load_ticker_data(ticker, data_dir='./', use_large_dataset=True):
    """Load data for a specific ticker"""

    # Try large dataset first
    if use_large_dataset:
        large_filepath = os.path.join(data_dir, 'data', f'{ticker.lower()}_large_features.json')
        if os.path.exists(large_filepath):
            with open(large_filepath, 'r') as f:
                data = json.load(f)
            if 'features' in data:
                df = pd.DataFrame(data['features'])
                print(f"Loaded {len(df)} samples for {ticker.upper()} (LARGE DATASET)")
                return df

    # Fallback to old small dataset
    filepath = os.path.join(data_dir, f'{ticker.lower()}_features.json')

    if not os.path.exists(filepath):
        print(f"WARNING: {filepath} not found")
        return pd.DataFrame()

    with open(filepath, 'r') as f:
        data = json.load(f)

    if 'features' not in data:
        print(f"WARNING: No 'features' key in {filepath}")
        return pd.DataFrame()

    df = pd.DataFrame(data['features'])
    print(f"Loaded {len(df)} samples for {ticker.upper()}")
    return df


def load_all_data(data_dir='./', use_large_dataset=True):
    """Load and combine all ticker data (for combined model)"""

    # Try loading the combined large dataset first
    if use_large_dataset:
        combined_path = os.path.join(data_dir, 'data', 'combined_large_dataset.json')
        if os.path.exists(combined_path):
            with open(combined_path, 'r') as f:
                data = json.load(f)
            train_df = pd.DataFrame(data.get('train_features', []))
            test_df = pd.DataFrame(data.get('test_features', []))

            if len(train_df) > 0:
                train_df['dataset'] = 'train'
            if len(test_df) > 0:
                test_df['dataset'] = 'test'

            combined = pd.concat([train_df, test_df], ignore_index=True)
            print(f"Loaded {len(combined)} total samples from LARGE combined dataset")
            print(f"  Training: {len(train_df)}, Testing: {len(test_df)}")
            return combined

    # Fallback: load individual ticker files
    all_data = []

    for ticker in ['spy', 'qqq', 'iwm']:
        df = load_ticker_data(ticker, data_dir, use_large_dataset)
        if len(df) > 0:
            df['ticker'] = ticker.upper()
            all_data.append(df)

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def prepare_features(df):
    """Prepare features for training"""
    df = df.copy()

    # Create binary target: 1 = win (hit any TP), 0 = loss (hit stop loss)
    df['target'] = df['final_outcome'].apply(
        lambda x: 1 if x in ['tp1', 'tp2', 'tp3'] else 0
    )

    # Encode categorical features
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].fillna('unknown')
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le

    # Get feature columns (numeric + encoded categorical)
    feature_cols = FEATURE_COLUMNS.copy()
    for col in CATEGORICAL_COLUMNS:
        if f'{col}_encoded' in df.columns:
            feature_cols.append(f'{col}_encoded')

    # Fill missing numeric values with median
    for col in feature_cols:
        if col in df.columns and df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].fillna(df[col].median())

    # Filter to only rows with valid features
    valid_mask = df[feature_cols].notna().all(axis=1)
    df_valid = df[valid_mask].copy()

    print(f"Valid samples: {len(df_valid)} / {len(df)}")

    return df_valid, feature_cols, label_encoders


def train_model(X_train, y_train, X_test, y_test, ticker='spy'):
    """Train XGBoost classifier with ticker-specific parameters"""

    # Get ticker-specific params or use defaults
    ticker_params = TICKER_PARAMS.get(ticker.lower(), TICKER_PARAMS['spy'])

    params = {
        **ticker_params,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model


def hyperparameter_search(X_train, y_train, ticker='spy'):
    """Optional: Grid search for optimal hyperparameters"""
    print(f"\nRunning hyperparameter search for {ticker.upper()}...")

    param_grid = {
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.08, 0.1, 0.12],
        'n_estimators': [80, 100, 120, 150],
        'min_child_weight': [2, 3, 4],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
    }

    base_model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
    )

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    print(f"Best params for {ticker.upper()}: {grid_search.best_params_}")
    print(f"Best F1 score: {grid_search.best_score_:.4f}")

    return grid_search.best_params_


def evaluate_model(model, X_test, y_test, feature_cols, ticker=''):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = 0.0

    ticker_label = f" ({ticker.upper()})" if ticker else ""

    print("\n" + "="*50)
    print(f"MODEL EVALUATION RESULTS{ticker_label}")
    print("="*50)
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")
    print(f"AUC-ROC:   {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win'], zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Predicted:  Loss  Win")
    if len(cm) == 2:
        print(f"  Actual Loss: {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"  Actual Win:  {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Feature importance
    print("\nTop 10 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for _, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'feature_importance': importance.to_dict('records')
    }


def train_ticker_model(ticker, data_dir='./', output_dir='./models', walk_forward=True):
    """Train a model for a specific ticker with optional walk-forward validation"""

    print("\n" + "="*60)
    print(f"TRAINING {ticker.upper()} MODEL")
    print("="*60)

    # Load data
    df = load_ticker_data(ticker, data_dir)
    if len(df) == 0:
        print(f"ERROR: No data available for {ticker}")
        return None, None

    # Check outcome distribution
    print("\nOutcome distribution:")
    print(df['final_outcome'].value_counts())

    # Prepare features
    df_valid, feature_cols, label_encoders = prepare_features(df)

    if len(df_valid) < 20:
        print(f"ERROR: Not enough valid samples ({len(df_valid)}) for {ticker}")
        return None, None

    # Walk-forward split: train on 2022-2023, test on 2024
    if walk_forward and 'dataset' in df_valid.columns:
        train_mask = df_valid['dataset'] == 'train'
        test_mask = df_valid['dataset'] == 'test'

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            X_train = df_valid.loc[train_mask, feature_cols]
            y_train = df_valid.loc[train_mask, 'target']
            X_test = df_valid.loc[test_mask, feature_cols]
            y_test = df_valid.loc[test_mask, 'target']
            print(f"\n*** WALK-FORWARD VALIDATION ***")
            print(f"Training on 2022-2023, Testing on 2024")
        else:
            walk_forward = False

    # Fallback to random split
    if not walk_forward or 'dataset' not in df_valid.columns:
        X = df_valid[feature_cols]
        y = df_valid['target']

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        print(f"\n*** RANDOM SPLIT (no walk-forward data) ***")

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Win rate in training: {y_train.mean():.2%}")
    print(f"Win rate in test: {y_test.mean():.2%}")

    # Train model
    print(f"\nTraining XGBoost model for {ticker.upper()}...")
    model = train_model(X_train, y_train, X_test, y_test, ticker)

    # Cross-validation
    print("\nCross-validation (5-fold)...")
    try:
        cv_scores = cross_val_score(model, X, y, cv=min(5, len(y)//2), scoring='accuracy')
        print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")
    except Exception as e:
        print(f"CV failed: {e}")
        cv_scores = np.array([0])

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_cols, ticker)
    metrics['cv_accuracy_mean'] = float(cv_scores.mean())
    metrics['cv_accuracy_std'] = float(cv_scores.std())

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_filename = f'{ticker.lower()}_fvg_model.pkl'
    model_path = os.path.join(output_dir, model_filename)

    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'ticker': ticker.upper(),
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
        }, f)
    print(f"\nModel saved to {model_path}")

    # Save model info as JSON
    info_filename = f'{ticker.lower()}_model_info.json'
    info_path = os.path.join(output_dir, info_filename)

    model_info = {
        'ticker': ticker.upper(),
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'auc': float(metrics['auc']),
            'cv_accuracy_mean': float(metrics['cv_accuracy_mean']),
            'cv_accuracy_std': float(metrics['cv_accuracy_std']),
        },
        'hyperparameters': TICKER_PARAMS.get(ticker.lower(), {}),
        'feature_importance': [
            {'feature': r['feature'], 'importance': float(r['importance'])}
            for r in metrics['feature_importance'][:15]
        ],
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'trained_at': datetime.now().isoformat(),
    }

    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    print(f"Model info saved to {info_path}")

    return model, metrics


def train_combined_model(data_dir='./', output_dir='./models', walk_forward=True):
    """Train a combined model using all tickers with optional walk-forward validation"""

    print("\n" + "="*60)
    print("TRAINING COMBINED MODEL (ALL TICKERS)")
    print("="*60)

    # Load all data
    df = load_all_data(data_dir)
    if len(df) == 0:
        print("ERROR: No data available")
        return None, None

    print(f"\nTotal samples: {len(df)}")
    if 'ticker' in df.columns:
        print("\nSamples per ticker:")
        print(df['ticker'].value_counts())

    # Prepare features
    df_valid, feature_cols, label_encoders = prepare_features(df)

    # Walk-forward split: train on 2022-2023, test on 2024
    if walk_forward and 'dataset' in df_valid.columns:
        train_mask = df_valid['dataset'] == 'train'
        test_mask = df_valid['dataset'] == 'test'

        if train_mask.sum() > 0 and test_mask.sum() > 0:
            X_train = df_valid.loc[train_mask, feature_cols]
            y_train = df_valid.loc[train_mask, 'target']
            X_test = df_valid.loc[test_mask, feature_cols]
            y_test = df_valid.loc[test_mask, 'target']
            print(f"\n*** WALK-FORWARD VALIDATION ***")
            print(f"Training on 2022-2023, Testing on 2024")
        else:
            walk_forward = False

    # Fallback to random split
    if not walk_forward or 'dataset' not in df_valid.columns:
        X = df_valid[feature_cols]
        y = df_valid['target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        print(f"\n*** RANDOM SPLIT (no walk-forward data) ***")

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Win rate in training: {y_train.mean():.2%}")
    print(f"Win rate in test: {y_test.mean():.2%}")

    # Train model
    model = train_model(X_train, y_train, X_test, y_test, 'combined')

    # Evaluate
    metrics = evaluate_model(model, X_test, y_test, feature_cols, 'COMBINED')

    # Save
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, 'combined_fvg_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'ticker': 'COMBINED',
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'metrics': metrics,
            'trained_at': datetime.now().isoformat(),
        }, f)
    print(f"\nModel saved to {model_path}")

    return model, metrics


def main():
    """Train all ticker models"""

    print("="*60)
    print("FVG MULTI-TICKER MODEL TRAINING")
    print("Weeks 5-8: QQQ & IWM Model Training")
    print("="*60)

    data_dir = './'
    output_dir = './models'

    results = {}

    # Train individual ticker models
    for ticker in ['spy', 'qqq', 'iwm']:
        model, metrics = train_ticker_model(ticker, data_dir, output_dir)
        if metrics:
            results[ticker] = metrics

    # Train combined model
    model, metrics = train_combined_model(data_dir, output_dir)
    if metrics:
        results['combined'] = metrics

    # Summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    print("\n{:<10} {:>10} {:>10} {:>10} {:>10}".format(
        "Ticker", "Accuracy", "Precision", "Recall", "F1"
    ))
    print("-" * 55)

    for ticker, m in results.items():
        print("{:<10} {:>9.1%} {:>10.1%} {:>9.1%} {:>9.1%}".format(
            ticker.upper(),
            m['accuracy'],
            m['precision'],
            m['recall'],
            m['f1']
        ))

    # Save summary
    summary_path = os.path.join(output_dir, 'training_summary.json')
    summary = {
        'trained_at': datetime.now().isoformat(),
        'tickers': list(results.keys()),
        'results': {
            ticker: {
                'accuracy': float(m['accuracy']),
                'precision': float(m['precision']),
                'recall': float(m['recall']),
                'f1': float(m['f1']),
            }
            for ticker, m in results.items()
        }
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

    return results


if __name__ == '__main__':
    results = main()
