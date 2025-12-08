"""
FVG Win Rate Prediction Model

Trains an XGBoost classifier to predict whether an FVG pattern
will hit its take profit targets or stop loss.

Features:
- Technical indicators (RSI, MACD, ATR, etc.)
- Gap characteristics (size, validation score)
- Market context (volume, trend)
- Time features (hour, day of week)

Target: Binary classification (win = hit TP3, loss = hit stop loss)
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import pickle
import os

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


def load_data(data_dir='./'):
    """Load and combine all ticker data"""
    all_features = []

    for ticker in ['spy', 'qqq', 'iwm']:
        filepath = os.path.join(data_dir, f'{ticker}_features.json')
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if 'features' in data:
                    all_features.extend(data['features'])
                    print(f"Loaded {len(data['features'])} features from {ticker.upper()}")

    return pd.DataFrame(all_features)


def prepare_features(df):
    """Prepare features for training"""
    # Create copy to avoid modifying original
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
            # Handle missing values
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


def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier"""
    # XGBoost parameters optimized for small dataset
    params = {
        'max_depth': 4,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'use_label_encoder': False,
        'random_state': 42,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
    }

    model = xgb.XGBClassifier(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    return model


def evaluate_model(model, X_test, y_test, feature_cols):
    """Evaluate model performance"""
    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1 Score:  {f1:.2%}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Loss', 'Win']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(f"  Predicted:  Loss  Win")
    print(f"  Actual Loss: {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"  Actual Win:  {cm[1][0]:4d}  {cm[1][1]:4d}")

    # Feature importance
    print("\nTop 10 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for i, row in importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'feature_importance': importance.to_dict('records')
    }


def main():
    print("="*50)
    print("FVG WIN RATE PREDICTION MODEL TRAINING")
    print("="*50)

    # Load data
    print("\n1. Loading data...")
    df = load_data()
    print(f"Total samples: {len(df)}")

    # Check outcome distribution
    print("\nOutcome distribution:")
    print(df['final_outcome'].value_counts())

    # Prepare features
    print("\n2. Preparing features...")
    df_valid, feature_cols, label_encoders = prepare_features(df)

    # Split data
    X = df_valid[feature_cols]
    y = df_valid['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Win rate in training: {y_train.mean():.2%}")
    print(f"Win rate in test: {y_test.mean():.2%}")

    # Train model
    print("\n3. Training XGBoost model...")
    model = train_model(X_train, y_train, X_test, y_test)

    # Cross-validation
    print("\n4. Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std()*2:.2%})")

    # Evaluate
    print("\n5. Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, feature_cols)

    # Save model
    print("\n6. Saving model...")
    model_path = 'fvg_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'feature_cols': feature_cols,
            'label_encoders': label_encoders,
            'metrics': metrics
        }, f)
    print(f"Model saved to {model_path}")

    # Save as JSON for Node.js integration
    model_info = {
        'feature_cols': feature_cols,
        'metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
        },
        'feature_importance': [
            {'feature': r['feature'], 'importance': float(r['importance'])}
            for r in metrics['feature_importance'][:15]
        ]
    }

    with open('model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    print("Model info saved to model_info.json")

    print("\n" + "="*50)
    print("TRAINING COMPLETE!")
    print("="*50)

    return model, metrics


if __name__ == '__main__':
    model, metrics = main()
