"""
Model Training with Walk-Forward Validation
Trains LightGBM models for SPY Daily Range Plan Prediction
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
import joblib
import os
from datetime import datetime
from tqdm import tqdm

from config import (
    ModelConfig, WalkForwardConfig, TARGET_LABELS,
    FIRST_TOUCH_CLASSES, ALL_FEATURES
)


class WalkForwardTrainer:
    """
    Walk-forward validation trainer for time series data

    For each test year Y, trains on data from 2004 to Y-1
    """

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        wf_config: Optional[WalkForwardConfig] = None
    ):
        self.model_config = model_config or ModelConfig()
        self.wf_config = wf_config or WalkForwardConfig()

    def _get_lgb_params(self, is_multiclass: bool = False) -> Dict:
        """Get LightGBM parameters"""
        mc = self.model_config

        params = {
            'n_estimators': mc.n_estimators,
            'learning_rate': mc.learning_rate,
            'max_depth': mc.max_depth,
            'num_leaves': mc.num_leaves,
            'min_child_samples': mc.min_child_samples,
            'subsample': mc.subsample,
            'colsample_bytree': mc.colsample_bytree,
            'reg_alpha': mc.reg_alpha,
            'reg_lambda': mc.reg_lambda,
            'random_state': mc.random_state,
            'n_jobs': -1,
            'verbose': -1,
        }

        if is_multiclass:
            params['objective'] = 'multiclass'
            params['num_class'] = len(FIRST_TOUCH_CLASSES)
        else:
            params['objective'] = 'binary'

        return params

    def train_binary_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.LGBMClassifier:
        """Train a binary classification model"""
        params = self._get_lgb_params(is_multiclass=False)

        model = lgb.LGBMClassifier(**params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)

        return model

    def train_multiclass_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> lgb.LGBMClassifier:
        """Train a multiclass classification model"""
        params = self._get_lgb_params(is_multiclass=True)

        model = lgb.LGBMClassifier(**params)

        if X_val is not None and y_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)]
            )
        else:
            model.fit(X_train, y_train)

        return model

    def walk_forward_split(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        test_year: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data for walk-forward validation

        Args:
            features_df: Feature matrix with date index
            labels_df: Labels with date index
            test_year: Year to use as test set

        Returns:
            X_train, X_test, y_train, y_test DataFrames
        """
        # Align features and labels
        common_dates = features_df.index.intersection(labels_df.index)
        features_aligned = features_df.loc[common_dates]
        labels_aligned = labels_df.loc[common_dates]

        # Split by year
        train_mask = features_aligned.index.year < test_year
        test_mask = features_aligned.index.year == test_year

        X_train = features_aligned[train_mask]
        X_test = features_aligned[test_mask]
        y_train = labels_aligned[train_mask]
        y_test = labels_aligned[test_mask]

        return X_train, X_test, y_train, y_test

    def train_all_models(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        output_dir: str = "models"
    ) -> Dict[str, Dict]:
        """
        Train all models using walk-forward validation

        Trains:
        - 8 binary models for each touch label
        - 1 multiclass model for first touch

        Args:
            features_df: Feature matrix
            labels_df: Labels DataFrame
            output_dir: Directory to save models

        Returns:
            Dictionary with training results
        """
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'binary_models': {},
            'multiclass_models': {},
            'walk_forward_results': []
        }

        # Encode first_touch for multiclass
        class_to_idx = {c: i for i, c in enumerate(FIRST_TOUCH_CLASSES)}
        labels_df = labels_df.copy()
        labels_df['first_touch_encoded'] = labels_df['first_touch'].map(class_to_idx)

        for test_year in tqdm(self.wf_config.test_years, desc="Walk-forward years"):
            print(f"\n{'='*50}")
            print(f"Training models for test year: {test_year}")
            print(f"{'='*50}")

            X_train, X_test, y_train, y_test = self.walk_forward_split(
                features_df, labels_df, test_year
            )

            if len(X_train) < 100 or len(X_test) < 10:
                print(f"Skipping {test_year}: insufficient data")
                continue

            # Use last 20% of training as validation for early stopping
            val_size = int(len(X_train) * 0.2)
            X_train_fit = X_train.iloc[:-val_size]
            X_val = X_train.iloc[-val_size:]
            y_train_fit = y_train.iloc[:-val_size]
            y_val = y_train.iloc[-val_size:]

            year_results = {
                'test_year': test_year,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'binary_results': {},
                'multiclass_results': {}
            }

            # Train binary models
            for target in TARGET_LABELS:
                print(f"  Training binary model: {target}")

                model = self.train_binary_model(
                    X_train_fit[ALL_FEATURES],
                    y_train_fit[target],
                    X_val[ALL_FEATURES],
                    y_val[target]
                )

                # Save model
                model_path = os.path.join(
                    output_dir,
                    f"binary_{target}_{test_year}.joblib"
                )
                joblib.dump(model, model_path)

                # Get predictions for calibration
                train_probs = model.predict_proba(X_train[ALL_FEATURES])[:, 1]
                test_probs = model.predict_proba(X_test[ALL_FEATURES])[:, 1]

                year_results['binary_results'][target] = {
                    'train_probs': train_probs,
                    'train_labels': y_train[target].values,
                    'test_probs': test_probs,
                    'test_labels': y_test[target].values,
                    'model_path': model_path
                }

            # Train multiclass model
            print(f"  Training multiclass model: first_touch")

            multiclass_model = self.train_multiclass_model(
                X_train_fit[ALL_FEATURES],
                y_train_fit['first_touch_encoded'],
                X_val[ALL_FEATURES],
                y_val['first_touch_encoded']
            )

            # Save model
            model_path = os.path.join(
                output_dir,
                f"multiclass_first_touch_{test_year}.joblib"
            )
            joblib.dump(multiclass_model, model_path)

            # Get predictions
            train_probs = multiclass_model.predict_proba(X_train[ALL_FEATURES])
            test_probs = multiclass_model.predict_proba(X_test[ALL_FEATURES])

            year_results['multiclass_results'] = {
                'train_probs': train_probs,
                'train_labels': y_train['first_touch_encoded'].values,
                'test_probs': test_probs,
                'test_labels': y_test['first_touch_encoded'].values,
                'model_path': model_path
            }

            results['walk_forward_results'].append(year_results)

        # Train final models on all data for production
        print(f"\n{'='*50}")
        print("Training final production models on all data")
        print(f"{'='*50}")

        val_size = int(len(features_df) * 0.1)
        X_train_final = features_df.iloc[:-val_size]
        X_val_final = features_df.iloc[-val_size:]
        y_train_final = labels_df.iloc[:-val_size]
        y_val_final = labels_df.iloc[-val_size:]

        # Binary models
        for target in TARGET_LABELS:
            print(f"  Training final binary model: {target}")

            model = self.train_binary_model(
                X_train_final[ALL_FEATURES],
                y_train_final[target],
                X_val_final[ALL_FEATURES],
                y_val_final[target]
            )

            model_path = os.path.join(output_dir, f"binary_{target}_final.joblib")
            joblib.dump(model, model_path)
            results['binary_models'][target] = model_path

        # Multiclass model
        print(f"  Training final multiclass model: first_touch")

        multiclass_model = self.train_multiclass_model(
            X_train_final[ALL_FEATURES],
            y_train_final['first_touch_encoded'],
            X_val_final[ALL_FEATURES],
            y_val_final['first_touch_encoded']
        )

        model_path = os.path.join(output_dir, "multiclass_first_touch_final.joblib")
        joblib.dump(multiclass_model, model_path)
        results['multiclass_models']['first_touch'] = model_path

        return results

    def get_feature_importance(
        self,
        model_path: str,
        feature_names: List[str] = None
    ) -> pd.DataFrame:
        """Get feature importance from a trained model"""
        model = joblib.load(model_path)
        feature_names = feature_names or ALL_FEATURES

        importance = model.feature_importances_

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        df = df.sort_values('importance', ascending=False)

        return df


def create_ensemble_predictions(
    models: Dict[str, str],
    X: pd.DataFrame,
    feature_names: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Generate predictions from all models

    Args:
        models: Dict of target -> model_path
        X: Feature DataFrame
        feature_names: List of feature names to use

    Returns:
        Dict of target -> probability array
    """
    feature_names = feature_names or ALL_FEATURES
    predictions = {}

    for target, model_path in models.items():
        model = joblib.load(model_path)
        probs = model.predict_proba(X[feature_names])

        if probs.shape[1] == 2:
            # Binary model
            predictions[target] = probs[:, 1]
        else:
            # Multiclass model
            predictions[target] = probs

    return predictions


if __name__ == "__main__":
    # Test training pipeline
    from data_loader import load_and_prepare_data
    from features import FeatureEngineer
    from labels import LabelGenerator

    # Load data
    sessions, daily = load_and_prepare_data(
        start_year=2020,  # Use shorter period for testing
        end_year=2024
    )

    # Build features
    engineer = FeatureEngineer()
    features_df = engineer.build_feature_matrix(sessions, daily)

    # Build labels
    generator = LabelGenerator()
    labels_df, levels_df = generator.build_label_dataframe(sessions, daily)

    # Train models
    trainer = WalkForwardTrainer()

    # Use only 2023-2024 for testing
    trainer.wf_config.test_years = [2023, 2024]

    results = trainer.train_all_models(
        features_df,
        labels_df,
        output_dir="models"
    )

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"{'='*50}")
    print(f"Binary models: {list(results['binary_models'].keys())}")
    print(f"Multiclass models: {list(results['multiclass_models'].keys())}")
    print(f"Walk-forward results: {len(results['walk_forward_results'])} years")
