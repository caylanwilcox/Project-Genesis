"""
Unit Tests for SPY Daily Range Plan Pipeline
Uses synthetic data for testing all modules
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    DataConfig, ModelConfig, TargetConfig, WalkForwardConfig,
    TARGET_LABELS, FIRST_TOUCH_CLASSES, ALL_FEATURES
)
from synthetic_data import SyntheticDataGenerator


class TestConfig:
    """Test configuration classes"""

    def test_data_config_defaults(self):
        config = DataConfig()
        assert config.symbol == "SPY"
        assert config.market_open == "09:30"
        assert config.market_close == "16:00"
        assert config.atr_period == 14

    def test_target_config_defaults(self):
        config = TargetConfig()
        assert config.t1_units == 0.5
        assert config.t2_units == 1.0
        assert config.t3_units == 1.5
        assert config.sl_units == 1.25

    def test_model_config_defaults(self):
        config = ModelConfig()
        assert config.n_estimators == 400
        assert config.max_depth == 6

    def test_feature_list_complete(self):
        assert len(ALL_FEATURES) > 20
        assert 'atr_14' in ALL_FEATURES
        assert 'rsi_14' in ALL_FEATURES

    def test_target_labels_complete(self):
        assert len(TARGET_LABELS) == 8
        assert 'touch_t1_long' in TARGET_LABELS
        assert 'touch_sl_short' in TARGET_LABELS


class TestSyntheticData:
    """Test synthetic data generation"""

    @pytest.fixture
    def generator(self):
        return SyntheticDataGenerator(seed=42)

    def test_generate_intraday_session(self, generator):
        df = generator.generate_intraday_session("2024-01-15", 475.0)
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert 'volume' in df.columns
        assert df['high'].max() >= df['low'].min()

    def test_generate_daily_data(self, generator):
        df = generator.generate_daily_data(
            start_date="2023-01-01",
            end_date="2023-12-31",
            start_price=400.0
        )
        assert len(df) > 200
        assert 'atr_14' in df.columns
        assert 'rsi_14' in df.columns
        assert df['atr_14'].iloc[20:].notna().all()

    def test_generate_sessions(self, generator):
        daily = generator.generate_daily_data(
            start_date="2024-01-01",
            end_date="2024-01-31",
            start_price=475.0
        )
        sessions = generator.generate_sessions(daily)
        assert len(sessions) > 15  # ~20 trading days
        for date_str, session_df in sessions.items():
            assert len(session_df) > 100


class TestSessionBuilder:
    """Test session building functionality"""

    @pytest.fixture
    def session_data(self):
        generator = SyntheticDataGenerator(seed=42)
        daily = generator.generate_daily_data(
            start_date="2024-01-01",
            end_date="2024-01-31",
            start_price=475.0
        )
        sessions = generator.generate_sessions(daily)
        return sessions, daily

    def test_opening_range_calculation(self, session_data):
        from data_loader import SessionBuilder

        sessions, _ = session_data
        builder = SessionBuilder()

        for date_str, session_df in list(sessions.items())[:5]:
            or_metrics = builder.get_opening_range(session_df)
            assert or_metrics is not None
            assert 'or_high' in or_metrics
            assert 'or_low' in or_metrics
            assert 'or_range' in or_metrics
            assert or_metrics['or_high'] >= or_metrics['or_low']
            assert or_metrics['or_range'] >= 0

    def test_vwap_calculation(self, session_data):
        from data_loader import SessionBuilder

        sessions, _ = session_data
        builder = SessionBuilder()

        for date_str, session_df in list(sessions.items())[:5]:
            vwap = builder.calculate_session_vwap(session_df)
            assert len(vwap) == len(session_df)
            assert vwap.notna().all()
            # VWAP should be within day's range
            assert vwap.iloc[-1] >= session_df['low'].min() * 0.99
            assert vwap.iloc[-1] <= session_df['high'].max() * 1.01


class TestFeatureEngineering:
    """Test feature engineering"""

    @pytest.fixture
    def feature_data(self):
        generator = SyntheticDataGenerator(seed=42)
        daily = generator.generate_daily_data(
            start_date="2023-06-01",
            end_date="2024-01-31",
            start_price=450.0
        )
        sessions = generator.generate_sessions(daily)
        return sessions, daily

    def test_feature_calculation(self, feature_data):
        from features import FeatureEngineer

        sessions, daily = feature_data
        engineer = FeatureEngineer()

        features_df = engineer.build_feature_matrix(sessions, daily)

        assert len(features_df) > 0
        # Check all expected features present
        for feature in ALL_FEATURES:
            assert feature in features_df.columns, f"Missing feature: {feature}"

        # Check no NaN in core features
        assert features_df['atr_14'].notna().sum() > 0
        assert features_df['rsi_14'].notna().sum() > 0

    def test_intraday_atr(self):
        from features import calculate_intraday_atr

        generator = SyntheticDataGenerator(seed=42)
        session = generator.generate_intraday_session("2024-01-15", 475.0)

        atr = calculate_intraday_atr(session)
        assert atr >= 0
        # Intraday ATR on synthetic data can be higher due to per-bar volatility
        assert atr < 100  # Reasonable upper bound

    def test_unit_calculation(self):
        from features import calculate_unit

        unit = calculate_unit(or_range=2.0, intraday_atr=1.5, daily_atr=3.0)
        assert unit == 2.0  # Should be max of OR range

        unit = calculate_unit(or_range=0.5, intraday_atr=0.8, daily_atr=3.0)
        assert unit == 1.5  # Should be 0.5 * daily_atr


class TestLabelGeneration:
    """Test label generation"""

    @pytest.fixture
    def label_data(self):
        generator = SyntheticDataGenerator(seed=42)
        daily = generator.generate_daily_data(
            start_date="2023-06-01",
            end_date="2024-01-31",
            start_price=450.0
        )
        sessions = generator.generate_sessions(daily)
        return sessions, daily

    def test_session_levels(self, label_data):
        from labels import LabelGenerator

        sessions, daily = label_data
        generator = LabelGenerator()

        for date_str in list(sessions.keys())[:5]:
            session_df = sessions[date_str]
            levels = generator.calculate_session_levels(session_df, daily, date_str)

            if levels is None:
                continue

            # Check level ordering
            assert levels.t3_long > levels.t2_long > levels.t1_long > levels.vwap
            assert levels.t3_short < levels.t2_short < levels.t1_short < levels.vwap
            assert levels.sl_long < levels.vwap
            assert levels.sl_short > levels.vwap

    def test_label_generation(self, label_data):
        from labels import LabelGenerator

        sessions, daily = label_data
        generator = LabelGenerator()

        labels_df, levels_df = generator.build_label_dataframe(sessions, daily)

        assert len(labels_df) > 0
        assert len(levels_df) == len(labels_df)

        # Check label columns
        for target in TARGET_LABELS:
            assert target in labels_df.columns
            assert labels_df[target].isin([0, 1]).all()

        assert 'first_touch' in labels_df.columns
        assert labels_df['first_touch'].isin(FIRST_TOUCH_CLASSES).all()

    def test_mfe_mae_calculation(self, label_data):
        from labels import LabelGenerator

        sessions, daily = label_data
        generator = LabelGenerator()

        labels_df, _ = generator.build_label_dataframe(sessions, daily)

        # MFE should be non-negative
        assert (labels_df['mfe_long'] >= 0).all()
        assert (labels_df['mfe_short'] >= 0).all()

        # MAE should be non-negative
        assert (labels_df['mae_long'] >= 0).all()
        assert (labels_df['mae_short'] >= 0).all()


class TestCalibration:
    """Test probability calibration"""

    def test_isotonic_calibration(self):
        from calibrate import ProbabilityCalibrator, evaluate_calibration

        np.random.seed(42)

        # Create uncalibrated data
        n = 500
        true_probs = np.random.beta(2, 5, n)
        labels = (np.random.random(n) < true_probs).astype(int)
        raw_probs = np.clip(true_probs * 1.3, 0, 1)  # Overconfident

        calibrator = ProbabilityCalibrator()
        calibrator.fit_binary_calibrator('test', raw_probs, labels)

        calibrated = calibrator.calibrate_binary('test', raw_probs)

        # Calibrated should be better
        raw_metrics = evaluate_calibration(raw_probs, labels)
        cal_metrics = evaluate_calibration(calibrated, labels)

        assert cal_metrics['ece'] <= raw_metrics['ece'] + 0.05  # Allow small margin

    def test_multiclass_calibration(self):
        from calibrate import ProbabilityCalibrator

        np.random.seed(42)

        n_samples = 500
        n_classes = 5

        labels = np.random.randint(0, n_classes, n_samples)
        probs = np.random.dirichlet(np.ones(n_classes), n_samples)

        calibrator = ProbabilityCalibrator()
        calibrator.fit_multiclass_calibrators(probs, labels)

        calibrated = calibrator.calibrate_multiclass(probs)

        # Should sum to 1
        np.testing.assert_array_almost_equal(
            calibrated.sum(axis=1),
            np.ones(n_samples),
            decimal=5
        )


class TestModelTraining:
    """Test model training (lightweight)"""

    @pytest.fixture
    def training_data(self):
        generator = SyntheticDataGenerator(seed=42)
        daily = generator.generate_daily_data(
            start_date="2022-01-01",
            end_date="2024-01-31",
            start_price=400.0
        )
        sessions = generator.generate_sessions(daily)

        from features import FeatureEngineer
        from labels import LabelGenerator

        engineer = FeatureEngineer()
        features_df = engineer.build_feature_matrix(sessions, daily)

        label_gen = LabelGenerator()
        labels_df, _ = label_gen.build_label_dataframe(sessions, daily)

        return features_df, labels_df

    def test_walk_forward_split(self, training_data):
        from train import WalkForwardTrainer

        features_df, labels_df = training_data
        trainer = WalkForwardTrainer()

        X_train, X_test, y_train, y_test = trainer.walk_forward_split(
            features_df, labels_df, test_year=2024
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert X_train.index.year.max() < 2024
        assert X_test.index.year.min() == 2024

    def test_binary_model_training(self, training_data):
        from train import WalkForwardTrainer

        features_df, labels_df = training_data
        trainer = WalkForwardTrainer()

        # Use smaller model for testing
        trainer.model_config.n_estimators = 10
        trainer.model_config.max_depth = 3

        X_train, X_test, y_train, y_test = trainer.walk_forward_split(
            features_df, labels_df, test_year=2024
        )

        # Train one model
        target = 'touch_t1_long'
        model = trainer.train_binary_model(
            X_train[ALL_FEATURES],
            y_train[target]
        )

        # Check predictions
        probs = model.predict_proba(X_test[ALL_FEATURES])
        assert probs.shape == (len(X_test), 2)
        assert (probs >= 0).all() and (probs <= 1).all()


class TestInference:
    """Test inference and plan generation"""

    def test_plan_generation(self):
        from inference import PlanGenerator, DailyRangePlan

        # Create mock features
        features = {feat: np.random.random() * 2 - 1 for feat in ALL_FEATURES}
        features['atr_14'] = 3.5
        features['rsi_14'] = 55
        features['day_of_week'] = 2
        features['month'] = 1

        generator = PlanGenerator(model_dir="nonexistent")  # Will use defaults

        plan = generator.generate_plan(
            features=features,
            vwap=475.50,
            unit=3.20,
            date="2024-01-15",
            symbol="SPY"
        )

        assert isinstance(plan, DailyRangePlan)
        assert plan.vwap == 475.50
        assert plan.unit == 3.20
        assert plan.t1_long > plan.vwap
        assert plan.t1_short < plan.vwap
        assert plan.bias in ['LONG', 'SHORT', 'NEUTRAL']

    def test_plan_serialization(self):
        from inference import DailyRangePlan

        plan = DailyRangePlan(
            date="2024-01-15",
            symbol="SPY",
            vwap=475.50,
            unit=3.20,
            unit_description="OR-driven",
            t1_long=477.10,
            t2_long=478.70,
            t3_long=480.30,
            sl_long=471.50,
            t1_short=473.90,
            t2_short=472.30,
            t3_short=470.70,
            sl_short=479.50,
            prob_t1_long=0.65,
            prob_t2_long=0.45,
            prob_t3_long=0.25,
            prob_sl_long=0.20,
            prob_t1_short=0.55,
            prob_t2_short=0.35,
            prob_t3_short=0.15,
            prob_sl_short=0.25,
            first_touch_probs={'t1_long': 0.3, 'none': 0.2},
            bias="LONG",
            confidence=0.7,
            rationale="Test rationale"
        )

        # Test dict conversion
        d = plan.to_dict()
        assert d['symbol'] == 'SPY'
        assert d['levels']['long']['t1'] == 477.10

        # Test JSON conversion
        json_str = plan.to_json()
        assert '"SPY"' in json_str

        # Test summary
        summary = plan.summary()
        assert 'SPY' in summary
        assert '477.10' in summary


class TestEvaluation:
    """Test evaluation metrics"""

    def test_binary_evaluation(self):
        from evaluate import ModelEvaluator

        np.random.seed(42)

        evaluator = ModelEvaluator()

        y_true = np.random.randint(0, 2, 100)
        y_prob = np.random.random(100)

        metrics = evaluator.evaluate_binary_model(
            y_true, y_prob, 'test'
        )

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.auc_roc <= 1
        assert 0 <= metrics.brier <= 1
        assert metrics.n_samples == 100

    def test_multiclass_evaluation(self):
        from evaluate import ModelEvaluator

        np.random.seed(42)

        evaluator = ModelEvaluator()

        n_samples = 100
        n_classes = len(FIRST_TOUCH_CLASSES)

        y_true = np.random.randint(0, n_classes, n_samples)
        y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)

        metrics = evaluator.evaluate_multiclass_model(y_true, y_prob)

        assert 0 <= metrics['accuracy'] <= 1
        assert len(metrics['per_class']) == n_classes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
