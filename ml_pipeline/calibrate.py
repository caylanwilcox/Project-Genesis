"""
Probability Calibration for SPY Daily Range Plan Models
Uses isotonic regression for well-calibrated probability estimates
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import calibration_curve
import joblib
import os
from tqdm import tqdm

from config import TARGET_LABELS, FIRST_TOUCH_CLASSES


class ProbabilityCalibrator:
    """
    Calibrates model probabilities using isotonic regression

    Isotonic regression is non-parametric and works well for
    well-separated probability distributions
    """

    def __init__(self):
        self.calibrators: Dict[str, IsotonicRegression] = {}
        self.multiclass_calibrators: Dict[int, IsotonicRegression] = {}

    def fit_binary_calibrator(
        self,
        target: str,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> IsotonicRegression:
        """
        Fit isotonic regression calibrator for binary classification

        Args:
            target: Target name
            probabilities: Raw model probabilities
            labels: True binary labels

        Returns:
            Fitted IsotonicRegression
        """
        calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds='clip'
        )

        # Sort by probability for isotonic regression
        sorted_idx = np.argsort(probabilities)
        sorted_probs = probabilities[sorted_idx]
        sorted_labels = labels[sorted_idx]

        calibrator.fit(sorted_probs, sorted_labels)
        self.calibrators[target] = calibrator

        return calibrator

    def fit_multiclass_calibrators(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> Dict[int, IsotonicRegression]:
        """
        Fit calibrators for each class in multiclass problem

        Uses one-vs-rest approach with isotonic regression per class
        """
        n_classes = probabilities.shape[1]

        for class_idx in range(n_classes):
            # Binary labels for this class
            binary_labels = (labels == class_idx).astype(int)
            class_probs = probabilities[:, class_idx]

            calibrator = IsotonicRegression(
                y_min=0.0,
                y_max=1.0,
                out_of_bounds='clip'
            )

            sorted_idx = np.argsort(class_probs)
            calibrator.fit(
                class_probs[sorted_idx],
                binary_labels[sorted_idx]
            )

            self.multiclass_calibrators[class_idx] = calibrator

        return self.multiclass_calibrators

    def calibrate_binary(
        self,
        target: str,
        probabilities: np.ndarray,
        clip_min: float = 0.02,
        clip_max: float = 0.98
    ) -> np.ndarray:
        """
        Apply calibration to binary probabilities

        Clips to [clip_min, clip_max] to avoid overconfident 0%/100% predictions
        """
        if target not in self.calibrators:
            raise ValueError(f"No calibrator found for {target}")

        calibrated = self.calibrators[target].predict(probabilities)
        return np.clip(calibrated, clip_min, clip_max)

    def calibrate_multiclass(
        self,
        probabilities: np.ndarray
    ) -> np.ndarray:
        """
        Apply calibration to multiclass probabilities

        Calibrates each class independently then renormalizes
        """
        n_samples, n_classes = probabilities.shape
        calibrated = np.zeros_like(probabilities)

        for class_idx in range(n_classes):
            if class_idx in self.multiclass_calibrators:
                calibrated[:, class_idx] = self.multiclass_calibrators[class_idx].predict(
                    probabilities[:, class_idx]
                )
            else:
                calibrated[:, class_idx] = probabilities[:, class_idx]

        # Renormalize to sum to 1
        row_sums = calibrated.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        calibrated = calibrated / row_sums

        return calibrated

    def save(self, output_dir: str):
        """Save all calibrators to disk"""
        os.makedirs(output_dir, exist_ok=True)

        # Save binary calibrators
        for target, calibrator in self.calibrators.items():
            path = os.path.join(output_dir, f"calibrator_{target}.joblib")
            joblib.dump(calibrator, path)

        # Save multiclass calibrators
        for class_idx, calibrator in self.multiclass_calibrators.items():
            path = os.path.join(output_dir, f"calibrator_multiclass_{class_idx}.joblib")
            joblib.dump(calibrator, path)

    def load(self, input_dir: str):
        """Load all calibrators from disk"""
        # Load binary calibrators
        for target in TARGET_LABELS:
            path = os.path.join(input_dir, f"calibrator_{target}.joblib")
            if os.path.exists(path):
                self.calibrators[target] = joblib.load(path)

        # Load multiclass calibrators
        for class_idx in range(len(FIRST_TOUCH_CLASSES)):
            path = os.path.join(input_dir, f"calibrator_multiclass_{class_idx}.joblib")
            if os.path.exists(path):
                self.multiclass_calibrators[class_idx] = joblib.load(path)


def evaluate_calibration(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10
) -> Dict:
    """
    Evaluate calibration quality

    Returns:
        Dict with calibration metrics including ECE
    """
    # Calculate calibration curve
    prob_true, prob_pred = calibration_curve(
        labels,
        probabilities,
        n_bins=n_bins,
        strategy='uniform'
    )

    # Expected Calibration Error (ECE)
    bin_counts = np.histogram(probabilities, bins=n_bins, range=(0, 1))[0]
    bin_weights = bin_counts / len(probabilities)

    # Calculate ECE
    ece = 0
    for i in range(len(prob_true)):
        if i < len(bin_weights):
            ece += bin_weights[i] * abs(prob_true[i] - prob_pred[i])

    # Brier score
    brier = np.mean((probabilities - labels) ** 2)

    return {
        'ece': ece,
        'brier_score': brier,
        'prob_true': prob_true,
        'prob_pred': prob_pred,
        'mean_predicted_prob': probabilities.mean(),
        'actual_positive_rate': labels.mean()
    }


def build_calibration_pipeline(
    walk_forward_results: List[Dict],
    output_dir: str = "models"
) -> ProbabilityCalibrator:
    """
    Build calibration from walk-forward validation results

    Uses out-of-sample predictions from walk-forward to fit calibrators
    """
    calibrator = ProbabilityCalibrator()

    # Aggregate all out-of-sample predictions
    binary_probs = {target: [] for target in TARGET_LABELS}
    binary_labels = {target: [] for target in TARGET_LABELS}
    multiclass_probs = []
    multiclass_labels = []

    for year_result in walk_forward_results:
        # Binary models
        for target in TARGET_LABELS:
            if target in year_result['binary_results']:
                result = year_result['binary_results'][target]
                binary_probs[target].extend(result['test_probs'])
                binary_labels[target].extend(result['test_labels'])

        # Multiclass model
        if 'multiclass_results' in year_result:
            result = year_result['multiclass_results']
            multiclass_probs.append(result['test_probs'])
            multiclass_labels.extend(result['test_labels'])

    # Fit binary calibrators
    print("Fitting binary calibrators...")
    for target in tqdm(TARGET_LABELS):
        if binary_probs[target]:
            probs = np.array(binary_probs[target])
            labels = np.array(binary_labels[target])

            calibrator.fit_binary_calibrator(target, probs, labels)

            # Evaluate
            metrics = evaluate_calibration(probs, labels)
            print(f"  {target}: ECE={metrics['ece']:.4f}, Brier={metrics['brier_score']:.4f}")

    # Fit multiclass calibrators
    print("\nFitting multiclass calibrators...")
    if multiclass_probs:
        probs = np.vstack(multiclass_probs)
        labels = np.array(multiclass_labels)

        calibrator.fit_multiclass_calibrators(probs, labels)

        # Evaluate per-class
        for class_idx, class_name in enumerate(FIRST_TOUCH_CLASSES):
            class_probs = probs[:, class_idx]
            class_labels = (labels == class_idx).astype(int)
            metrics = evaluate_calibration(class_probs, class_labels)
            print(f"  {class_name}: ECE={metrics['ece']:.4f}")

    # Save calibrators
    calibrator.save(output_dir)
    print(f"\nCalibrators saved to {output_dir}")

    return calibrator


def create_reliability_diagram(
    probabilities: np.ndarray,
    labels: np.ndarray,
    title: str = "Calibration Plot",
    n_bins: int = 10
) -> Dict:
    """
    Create data for reliability diagram visualization

    Returns:
        Dict with plotting data
    """
    metrics = evaluate_calibration(probabilities, labels, n_bins)

    return {
        'title': title,
        'prob_true': metrics['prob_true'],
        'prob_pred': metrics['prob_pred'],
        'ece': metrics['ece'],
        'brier': metrics['brier_score'],
        'n_samples': len(probabilities)
    }


if __name__ == "__main__":
    # Test calibration with synthetic data
    np.random.seed(42)

    # Create synthetic uncalibrated probabilities
    n_samples = 1000
    true_probs = np.random.beta(2, 5, n_samples)
    labels = (np.random.random(n_samples) < true_probs).astype(int)

    # Simulate overconfident model
    raw_probs = np.clip(true_probs * 1.3, 0, 1)

    print("Before calibration:")
    metrics_before = evaluate_calibration(raw_probs, labels)
    print(f"  ECE: {metrics_before['ece']:.4f}")
    print(f"  Brier: {metrics_before['brier_score']:.4f}")

    # Calibrate
    calibrator = ProbabilityCalibrator()
    calibrator.fit_binary_calibrator('test', raw_probs, labels)

    # Apply calibration
    calibrated_probs = calibrator.calibrate_binary('test', raw_probs)

    print("\nAfter calibration:")
    metrics_after = evaluate_calibration(calibrated_probs, labels)
    print(f"  ECE: {metrics_after['ece']:.4f}")
    print(f"  Brier: {metrics_after['brier_score']:.4f}")

    # Test save/load
    calibrator.save("models")

    calibrator2 = ProbabilityCalibrator()
    calibrator2.load("models")

    calibrated_probs2 = calibrator2.calibrate_binary('test', raw_probs)
    assert np.allclose(calibrated_probs, calibrated_probs2)
    print("\nSave/load test passed!")
