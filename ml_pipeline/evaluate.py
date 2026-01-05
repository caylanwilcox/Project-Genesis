"""
Evaluation and Metrics for SPY Daily Range Plan Models
Comprehensive performance analysis across walk-forward windows
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, brier_score_loss, log_loss,
    confusion_matrix, classification_report
)
import json
import os

from config import TARGET_LABELS, FIRST_TOUCH_CLASSES
from calibrate import evaluate_calibration


@dataclass
class ModelMetrics:
    """Metrics for a single model"""
    target: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float
    brier: float
    log_loss_val: float
    ece: float  # Expected Calibration Error
    positive_rate: float
    n_samples: int


@dataclass
class WalkForwardMetrics:
    """Metrics across walk-forward validation"""
    target: str
    years: List[int]
    metrics_by_year: Dict[int, ModelMetrics]
    aggregate_metrics: ModelMetrics
    stability_score: float  # How stable is performance across years


class ModelEvaluator:
    """Evaluates model performance with comprehensive metrics"""

    def evaluate_binary_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        target: str,
        threshold: float = 0.5
    ) -> ModelMetrics:
        """
        Evaluate binary classification model

        Args:
            y_true: True labels
            y_prob: Predicted probabilities
            target: Target name
            threshold: Classification threshold

        Returns:
            ModelMetrics object
        """
        y_pred = (y_prob >= threshold).astype(int)

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Probabilistic metrics
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except ValueError:
            auc_roc = 0.5  # Single class present

        brier = brier_score_loss(y_true, y_prob)

        try:
            ll = log_loss(y_true, y_prob)
        except ValueError:
            ll = 1.0

        # Calibration
        cal_metrics = evaluate_calibration(y_prob, y_true)
        ece = cal_metrics['ece']

        return ModelMetrics(
            target=target,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            auc_roc=auc_roc,
            brier=brier,
            log_loss_val=ll,
            ece=ece,
            positive_rate=y_true.mean(),
            n_samples=len(y_true)
        )

    def evaluate_multiclass_model(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """
        Evaluate multiclass classification model

        Returns:
            Dictionary with per-class and aggregate metrics
        """
        y_pred = y_prob.argmax(axis=1)

        # Overall accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Per-class metrics
        per_class = {}
        for class_idx, class_name in enumerate(FIRST_TOUCH_CLASSES):
            class_true = (y_true == class_idx).astype(int)
            class_prob = y_prob[:, class_idx]
            class_pred = (y_pred == class_idx).astype(int)

            per_class[class_name] = {
                'precision': precision_score(class_true, class_pred, zero_division=0),
                'recall': recall_score(class_true, class_pred, zero_division=0),
                'f1': f1_score(class_true, class_pred, zero_division=0),
                'support': class_true.sum(),
            }

            try:
                per_class[class_name]['auc_roc'] = roc_auc_score(class_true, class_prob)
            except ValueError:
                per_class[class_name]['auc_roc'] = 0.5

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        return {
            'accuracy': accuracy,
            'per_class': per_class,
            'confusion_matrix': cm.tolist(),
            'n_samples': len(y_true)
        }

    def evaluate_walk_forward(
        self,
        walk_forward_results: List[Dict]
    ) -> Dict[str, WalkForwardMetrics]:
        """
        Evaluate performance across walk-forward windows

        Args:
            walk_forward_results: List of year results from training

        Returns:
            Dict of target -> WalkForwardMetrics
        """
        all_metrics = {}

        # Binary models
        for target in TARGET_LABELS:
            years = []
            metrics_by_year = {}
            all_probs = []
            all_labels = []

            for year_result in walk_forward_results:
                if target not in year_result['binary_results']:
                    continue

                year = year_result['test_year']
                result = year_result['binary_results'][target]

                probs = result['test_probs']
                labels = result['test_labels']

                if isinstance(probs, list):
                    probs = np.array(probs)
                if isinstance(labels, list):
                    labels = np.array(labels)

                metrics = self.evaluate_binary_model(
                    labels, probs, target
                )

                years.append(year)
                metrics_by_year[year] = metrics
                all_probs.extend(probs)
                all_labels.extend(labels)

            if not years:
                continue

            # Aggregate metrics
            aggregate = self.evaluate_binary_model(
                np.array(all_labels),
                np.array(all_probs),
                target
            )

            # Calculate stability (std of AUC across years)
            aucs = [m.auc_roc for m in metrics_by_year.values()]
            stability = 1.0 - np.std(aucs) if len(aucs) > 1 else 1.0

            all_metrics[target] = WalkForwardMetrics(
                target=target,
                years=years,
                metrics_by_year=metrics_by_year,
                aggregate_metrics=aggregate,
                stability_score=stability
            )

        return all_metrics

    def generate_report(
        self,
        wf_metrics: Dict[str, WalkForwardMetrics],
        multiclass_results: Optional[Dict] = None
    ) -> str:
        """Generate comprehensive evaluation report"""
        lines = [
            "=" * 70,
            "SPY DAILY RANGE PLAN - MODEL EVALUATION REPORT",
            "=" * 70,
            "",
            "BINARY TOUCH MODELS",
            "-" * 70,
        ]

        # Summary table
        lines.append(f"{'Target':<20} {'AUC':<8} {'Brier':<8} {'ECE':<8} {'Stability':<10} {'Pos Rate':<10}")
        lines.append("-" * 70)

        for target, metrics in sorted(wf_metrics.items()):
            agg = metrics.aggregate_metrics
            lines.append(
                f"{target:<20} {agg.auc_roc:<8.3f} {agg.brier:<8.3f} "
                f"{agg.ece:<8.3f} {metrics.stability_score:<10.3f} {agg.positive_rate:<10.3f}"
            )

        lines.append("")
        lines.append("YEAR-BY-YEAR BREAKDOWN")
        lines.append("-" * 70)

        # Per-year table for each target
        for target, metrics in sorted(wf_metrics.items()):
            lines.append(f"\n{target}:")
            lines.append(f"  {'Year':<8} {'AUC':<8} {'Precision':<10} {'Recall':<8} {'F1':<8} {'N':<8}")

            for year in sorted(metrics.years):
                m = metrics.metrics_by_year[year]
                lines.append(
                    f"  {year:<8} {m.auc_roc:<8.3f} {m.precision:<10.3f} "
                    f"{m.recall:<8.3f} {m.f1:<8.3f} {m.n_samples:<8}"
                )

        # Multiclass results
        if multiclass_results:
            lines.extend([
                "",
                "=" * 70,
                "FIRST TOUCH MULTICLASS MODEL",
                "=" * 70,
                f"Overall Accuracy: {multiclass_results.get('accuracy', 0):.3f}",
                "",
                "Per-Class Metrics:",
                f"{'Class':<15} {'Precision':<10} {'Recall':<8} {'F1':<8} {'AUC':<8} {'Support':<8}",
                "-" * 60,
            ])

            for class_name, class_metrics in multiclass_results.get('per_class', {}).items():
                lines.append(
                    f"{class_name:<15} {class_metrics['precision']:<10.3f} "
                    f"{class_metrics['recall']:<8.3f} {class_metrics['f1']:<8.3f} "
                    f"{class_metrics['auc_roc']:<8.3f} {class_metrics['support']:<8}"
                )

        lines.extend([
            "",
            "=" * 70,
            "INTERPRETATION GUIDE",
            "=" * 70,
            "",
            "AUC-ROC: >0.7 good, >0.8 excellent (ranking quality)",
            "Brier Score: <0.25 good, <0.15 excellent (calibration)",
            "ECE: <0.10 well-calibrated, <0.05 excellent",
            "Stability: >0.8 consistent across years",
            "",
            "Note: Low positive rates (Pos Rate) indicate imbalanced targets.",
            "Focus on Brier and calibration for probability accuracy.",
        ])

        return "\n".join(lines)


class TradingSimulator:
    """Simulates trading based on model predictions"""

    def simulate_strategy(
        self,
        predictions: Dict[str, np.ndarray],
        actual_labels: pd.DataFrame,
        levels_df: pd.DataFrame,
        threshold: float = 0.6
    ) -> Dict:
        """
        Simulate trading strategy based on predictions

        Args:
            predictions: Dict of target -> probabilities
            actual_labels: DataFrame with actual outcomes
            levels_df: DataFrame with price levels
            threshold: Probability threshold for taking trades

        Returns:
            Simulation results dictionary
        """
        results = {
            'trades': [],
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0,
            'max_drawdown': 0,
        }

        equity = 10000  # Starting equity
        peak_equity = equity
        equity_curve = [equity]

        for i, (date, row) in enumerate(actual_labels.iterrows()):
            # Check if we have predictions for this date
            if i >= len(list(predictions.values())[0]):
                continue

            # Get predictions
            p_t1_long = predictions.get('touch_t1_long', np.array([0.5]))[i]
            p_sl_long = predictions.get('touch_sl_long', np.array([0.5]))[i]
            p_t1_short = predictions.get('touch_t1_short', np.array([0.5]))[i]
            p_sl_short = predictions.get('touch_sl_short', np.array([0.5]))[i]

            # Simple strategy: go long if P(T1) > threshold and P(SL) < 0.3
            trade = None
            if p_t1_long > threshold and p_sl_long < 0.3:
                trade = 'LONG'
            elif p_t1_short > threshold and p_sl_short < 0.3:
                trade = 'SHORT'

            if trade:
                # Calculate P&L based on actual outcome
                if trade == 'LONG':
                    if row['touch_t1_long']:
                        pnl = 0.5  # Hit T1 = 0.5 unit profit
                    elif row['touch_sl_long']:
                        pnl = -1.25  # Hit SL = 1.25 unit loss
                    else:
                        pnl = 0  # No outcome
                else:  # SHORT
                    if row['touch_t1_short']:
                        pnl = 0.5
                    elif row['touch_sl_short']:
                        pnl = -1.25
                    else:
                        pnl = 0

                results['trades'].append({
                    'date': str(date),
                    'direction': trade,
                    'pnl': pnl,
                    'p_target': p_t1_long if trade == 'LONG' else p_t1_short,
                    'p_stop': p_sl_long if trade == 'LONG' else p_sl_short,
                })

                results['total_trades'] += 1
                if pnl > 0:
                    results['winning_trades'] += 1

                results['total_pnl'] += pnl
                equity += pnl * 100  # Scale by $100 per unit
                equity_curve.append(equity)

                if equity > peak_equity:
                    peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown > results['max_drawdown']:
                    results['max_drawdown'] = drawdown

        # Calculate summary stats
        if results['total_trades'] > 0:
            results['win_rate'] = results['winning_trades'] / results['total_trades']
            results['avg_pnl'] = results['total_pnl'] / results['total_trades']
        else:
            results['win_rate'] = 0
            results['avg_pnl'] = 0

        results['final_equity'] = equity
        results['total_return'] = (equity - 10000) / 10000

        return results


def run_full_evaluation(
    walk_forward_results: List[Dict],
    output_dir: str = "outputs"
) -> Dict:
    """
    Run complete evaluation pipeline

    Args:
        walk_forward_results: Results from walk-forward training
        output_dir: Directory for output files

    Returns:
        Complete evaluation results
    """
    os.makedirs(output_dir, exist_ok=True)

    evaluator = ModelEvaluator()

    # Evaluate walk-forward
    wf_metrics = evaluator.evaluate_walk_forward(walk_forward_results)

    # Aggregate multiclass results
    multiclass_agg = None
    all_mc_probs = []
    all_mc_labels = []

    for year_result in walk_forward_results:
        if 'multiclass_results' in year_result:
            result = year_result['multiclass_results']
            all_mc_probs.append(result['test_probs'])
            all_mc_labels.extend(result['test_labels'])

    if all_mc_probs:
        probs = np.vstack(all_mc_probs)
        labels = np.array(all_mc_labels)
        multiclass_agg = evaluator.evaluate_multiclass_model(labels, probs)

    # Generate report
    report = evaluator.generate_report(wf_metrics, multiclass_agg)

    # Save report
    report_path = os.path.join(output_dir, "evaluation_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)

    print(report)
    print(f"\nReport saved to {report_path}")

    # Save metrics as JSON
    metrics_dict = {}
    for target, m in wf_metrics.items():
        metrics_dict[target] = {
            'aggregate': {
                'auc_roc': m.aggregate_metrics.auc_roc,
                'brier': m.aggregate_metrics.brier,
                'ece': m.aggregate_metrics.ece,
                'positive_rate': m.aggregate_metrics.positive_rate,
            },
            'stability': m.stability_score,
            'years': m.years
        }

    json_path = os.path.join(output_dir, "metrics.json")
    with open(json_path, 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    return {
        'walk_forward_metrics': wf_metrics,
        'multiclass_metrics': multiclass_agg,
        'report': report
    }


if __name__ == "__main__":
    # Test with mock data
    print("Running evaluation with mock data...")

    # Create mock walk-forward results
    np.random.seed(42)

    mock_results = []
    for year in [2020, 2021, 2022, 2023]:
        n_samples = 250

        year_result = {
            'test_year': year,
            'binary_results': {},
            'multiclass_results': {}
        }

        for target in TARGET_LABELS:
            # Generate somewhat realistic probabilities
            true_rate = 0.4 + np.random.random() * 0.2
            labels = (np.random.random(n_samples) < true_rate).astype(int)

            # Model has some predictive power
            base_prob = 0.3 + 0.4 * labels + 0.2 * np.random.random(n_samples)
            probs = np.clip(base_prob, 0, 1)

            year_result['binary_results'][target] = {
                'test_probs': probs,
                'test_labels': labels
            }

        # Multiclass
        n_classes = len(FIRST_TOUCH_CLASSES)
        labels = np.random.randint(0, n_classes, n_samples)
        probs = np.random.dirichlet(np.ones(n_classes), n_samples)

        year_result['multiclass_results'] = {
            'test_probs': probs,
            'test_labels': labels
        }

        mock_results.append(year_result)

    # Run evaluation
    results = run_full_evaluation(mock_results, output_dir="outputs")
