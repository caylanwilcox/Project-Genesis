"""
Health Check Endpoints

Basic server health and status endpoints.
"""

from flask import Blueprint, jsonify
from ..models.store import intraday_v6_models
from ..config import SUPPORTED_TICKERS, ACTIVE_MODEL_VERSION

bp = Blueprint('health', __name__)


@bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'version': 'v2.0-spr',
        'active_model': ACTIVE_MODEL_VERSION,
        'v6_models_loaded': list(intraday_v6_models.keys()),
        'supported_tickers': SUPPORTED_TICKERS
    })


@bp.route('/models', methods=['GET'])
def list_models():
    """List all loaded models and their status"""
    models_status = {}

    for ticker in SUPPORTED_TICKERS:
        models_status[ticker] = {
            'v6_loaded': ticker in intraday_v6_models,
        }

        if ticker in intraday_v6_models:
            model = intraday_v6_models[ticker]
            models_status[ticker]['v6_accuracy'] = {
                'early': model.get('acc_early', 0),
                'late_a': model.get('acc_late_a', 0),
                'late_b': model.get('acc_late_b', 0),
            }

    return jsonify({
        'active_version': ACTIVE_MODEL_VERSION,
        'models': models_status
    })
