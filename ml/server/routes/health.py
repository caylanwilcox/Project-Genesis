"""
Health Check Endpoints

Basic server health and status endpoints.
"""

import os
from flask import Blueprint, jsonify
from ..models.store import intraday_v6_models, swing_v6_models, swing_3d_models, swing_1d_models
from ..config import SUPPORTED_TICKERS, ACTIVE_MODEL_VERSION, MODELS_DIR

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
            'swing_loaded': ticker in swing_v6_models,
            'swing_3d_loaded': ticker in swing_3d_models,
            'swing_1d_loaded': ticker in swing_1d_models,
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


@bp.route('/debug/files', methods=['GET'])
def debug_files():
    """Debug endpoint to check model files on disk"""
    files_info = {
        'models_dir': MODELS_DIR,
        'models_dir_exists': os.path.exists(MODELS_DIR),
        'files': [],
        'lfs_pointers': []
    }

    if os.path.exists(MODELS_DIR):
        for f in os.listdir(MODELS_DIR):
            filepath = os.path.join(MODELS_DIR, f)
            size = os.path.getsize(filepath)
            files_info['files'].append({
                'name': f,
                'size_bytes': size,
                'size_mb': round(size / 1024 / 1024, 2)
            })

            # Check if it's an LFS pointer (small text file starting with "version")
            if size < 500:
                try:
                    with open(filepath, 'r') as fp:
                        content = fp.read(100)
                        if content.startswith('version https://git-lfs'):
                            files_info['lfs_pointers'].append(f)
                except:
                    pass

    return jsonify(files_info)
