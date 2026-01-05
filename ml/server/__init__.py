"""
SPR-Refactored Prediction Server

Modular structure:
- models/   : Model loading and management
- data/     : Polygon API and data fetching
- v6/       : V6 time-split model predictions
- features/ : Feature engineering
- routes/   : Flask endpoints
- utils/    : Shared utilities
"""

from flask import Flask
from flask_cors import CORS

def create_app():
    """Create and configure the Flask app"""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    from .routes import health, predictions, signals, analysis
    app.register_blueprint(health.bp)
    app.register_blueprint(predictions.bp)
    app.register_blueprint(signals.bp)
    app.register_blueprint(analysis.bp)

    # Load models on startup
    from .models import loader
    loader.load_all_models()

    return app
