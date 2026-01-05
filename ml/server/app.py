"""
SPR Prediction Server - Main Entry Point

Modular Flask server for V6 and RPE predictions.

Usage:
    python -m server.app
    # or
    gunicorn server.app:app
"""

from flask import Flask
from flask_cors import CORS

from .config import POLYGON_API_KEY
from .models.loader import load_all_models
from .routes import health, predictions, signals, analysis


def create_app():
    """Create and configure the Flask application"""
    app = Flask(__name__)
    CORS(app)

    # Register blueprints
    app.register_blueprint(health.bp)
    app.register_blueprint(predictions.bp)
    app.register_blueprint(signals.bp)
    app.register_blueprint(analysis.bp)

    return app


# Create app instance
app = create_app()

# Load models on import
print(f"POLYGON_API_KEY loaded: {'Yes' if POLYGON_API_KEY else 'No'} (length: {len(POLYGON_API_KEY)})")
load_all_models()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
