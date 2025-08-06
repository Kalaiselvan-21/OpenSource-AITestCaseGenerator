from flask import Flask
from flask_cors import CORS
import os
import sys

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

def create_app():
    app = Flask(__name__)
    CORS(app)  # Enable CORS for all routes
    
    # Import and register routes
    from backend.api.routes import api_bp
    app.register_blueprint(api_bp)
    
    return app

if __name__ == '__main__':
    # Add the parent directory to the path
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
    
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)
