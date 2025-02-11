from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy

import os

# Initialize the database instance globally
db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # Set up configuration from environment variables or defaults
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///example.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Avoid unnecessary overhead
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'Z9Q7E2KzC6b1f6z2qz9g3mE7P4q8zN0l')  # To handle sessions and CSRF

    # 🔥 Allow CORS for all routes
    CORS(app)

    # Initialize the database with the app
    db.init_app(app)

    # Register Blueprints for routes
    from app.routes import api
    app.register_blueprint(api, url_prefix='/api')

    # Ensure all tables are created
    with app.app_context():
        db.create_all()

    return app
