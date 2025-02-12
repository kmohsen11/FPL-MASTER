from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
import os

# Initialize the database instance globally
db = SQLAlchemy()

# Initialize Flask-Migrate
migrate = Migrate()

def create_app():
    app = Flask(__name__)

    # Set up configuration from environment variables or defaults
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///example.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Avoid unnecessary overhead
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'Z9Q7E2KzC6b1f6z2qz9g3mE7P4q8zN0l')  # To handle sessions and CSRF

    # Allow CORS for all routes
    CORS(app)

    # Initialize the database with the app
    db.init_app(app)

    # Initialize Flask-Migrate with the app and database
    migrate.init_app(app, db)

    # Register Blueprints for routes
    from app.routes import api  # Ensure this import is correct
    app.register_blueprint(api, url_prefix='/api')

    # Avoid using db.create_all() for migrations – this is for development use only
    # You should be handling migrations with flask db migrate/upgrade commands

    return app
