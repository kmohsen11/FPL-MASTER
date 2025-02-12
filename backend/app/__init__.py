from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate  # Flask-Migrate for handling migrations
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

   

    # Ensure all tables are created
    with app.app_context():
        # Register Blueprints for routes
        from .routes import api  # Make sure this import is correct
        app.register_blueprint(api, url_prefix='/api')
        db.create_all()  # This creates tables if they don't already exist.

    return app
