import os
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL','postgresql://neondb_owner:npg_GpF01WdNIYBM@ep-frosty-dream-a5ya4h7n-pooler.us-east-2.aws.neon.tech/neondb?sslmode=require')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'Z9Q7E2KzC6b1f6z2qz9g3mE7P4q8zN0l')

    CORS(app)
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from .routes import api
        app.register_blueprint(api, url_prefix='/api')
        db.create_all()

    return app