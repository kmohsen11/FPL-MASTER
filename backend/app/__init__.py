import os
from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

db = SQLAlchemy()
migrate = Migrate()

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('postgres://u51dk6770aoold:pe5a305e584f985af8bf58fded594bc70cb31126ee1e6d733b7d88ae9f38ab4a0@c3gtj1dt5vh48j.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/daq2l689qp9u84')
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