from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Dynamically configure CORS   # Allow all origins for Heroku deployment
    
    CORS(app, resources = {r"/api/*": {"origins": "*"}})
    
    

    # Initialize database and register blueprints
    from app.models import db
    db.init_app(app)
    from app.routes import api
    app.register_blueprint(api, url_prefix='/api')

    with app.app_context():
        db.create_all()

    return app
