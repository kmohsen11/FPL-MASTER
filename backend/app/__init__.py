from flask import Flask, send_from_directory
from flask_cors import CORS

def create_app():
    app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")  # Serve React frontend

    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///example.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Dynamically configure CORS for API routes
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Initialize database and register blueprints
    from app.models import db
    db.init_app(app)
    from app.routes import api
    app.register_blueprint(api, url_prefix='/api')

    with app.app_context():
        db.create_all()

    # Serve React frontend
    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def serve_react_frontend(path):
        if path and path.startswith("api"):  # Don't serve React for API calls
            return "API Endpoint", 404
        elif path and path != "index.html":
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, "index.html")

    return app
