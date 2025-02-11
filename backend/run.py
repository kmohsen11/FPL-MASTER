import os
from backend.app import create_app

# Get the port from the environment variable, default to 5000 if not set
port = int(os.environ.get("PORT", 5000))

# Create the Flask app using the create_app function
app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
