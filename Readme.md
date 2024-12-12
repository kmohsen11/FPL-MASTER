# FPL-MASTER

FPL-MASTER is a full-stack Fantasy Premier League (FPL) management tool designed to provide insights, recommend optimized squads, and fetch FPL-related data using a custom proxy server. The app integrates React.js for the frontend, Flask for the backend, and uses data from the official Fantasy Premier League API.

## Features

### Frontend
- Search for players by name with fuzzy matching.
- Display detailed player data including team, position, and predicted points.
- Show the best squad for the next round, including main and bench players.
- Fetch FPL fixtures and team data through a proxy.

### Backend
- Flask API with endpoints for searching players and optimizing squads.
- Integration with the Fantasy Premier League API via a Node.js proxy.
- Use of linear programming to recommend the best squad based on constraints (e.g., max team players, positions).

## Technologies Used

### Frontend
- **React.js**: For building the user interface.
- **CSS**: For styling the application.

### Backend
- **Flask**: For building the REST API.
- **SQLAlchemy**: For database ORM.
- **Node.js & Express**: Proxy server to interact with the Fantasy Premier League API.
- **SciPy**: For optimization algorithms.

## Setup Instructions

### Prerequisites
- [Node.js](https://nodejs.org/) installed.
- [Python 3.x](https://www.python.org/downloads/) installed.
- [npm](https://www.npmjs.com/) or [yarn](https://yarnpkg.com/) for managing dependencies.

### Clone the Repository
```bash
git clone https://github.com/kmohsen11/FPL-MASTER.git
cd FPL-MASTER
```

### Backend Setup
1. **Navigate to the Backend Directory:**
   ```bash
   cd backend
   ```
2. **Set up a Python Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate  # For Windows
   ```
3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the Flask Backend:**
   ```bash
   flask run
   ```

### Proxy Server Setup
1. **Navigate to the Proxy Directory:**
   ```bash
   cd proxy
   ```
2. **Install Node.js Dependencies:**
   ```bash
   npm install
   ```
3. **Run the Proxy Server:**
   ```bash
   node proxy.js
   ```

### Frontend Setup
1. **Navigate to the Frontend Directory:**
   ```bash
   cd frontend
   ```
2. **Install Dependencies:**
   ```bash
   npm install
   ```
3. **Run the React Frontend:**
   ```bash
   npm start
   ```
# Force new build
