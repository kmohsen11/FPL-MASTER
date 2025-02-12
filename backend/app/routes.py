from flask import Blueprint, jsonify, request
#import db from its bath
from . import db
from .models import Player  # Corrected import
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError
import numpy as np
from collections import Counter
from scipy.optimize import linprog
from rapidfuzz import process  # Import for fuzzy string matching
import schedule
from app.populate import fetch_merged_gw, clean_database, load_predictions, populate_database  
import requests
from bs4 import BeautifulSoup
#import squads from pipeline
# from app.pipeline import squads
import json
import os

# Create a Blueprint for the API
api = Blueprint('api', __name__)

# Ensure correct path to squads.json (relative to backend/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQUAD_FILE = os.path.join(BASE_DIR, "squads.json")

def load_squads():
    """Load squads from JSON file if it exists."""
    if os.path.exists(SQUAD_FILE):
        with open(SQUAD_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    print(f"❌ squads.json not found at {SQUAD_FILE}!")
    return {}  # Return an empty dictionary if file does not exist


# Load squads at server startup
squads = load_squads()


@api.route('/squads', methods=['GET'])
def get_all_squads():
    """Return the cached Premier League squads."""
    return jsonify(squads)

@api.route('/search', methods=['GET'])
def search_players():
    try:
        query = request.args.get('query')

        # Fetch all players from the database
        players = Player.query.options(joinedload(Player.round_performances)).all()

        # Extract player names and points
        player_data = [
            {
                "name": player.web_name,
                "team": player.team.name if player.team else "Unknown",
                "points": max([performance.predicted_points for performance in player.round_performances], default=0)
            }
            for player in players
        ]

        # Use fuzzy string matching to find close matches
        matches = process.extract(query, [p["name"] for p in player_data], limit=10)

        # Return top matches with predicted points
        results = [{"name": match[0], "points": next(p["points"] for p in player_data if p["name"] == match[0])} for match in matches]

        return jsonify({"matches": results})

    except Exception as e:
        print(f"Error in /search: {e}")
        return jsonify({"error": "Failed to fetch players"}), 500

@api.route('/best-squad', methods=['GET'])
def get_best_squad():
    """Fetch the best squad based purely on predicted points and positions (no validity checks)."""
    try:
        with current_app.app_context():  # Ensure this is within the app context
            print("🔍 Fetching best squad...")
       

            # ✅ Position Mapping (Convert number-based positions)
            POSITION_MAP = {"GK": "GK", "DEF": "DEF", "MID": "MID", "FWD": "FWD"}
            # ✅ Fetch all players with predicted points
            players = Player.query.options(joinedload(Player.round_performances)).all()
            print(f"✅ Total players fetched: {len(players)}")

            # ✅ Extract player details **without checking validity**
            all_players = [
                {
                    "id": player.id,
                    "name": player.web_name,
                    "team": player.team.name if player.team else "Unknown",
                    "position": POSITION_MAP.get(player.position, "UNK"),  # Convert positions
                    "predicted_points": max([p.predicted_points for p in player.round_performances], default=0)
                }
                for player in players
            ]

            # ✅ Sort players by predicted points (highest first)
            sorted_players = sorted(all_players, key=lambda x: x["predicted_points"], reverse=True)
            print(f"✅ Players sorted by predicted points: {sorted_players[:3]}")
            # ✅ **Categorize Players**
            goalkeepers = [p for p in sorted_players if p["position"] == "GK"]
            defenders = [p for p in sorted_players if p["position"] == "DEF"]
            midfielders = [p for p in sorted_players if p["position"] == "MID"]
            forwards = [p for p in sorted_players if p["position"] == "FWD"]

            print(f"🧤 GK: {len(goalkeepers)}, 🛡️ DEF: {len(defenders)}, ⚡ MID: {len(midfielders)}, 🔥 FWD: {len(forwards)}")

            # ✅ **Ensure correct squad formation (15 players total)**
            selected_gk = goalkeepers[:2]  # **Top 2 GKs**
            selected_def = defenders[:5]   # **Top 5 DEF**
            selected_mid = midfielders[:5] # **Top 5 MID**
            selected_fwd = forwards[:3]    # **Top 3 FWD**

            # ✅ **Starting XI (11 players)**
            main_squad = [
                selected_gk[0],  # **1 GK in starting XI**
                *selected_def[:4],  # **4 Defenders**
                *selected_mid[:4],  # **4 Midfielders**
                *selected_fwd[:2],  # **2 Forwards**
            ]

            # ✅ **Bench (4 Players)**
            bench = [
                selected_gk[1],  # **Backup GK**
                selected_def[4],  # **1 extra Defender**
                selected_mid[4],  # **1 extra Midfielder**
                selected_fwd[2],  # **1 extra Forward**
            ]

        return jsonify({"main": main_squad, "bench": bench})

    except Exception as e:
        print(f"❌ Error in /best-squad: {str(e)}")
        return jsonify({"error": f"Failed to fetch best squad: {str(e)}"}), 500


def optimize_squad(players, squad_constraints):
    # Player data
    predicted_points = np.array([p["predicted_points"] for p in players])

    # ✅ Ensure it's a 1D array
    predicted_points = predicted_points.flatten()

    # Check the shape of predicted_points before passing to linprog
    print(f"Shape of predicted_points: {predicted_points.shape}")  # Debugging

    positions = [p["position"] for p in players]
    teams = [p["team"] for p in players]

    # Constraints
    num_players = len(players)
    A = []
    b = []

    # Position constraints (e.g., specific counts for GK, DEF, MID, FWD)
    for position, count in squad_constraints.items():
        if position in ["GK", "DEF", "MID", "FWD"]:
            A.append([1 if positions[i] == position else 0 for i in range(num_players)])
            b.append(count)

    # Specific constraint: Exactly 1 main GK and 1 bench GK
    A.append([1 if positions[i] == "GK" else 0 for i in range(num_players)])
    b.append(2)  # Total GKs = 1 main + 1 bench

    # Team constraints (e.g., max players per team)
    team_counts = Counter(teams)
    for team, _ in team_counts.items():
        A.append([1 if teams[i] == team else 0 for i in range(num_players)])
        b.append(squad_constraints["max_team_players"])

    # Convert A and b into NumPy arrays and ensure their shapes match
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # ✅ Ensure bounds are formatted correctly
    bounds = [(0, 1) for _ in range(num_players)]

    # ✅ Ensure A_ub and b_ub have matching dimensions
    print(f"A.shape: {A.shape}, b.shape: {b.shape}")  # Debugging

    # ✅ Fix for A_ub shape mismatch (transpose if needed)
    if A.shape[1] != num_players:
        A = A.T

    # Linear programming: maximize predicted points
    result = linprog(-predicted_points, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # Check if optimization succeeded
    if result.success:
        selected_indices = [i for i in range(num_players) if result.x[i] > 0.5]
        selected_players = [players[i] for i in selected_indices]

        # Ensure one GK is main and one is bench
        goalkeepers = [p for p in selected_players if p["position"] == "GK"]
        if len(goalkeepers) == 2:
            goalkeepers.sort(key=lambda x: x["predicted_points"], reverse=True)
            main_gk = goalkeepers[0]
            bench_gk = goalkeepers[1]
        else:
            raise ValueError("Invalid GK selection. Ensure exactly 2 GKs are selected.")

        # Split main and bench players
        main_squad = [main_gk]
        bench_squad = [bench_gk]
        for player in selected_players:
            if player["position"] != "GK":
                if len(main_squad) < 11:
                    main_squad.append(player)
                else:
                    bench_squad.append(player)

        return {"main": main_squad, "bench": bench_squad}

    else:
        raise ValueError(f"Optimization failed")


@api.route('/new_predictions', methods=['GET'])
def new_predictions_info():
    """
    Endpoint to get the status of the prediction pipeline schedule.
    """
    try:
        # Fetch scheduled jobs
        jobs = schedule.get_jobs()
        if jobs:
            # Get the next scheduled run time
            next_run = jobs[0].next_run.isoformat() if jobs[0].next_run else "Not scheduled"
            return jsonify({"message": "Prediction pipeline is scheduled.", "next_run": next_run}), 200
        else:
            return jsonify({"message": "No prediction pipeline is currently scheduled.", "next_run": None}), 200
    except Exception as e:
        print(f"Error in /new_predictions: {e}")
        return jsonify({"error": "Failed to get next prediction run time"}), 500


@api.route('/run_new_predictions', methods=['POST'])
def run_new_predictions():
    """
    Endpoint to manually run the prediction pipeline.
    """
    try:
        from app.update_predictions import run_pipeline  # Import the pipeline function

        # Run the pipeline and update the database
        print("Starting the prediction pipeline...")
        result = run_pipeline()
        if "error" in result:
            raise Exception(result["error"])

        # Fetch the merged game week data and populate the database
        print("Fetching game week data...")
        merged_gw = fetch_merged_gw()
        print("Cleaning the database...")
        clean_database()
        print("Loading predictions...")
        predictions_data = load_predictions()
        print("Populating the database...")
        populate_database(predictions_data, merged_gw)

        return jsonify({"message": "Prediction pipeline executed and database updated successfully."}), 200
    except Exception as e:
        print(f"Error in /run_new_predictions: {e}")
        return jsonify({"error": f"Failed to run prediction pipeline: {str(e)}"}), 500