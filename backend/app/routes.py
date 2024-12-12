from flask import Blueprint, jsonify, request
from app.models import Player, db
from sqlalchemy.orm import joinedload
from sqlalchemy.exc import IntegrityError
import numpy as np
from collections import Counter
from scipy.optimize import linprog
from rapidfuzz import process  # Import for fuzzy string matching

from app.populate import fetch_merged_gw, clean_database, load_predictions, populate_database  

# Create a Blueprint for the API
api = Blueprint('api', __name__)

#search for players
@api.route('/search', methods=['GET'])
def search_players():
    try:
        # Get query parameter
        query = request.args.get('query')

        # Fetch all players
        players = Player.query.all()

        # Extract player names and points
        player_data = [{"name": player.web_name, "points": max(
            [performance.predicted_points for performance in player.round_performances], default=0)} for player in players]

        # Use fuzzy string matching to find close matches
        matches = process.extract(query, [player["name"] for player in player_data], limit=10)

        # Combine matches with their points
        results = [{"name": match[0], "points": next(
            (p["points"] for p in player_data if p["name"] == match[0]), 0)} for match in matches]

        return jsonify({"matches": results})

    except Exception as e:
        print(f"Error in /api/search: {e}")
        return jsonify({"error": "Failed to fetch players"}), 500


@api.route('/best-squad', methods=['GET'])
def get_best_squad():
    try:
        # Fetch all players and prepare optimization data
        players = Player.query.options(joinedload(Player.round_performances)).all()
        data = [
            {
                "id": player.id,
                "name": player.web_name,
                "team": player.team.name if player.team else None,
                "position": player.position,
                "predicted_points": max([p.predicted_points for p in player.round_performances], default=0)
            }
            for player in players
        ]

        # Define FPL constraints
        squad_constraints = {
            "GK": 2,
            "DEF": 5,
            "MID": 5,
            "FWD": 3,
            "max_team_players": 3,
        }

        # Optimize squad
        best_squad = optimize_squad(data, squad_constraints)

        return jsonify(best_squad)

    except Exception as e:
        print(f"Error in /best-squad: {e}")
        return jsonify({"error": "Failed to fetch best squad"}), 500

def optimize_squad(players, squad_constraints):
    # Player data
    predicted_points = np.array([p["predicted_points"] for p in players])
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

    # Decision variables (binary: 1 if selected, 0 otherwise)
    bounds = [(0, 1) for _ in range(num_players)]

    # Linear programming: maximize predicted points
    result = linprog(-predicted_points, A_ub=A, b_ub=b, bounds=bounds, method="highs")

    # Parse results
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



@api.route('/new_predictions', methods=['GET'])
def new_predictions_info():
    """
    Endpoint to get the status of the prediction pipeline schedule.
    """
    try:
        from app import update_predictions
        # Provide information about the schedule
        return jsonify({"message": "Predictions are scheduled to run every Monday night at 11 PM."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
