import sys
import os
import pandas as pd
from sqlalchemy.exc import IntegrityError
from app.models import db, Team, Player, PlayerRoundPerformance
from app import create_app  # Import the application factory function

# Create the Flask app and set the context
app = create_app()

# File paths for CSV files
mid_predictions_path = "/Users/nayeb/Desktop/FPL-Model/performance/mid_highest_predictions.csv"
gk_predictions_path = "/Users/nayeb/Desktop/FPL-Model/performance/gk_highest_predictions.csv"
def_predictions_path = "/Users/nayeb/Desktop/FPL-Model/performance/def_highest_predictions.csv"
fwd_predictions_path = "/Users/nayeb/Desktop/FPL-Model/performance/fwd_highest_predictions.csv"
merged_gw_path = "/Users/nayeb/Desktop/FPL-Model/data/2024-25/gws/merged_gw.csv"

# Load the merged_gw.csv file
merged_gw = pd.read_csv(merged_gw_path)

# Function to get the full team name using player's full name
def get_team_name_by_player_name(player_name):
    team_row = merged_gw[merged_gw['name'] == player_name]
    return team_row['team'].values[0] if not team_row.empty else "Unknown"

# Function to load and process CSV data
def load_csv_data(filepath, position):
    data = pd.read_csv(filepath)
    data["position"] = position  # Add position to data
    return data

# Load all CSV files
mid_predictions = load_csv_data(mid_predictions_path, "MID")
gk_predictions = load_csv_data(gk_predictions_path, "GK")
def_predictions = load_csv_data(def_predictions_path, "DEF")
fwd_predictions = load_csv_data(fwd_predictions_path, "FWD")

# Combine all data into one DataFrame
all_data = pd.concat([mid_predictions, gk_predictions, def_predictions, fwd_predictions])

# Start populating the database
def populate_database(data):
    for _, row in data.iterrows():
        # Get the full team name using player's full name
        team_name = get_team_name_by_player_name(row["decoded_name"])

        # Check if the team exists, if not create it
        team = Team.query.filter_by(name=team_name).first()
        if not team:
            team = Team(name=team_name)
            db.session.add(team)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()

        # Check if the player exists, if not create them
        player = Player.query.filter_by(web_name=row["decoded_name"]).first()
        if not player:
            player = Player(
                first_name=row["decoded_name"].split(" ")[0],
                second_name=" ".join(row["decoded_name"].split(" ")[1:]),
                web_name=row["decoded_name"],
                team_id=team.id,
                position=row["position"],
            )
            db.session.add(player)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()

        # Add performance data
        performance = PlayerRoundPerformance(
            player_id=player.id,
            season="2024-25",  # Assuming a fixed season for now
            round=1,  # Placeholder; adjust as needed
            predicted_points=row["predicted_points_after"],
            opponent_team_id=team.id,  # Placeholder; adjust with actual opponent data
        )
        db.session.add(performance)
        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()

# Run the script within the Flask application context
with app.app_context():
    populate_database(all_data)