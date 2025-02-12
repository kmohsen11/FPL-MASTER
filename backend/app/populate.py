import os
import sys
import requests
import pandas as pd
from io import StringIO
from sqlalchemy.exc import IntegrityError
from . import db
from .models import Team, Player, PlayerRoundPerformance
from . import create_app  # Use relative import  # Import the application factory function
from .update_predictions import run_pipeline  # Import the pipeline runner

# Constants
MERGED_GW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"

def fetch_merged_gw():
    """Fetches the merged game week data from GitHub and ensures consistency."""
    response = requests.get(MERGED_GW_URL)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch merged_gw.csv. HTTP Status Code: {response.status_code}")

    raw_data = response.text.splitlines()  # Read the file line by line
    print(f"✅ First 5 lines from merged_gw.csv:\n{raw_data[:5]}")  # Debugging

    try:
        # Read CSV with error handling
        df = pd.read_csv(StringIO(response.text), on_bad_lines='skip', dtype=str)
        print(f"✅ Successfully loaded CSV with {df.shape[0]} rows and {df.shape[1]} columns.")

        # Check if expected columns exist, else raise an error
        expected_columns = [
            "element", "fixture", "opponent_team", "total_points", "was_home", "kickoff_time",
            "team_h_score", "team_a_score", "round", "minutes", "goals_scored", "assists",
            "clean_sheets", "goals_conceded", "own_goals", "penalties_saved", "penalties_missed",
            "yellow_cards", "red_cards", "saves", "bonus", "bps", "influence", "creativity",
            "threat", "ict_index", "value", "transfers_balance", "selected", "transfers_in",
            "transfers_out", "name", "team"
        ]
        
        missing_cols = [col for col in expected_columns if col not in df.columns]
        if missing_cols:
            print(f"❌ Warning: Missing columns in merged_gw.csv: {missing_cols}")

        return df
    except Exception as e:
        raise Exception(f"❌ Failed to parse merged_gw.csv: {e}")

def get_team_name_by_player_name(player_name, merged_gw):
    """Get the full team name using the player's name from `merged_gw`."""
    team_row = merged_gw[merged_gw['name'] == player_name]
    return team_row['team'].values[0] if not team_row.empty else "Unknown"

def clean_database():
    """Deletes all entries in the PlayerRoundPerformance, Player, and Team tables."""
    db.session.query(PlayerRoundPerformance).delete()
    db.session.query(Player).delete()
    db.session.query(Team).delete()
    db.session.commit()
    print("✅ Database cleaned successfully.")

def load_predictions():
    """Runs the prediction pipeline and loads predictions into a DataFrame."""
    pipeline_result = run_pipeline()
    if "error" in pipeline_result:
        raise Exception(pipeline_result["error"])

    predictions_dir = "predictions"
    files = {
        "MID": os.path.join(predictions_dir, "mid_predictions.csv"),
        "GK": os.path.join(predictions_dir, "gk_predictions.csv"),
        "DEF": os.path.join(predictions_dir, "def_predictions.csv"),
        "FWD": os.path.join(predictions_dir, "fwd_predictions.csv"),
    }

    def load_csv(filepath, position):
        if not os.path.exists(filepath):
            print(f"⚠️ Warning: {filepath} not found.")
            return pd.DataFrame()
        data = pd.read_csv(filepath)
        data["position"] = position
        return data

    prediction_dfs = [load_csv(path, pos) for pos, path in files.items()]
    return pd.concat(prediction_dfs, ignore_index=True) if prediction_dfs else pd.DataFrame()

def populate_database(data, merged_gw):
    """Populates the database with teams, players, and their predicted performances."""
    
    if data.empty:
        print("⚠️ No predictions found. Skipping database population.")
        return

    for _, row in data.iterrows():
        team_name = get_team_name_by_player_name(row["decoded_name"], merged_gw)

        # ✅ Ensure the team exists
        team = Team.query.filter_by(name=team_name).first()
        if not team:
            team = Team(name=team_name)
            db.session.add(team)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                team = Team.query.filter_by(name=team_name).first()  # Fetch after rollback

        # ✅ Ensure the player exists
        player = Player.query.filter_by(web_name=row["decoded_name"]).first()
        if not player:
            first_name, last_name = (row["decoded_name"].split(" ", 1) + [""])[:2]  # Handle single-word names
            player = Player(
                first_name=first_name.strip(),
                second_name=last_name.strip(),
                web_name=row["decoded_name"],
                team_id=team.id,
                position=row["position"],
            )
            db.session.add(player)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                player = Player.query.filter_by(web_name=row["decoded_name"]).first()

        # ✅ Ensure the performance entry is added
        performance = PlayerRoundPerformance.query.filter_by(
            player_id=player.id, season="2024-25", round=1
        ).first()

        if not performance:
            performance = PlayerRoundPerformance(
                player_id=player.id,
                season="2024-25",
                round=1,
                predicted_points=row["predicted_points_after"],
                opponent_team_id=team.id,
            )
            db.session.add(performance)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()
                print(f"⚠️ Skipping duplicate entry for {row['decoded_name']}.")

def main():
    """Main function to clean, fetch, and populate the database."""
    app = create_app()
    with app.app_context():
        print("🛠 Cleaning database...")
        clean_database()

        print("📥 Fetching merged game week data...")
        merged_gw = fetch_merged_gw()

        print("🔮 Running predictions pipeline...")
        predictions_data = load_predictions()

        print("📊 Populating database with predictions...")
        populate_database(predictions_data, merged_gw)

        print("✅ Database populated successfully!")

if __name__ == "__main__":
    main()
