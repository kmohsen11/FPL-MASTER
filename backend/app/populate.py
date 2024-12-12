import os
import sys
import requests
import pandas as pd
from io import StringIO
from sqlalchemy.exc import IntegrityError
from app.models import db, Team, Player, PlayerRoundPerformance
from app import create_app  # Import the application factory function
from app.update_predictions import run_pipeline  # Import the pipeline runner

# Constants
MERGED_GW_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"

def fetch_merged_gw():
    """
    Fetches the merged game week data from the provided URL.
    """
    response = requests.get(MERGED_GW_URL)
    if response.status_code == 200:
        return pd.read_csv(StringIO(response.text))
    else:
        raise Exception(f"Failed to fetch merged_gw.csv. HTTP Status Code: {response.status_code}")

def get_team_name_by_player_name(player_name, merged_gw):
    """
    Get the full team name using the player's full name.
    """
    team_row = merged_gw[merged_gw['name'] == player_name]
    return team_row['team'].values[0] if not team_row.empty else "Unknown"

def clean_database():
    """
    Deletes all entries in the PlayerRoundPerformance, Player, and Team tables.
    """
    db.session.query(PlayerRoundPerformance).delete()
    db.session.query(Player).delete()
    db.session.query(Team).delete()
    db.session.commit()
    print("Database cleaned successfully.")

def load_predictions():
    """
    Loads predictions from the pipeline and combines all data into a single DataFrame.
    """
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
        data = pd.read_csv(filepath)
        data["position"] = position
        return data

    return pd.concat([load_csv(path, pos) for pos, path in files.items()])

def populate_database(data, merged_gw):
    """
    Populates the database with predictions.
    """
    for _, row in data.iterrows():
        team_name = get_team_name_by_player_name(row["decoded_name"], merged_gw)

        team = Team.query.filter_by(name=team_name).first()
        if not team:
            team = Team(name=team_name)
            db.session.add(team)
            try:
                db.session.commit()
            except IntegrityError:
                db.session.rollback()

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


