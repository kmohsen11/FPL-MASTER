# backend/models.py
from app.db import db

from custom_type import JSONEncodedList

class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False, unique=True)
    logo_url = db.Column(db.String(500), nullable=True)  # URL or path to the team's logo

class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    second_name = db.Column(db.String(50), nullable=False)
    web_name = db.Column(db.String(50), nullable=False)
    team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)  # Foreign key to the Team model
    team = db.relationship('Team', backref='players')
    position = db.Column(db.String(50), nullable=False)  # GK, DEF, MID, FWD


class PlayerRoundPerformance(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    player = db.relationship('Player', backref='round_performances')
    season = db.Column(db.String(20), nullable=False)  # e.g., '2023-24'
    round = db.Column(db.Integer, nullable=False)  # Gameweek or round number
    predicted_points = db.Column(db.Float, nullable=False)  # Model predictions
    opponent_team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)  # Next opponent's team
    opponent_team = db.relationship('Team', foreign_keys=[opponent_team_id])
  
