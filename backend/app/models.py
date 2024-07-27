# backend/models.py
from . import db
from custom_type import JSONEncodedList

#class data of each player in general 
class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(50), nullable=False)
    second_name = db.Column(db.String(50), nullable=False)
    web_name = db.Column(db.String(50), nullable=False)
    team = db.Column(db.String(50), nullable=False)
    player_type = db.Column(db.String(50), nullable=False) # position
    total_points = db.Column(db.Integer, default=0)
    in_dreamteam = db.Column(db.Boolean, default=False)
    dreamteam_count = db.Column(db.Integer, default=0) # how many times was in team of the week
    price = db.Column(db.Float, nullable=False) 
    goals_scored = db.Column(db.Integer, default=0) # total
    assists = db.Column(db.Integer, default=0) # total
    clean_sheets = db.Column(db.Integer, default=0) 
    goals_conceded = db.Column(db.Integer, default=0) # of the whole team
    yellow_cards = db.Column(db.Integer, default=0) 
    red_cards = db.Column(db.Integer, default=0)
    transfers_in = db.Column(db.Integer, default=0) # from EPL fantasy website
    transfers_out = db.Column(db.Integer, default=0) # from EPL fantasy website

# EPL teams
class Team(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(200), nullable=False)
    region_id = db.Column(db.Integer, db.ForeignKey('region.id'), nullable=False)
    region = db.relationship('Region', back_populates='teams') # their city


class Region(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    country = db.Column(db.String(100), nullable=False)
    teams = db.relationship('Team', back_populates='region')

#general data of a given game
class Match(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.Date, nullable=False)
    round = db.Column(db.Integer, nullable=False)
    home_team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    away_team_id = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    home_score = db.Column(db.Integer)
    away_score = db.Column(db.Integer)
    home_team = db.relationship('Team', foreign_keys=[home_team_id])
    away_team = db.relationship('Team', foreign_keys=[away_team_id])

# data of a player in a given match
class PlayerMatchPoint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    match_id = db.Column(db.Integer, db.ForeignKey('match.id'), nullable=False)
    position_id = db.Column(db.Integer, db.ForeignKey('position.id'), nullable=False)
    goals_scored = db.Column(db.Integer, default=0)
    assists = db.Column(db.Integer, default=0)
    passes_completed = db.Column(db.Integer, default=0)
    tackles = db.Column(db.Integer, default=0)
    player = db.relationship('Player')
    match = db.relationship('Match')
    position = db.relationship('Position')

# state of a game
class Fixture(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    event = db.Column(db.Integer, nullable=False)
    finished = db.Column(db.Boolean, default=False)
    minutes = db.Column(db.Integer, nullable=False)
    kickoff_time = db.Column(db.String(50), nullable=False)
    away_team = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    away_team_score = db.Column(db.Integer)
    home_team = db.Column(db.Integer, db.ForeignKey('team.id'), nullable=False)
    home_team_score = db.Column(db.Integer)
    away_team_rel = db.relationship('Team', foreign_keys=[away_team])
    home_team_rel = db.relationship('Team', foreign_keys=[home_team])


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(50), nullable=False, unique=True)
    password = db.Column(db.String(200), nullable=False)
    teams = db.relationship('UserTeam', back_populates='user')

# the current team of a given user
class UserTeam(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    team_name = db.Column(db.String(200), nullable=False)
    players = db.relationship('UserTeamPlayer', back_populates='user_team')
    user = db.relationship('User', back_populates='teams')

# the players in the team of a given user individually
class UserTeamPlayer(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_team_id = db.Column(db.Integer, db.ForeignKey('user_team.id'), nullable=False)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    state = db.Column(db.String(10), nullable=False)  # 'main' or 'sub'
    user_team = db.relationship('UserTeam', back_populates='players')
    player = db.relationship('Player')

    __table_args__ = (
        db.CheckConstraint(
            state.in_(['main', 'sub']),
            name='state_check'
        ),
    )

# the statistics of a player in a given season
class PlayerStatistics(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    player_id = db.Column(db.Integer, db.ForeignKey('player.id'), nullable=False)
    season = db.Column(db.String(50), nullable=False)
    total_points = db.Column(db.Integer, default=0)
    goals = db.Column(db.Integer, default=0)
    assists = db.Column(db.Integer, default=0)
    clean_sheets = db.Column(db.Integer, default=0)
    goals_conceded = db.Column(db.Integer, default=0)
    yellow_cards = db.Column(db.Integer, default=0)
    red_cards = db.Column(db.Integer, default=0)
    player = db.relationship('Player')


class Position(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text, nullable=False)
