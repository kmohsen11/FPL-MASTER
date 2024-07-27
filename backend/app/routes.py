from flask import jsonify, request, current_app as app
from .models import Player
from . import db



    

@app.route('/home', methods=['GET'])
def hello():
    return jsonify(message="Home page!"), 200







@app.route('/players', methods=['GET'])
def get_players():
    players = Player.query.all()
    return jsonify([{
        'id': player.id,
        'name': player.name,
        'points': player.pts,
        'expected_points': player.expected,
        'price': player.price,
        'team': player.team
    } for player in players]), 200

""""
@app.route("/fixtures",methods = ["GET"] )
def show_fixtures():
    fixtures = Fixtures.query.all()
    return jsonify([{
        'id': fixture.id,
        'kickoff_time': fixture.kickoff_time,
        'away_team': fixture.away_team,
        'away_team_score': fixture.away_team_score,
        'home_team': fixture.home_team,
        'home_team_score': fixture.home_team_score
    } for fixture in fixtures]), 200
"""    
@app.route("/player", methods=["GET"])
def get_player():
    player_id = request.args.get('id')
    if not player_id:
        return jsonify({'error': 'No player ID provided'}), 400

    player = Player.query.get(player_id)
    if not player:
        return jsonify({'error': 'Player not found'}), 404

    return jsonify({
        'id': player.id,
        'name': player.name,
        'points': player.pts,
        'expected_points': player.expected,
        'price': player.price,
        'team': player.team,
        'all_stats': player.all_stats
    }), 200