import requests
import json
import os
import logging

class FPLScraper:
    def __init__(self):
        self.base_url = "https://fantasy.premierleague.com/api/"
        self.teams_url = self.base_url + "teams/"
        self.players_url = self.base_url + "bootstrap-static/"

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Define file to save squads
        self.squad_file = os.path.join(os.path.dirname(__file__), "squads.json")

    def fetch_fpl_data(self):
        """Fetch raw FPL data from the official Fantasy Premier League API"""
        try:
            response = requests.get(self.players_url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch FPL data: {str(e)}")
            return None

    def fetch_teams(self):
        """Fetch FPL team data and return a dictionary mapping team IDs to names"""
        fpl_data = self.fetch_fpl_data()
        if not fpl_data:
            return {}

        teams = {team["id"]: team["name"] for team in fpl_data["teams"]}
        return teams

    def fetch_players(self):
        """Fetch all players from FPL and organize them into their correct teams"""
        fpl_data = self.fetch_fpl_data()
        if not fpl_data:
            return {}

        teams = self.fetch_teams()
        squads = {team_name: [] for team_name in teams.values()}  # Create empty lists for each team

        for player in fpl_data["elements"]:
            try:
                team_id = player["team"]  # Get the team ID
                team_name = teams.get(team_id, "Unknown")

                player_info = {
                    "name": f"{player['first_name']} {player['second_name']}",
                    "number": player.get("squad_number", "N/A"),
                    "position": player["element_type"],  # Position as a number (1 = GK, 2 = DEF, etc.)
                    "nationality": player.get("nationality", "Unknown"),
                    "total_points": player.get("total_points", 0),
                }

                # Append to correct team
                if team_name in squads:
                    squads[team_name].append(player_info)
                else:
                    self.logger.warning(f"Skipping player {player_info['name']} - Team ID {team_id} not found")
            except Exception as e:
                self.logger.error(f"Error processing player data: {str(e)}")
                continue

        return squads

    def save_squads(self, squads):
        """Save squads to a JSON file"""
        if not squads:
            self.logger.warning("No squads to save!")
            return

        with open(self.squad_file, "w", encoding="utf-8") as f:
            json.dump(squads, f, indent=4)

        self.logger.info(f"✅ Squads saved to {self.squad_file}")

    def load_squads(self):
        """Load squads from JSON file"""
        if os.path.exists(self.squad_file):
            with open(self.squad_file, "r", encoding="utf-8") as f:
                squads = json.load(f)
            self.logger.info(f"📂 Loaded squads from {self.squad_file}")
            return squads
        return None

if __name__ == "__main__":
    scraper = FPLScraper()

    try:
        # Try loading squads from file
        squads = scraper.load_squads()

        if not squads:  # If no saved squads, fetch and save them
            print("\n🔄 Fetching new squad data...")
            squads = scraper.fetch_players()
            scraper.save_squads(squads)
        else:
            print("\n✅ Using cached squad data.")

        print(f"\nSuccessfully fetched data for {len(squads)} teams")

    except Exception as e:
        print(f"❌ Error: {str(e)}")
