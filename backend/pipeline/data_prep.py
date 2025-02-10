import pandas as pd
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import os
from io import StringIO

class DataHandler:
   
   
    def __init__(self):
        self.data = {"team_mapping": None, "general": None, "new_data": None, "positions": {"gk": None, "def": None, "mid": None, "fwd": None}}
        self.encoders = {"team_encoder": None, "opp_team_encoder": None, "season_stage_encoder": None, "name_encoder": None}
        self.last_round = {"gk": None, "def": None, "mid": None, "fwd": None}
    def fill_general_data(self):
        #use the csv file to fill the general data
        self.data["general"] = pd.read_csv(
            "pipeline/cleaned_merged_seasons.csv",
            dtype={"season_x": str, "position": str, "element": int},  
            low_memory=False
                )

    def fill_team_mapping(self):
        #use the csv file to fill the team mapping
        self.data["team_mapping"] = pd.read_csv("pipeline/teams.csv")
        
    def fetch_new_data(self):
        """Fetch and validate the latest gameweek data from the 2024-25 season"""
        url = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/2024-25/gws/merged_gw.csv"
        
        response = requests.get(url)
        if response.status_code == 200:
            self.data["new_data"] = pd.read_csv(
                StringIO(response.text), 
                delimiter=',', 
                header=0, 
                usecols=range(42)
            )

            # ✅ Debugging: Ensure new gameweeks are included
            latest_gameweeks = self.data["new_data"]["round"].unique()
            print(f"✅ Latest Gameweeks Fetched: {sorted(latest_gameweeks)}")

        else:
            raise Exception(f"❌ Failed to fetch the file. HTTP Status Code: {response.status_code}")

    def merge_opponent(self):
        # Merge new_data with team_mapping
        merged_gw = self.data['new_data'].merge(
            self.data['team_mapping'], 
            left_on='opponent_team', 
            right_on='id', 
            how='left'
        )

        # Handle overlapping column names
        if 'name_x' in merged_gw.columns and 'name_y' in merged_gw.columns:
            # Retain 'name_x' and drop 'name_y'
            merged_gw = merged_gw.rename(columns={'name_x': 'name'})
            merged_gw = merged_gw.drop(columns=['name_y'])
        elif 'name_x' in merged_gw.columns:
            # Rename 'name_x' if only it exists
            merged_gw = merged_gw.rename(columns={'name_x': 'name'})
        elif 'name_y' in merged_gw.columns:
            # Rename 'name_y' if only it exists
            merged_gw = merged_gw.rename(columns={'name_y': 'name'})

        # Handle other columns like position
        if 'position_x' in merged_gw.columns and 'position_y' in merged_gw.columns:
            merged_gw = merged_gw.rename(columns={'position_x': 'position'})
            merged_gw = merged_gw.drop(columns=['position_y'])
        elif 'position_x' in merged_gw.columns:
            merged_gw = merged_gw.rename(columns={'position_x': 'position'})
        elif 'position_y' in merged_gw.columns:
            merged_gw = merged_gw.rename(columns={'position_y': 'position'})

        # Drop unnecessary columns
        merged_gw = merged_gw.drop(columns=['id'], errors='ignore')

        print("Merged columns after handling overlaps:", merged_gw.columns)
        return merged_gw


    
    
    def merge_columns(self):
        try:
            # ✅ Ensure 'season_x' exists in new_data before merging
            self.data["new_data"]["season_x"] = "2024-25"

            # Remove duplicate columns in new_data and merged_gw
            self.data["new_data"] = self.data["new_data"].loc[:, ~self.data["new_data"].columns.duplicated()]
            merged_gw = self.merge_opponent()
            merged_gw = merged_gw.loc[:, ~merged_gw.columns.duplicated()]

            print("✅ new_data columns:", self.data["new_data"].columns)
            print("✅ merged_gw columns after deduplication:", merged_gw.columns)

            # Ensure `position` exists in both datasets
            if "position" not in self.data["new_data"].columns:
                raise Exception("❌ The 'position' column is missing in `new_data`.")
            if "position" not in merged_gw.columns:
                raise Exception("❌ The 'position' column is missing in `merged_gw`.")

            # Ensure both datasets have the same columns
            missing_columns = set(self.data["new_data"].columns) - set(merged_gw.columns)
            if missing_columns:
                raise Exception(f"❌ Missing columns in merged data: {missing_columns}")

            merged_gw = merged_gw[self.data["new_data"].columns]

            # Reset indexes to avoid conflicts
            self.data["new_data"] = self.data["new_data"].reset_index(drop=True)
            merged_gw = merged_gw.reset_index(drop=True)

            # Combine new_data and merged_gw
            combined_data = pd.concat([self.data["new_data"], merged_gw], ignore_index=True)

            # ✅ Ensure latest round for each player is kept
            combined_data = combined_data.sort_values(by=['element', 'season_x', 'round'], ascending=[True, True, False])
            combined_data = combined_data.drop_duplicates(subset=['element', 'season_x', 'round'], keep='first')

            # ✅ Rename `total_points` to `points`
            combined_data.rename(columns={"total_points": "points"}, inplace=True)
            self.data["general"].rename(columns={"total_points": "points"}, inplace=True)

            # ✅ Align columns between combined_data and general
            all_columns = list(set(combined_data.columns).union(set(self.data["general"].columns)))
            combined_data = combined_data.reindex(columns=all_columns)
            self.data["general"] = self.data["general"].reindex(columns=all_columns)

            # ✅ Concatenate general and combined_data into unified_data
            unified_data = pd.concat([self.data["general"], combined_data], ignore_index=True)

            # ✅ Final Debugging: Print latest rounds in data
            latest_gameweeks = unified_data.groupby("season_x")["round"].max().to_dict()
            print(f"✅ Latest gameweek per season: {latest_gameweeks}")

            print(f"✅ Data merged successfully with {len(unified_data)} records and columns:", unified_data.columns)
            return unified_data

        except Exception as e:
            print(f"❌ Error during column merging: {e}")
            raise


    
    def process_unified_data(self):
        
        unified_data = self.merge_columns()
        unified_data.rename(columns={"total_points": "points"}, inplace=True)
        print("After merging columns:", unified_data.columns)
        unified_data['rolling_bps'] = unified_data.groupby('element')['bps'].transform(lambda x: x.rolling(3, min_periods=1).mean())
        unified_data['rolling_clean_sheets'] = unified_data.groupby('element')['clean_sheets'].transform(lambda x: x.rolling(3, min_periods=1).mean())


        unified_data['cumulative_points'] = unified_data.groupby('element')['points'].cumsum()
        unified_data['cumulative_minutes'] = unified_data.groupby('element')['minutes'].cumsum()
        unified_data = unified_data.sort_values(by=['season_x', 'element', 'GW'])
        # Ensure DataFrame is sorted by player and GW
        unified_data = unified_data.sort_values(by=['element', 'season_x', 'GW']).reset_index(drop=True)

        # Step 1: Compute the next opponent team
        unified_data['next_opponent_team'] = unified_data.groupby(['season_x', 'element'])['opponent_team'].shift(-1)

        # Ensure 'next_opponent_team' is treated as a string
        unified_data['next_opponent_team'] = unified_data['next_opponent_team'].fillna('Unknown').astype(str)
        unified_data['opponent_team'] = unified_data['opponent_team'].fillna('Unknown').astype(str)
        unified_data['team_x'] = unified_data['team_x'].fillna('Unknown').astype(str)

        # Calculate cumulative stats for each team in each season
        unified_data['cumulative_goals_conceded'] = unified_data.groupby(['team_x', 'season_x'])['goals_conceded'].cumsum()
        unified_data['match_count'] = unified_data.groupby(['team_x', 'season_x']).cumcount() + 1
        unified_data['avg_goals_conceded'] = unified_data['cumulative_goals_conceded'] / unified_data['match_count']

        # Shift to avoid using the current game in the average
        unified_data['prev_avg_goals_conceded'] = unified_data.groupby(['team_x', 'season_x'])['avg_goals_conceded'].shift(1)

        # Map the next opponent's average goals conceded
        opponent_avg_goals_lookup = unified_data[['season_x', 'team_x', 'GW', 'prev_avg_goals_conceded']].rename(
            columns={'team_x': 'lookup_team'}
        )

        unified_data = unified_data.merge(
            opponent_avg_goals_lookup,
            left_on=['season_x', 'next_opponent_team', 'GW'],
            right_on=['season_x', 'lookup_team', 'GW'],
            how='left',
            suffixes=('', '_next_opponent')
        )

        unified_data['next_opponent_avg_goals_conceded'] = unified_data['prev_avg_goals_conceded_next_opponent']
        unified_data.drop(columns=['prev_avg_goals_conceded_next_opponent', 'lookup_team'], inplace=True)
        #drop next_opponent_avg_goals_conceded
        unified_data = unified_data.drop(columns=['next_opponent_avg_goals_conceded'])
        # Calculate average points or goals per team
        team_avg_points = unified_data.groupby('team_x')['points'].mean()
        team_avg_goals = unified_data.groupby('team_x')['goals_scored'].mean()

        # Map the averages to teams
        unified_data['team_x_strength'] = unified_data['team_x'].map(team_avg_points)
        unified_data['opponent_team_strength'] = unified_data['opponent_team'].map(team_avg_points)

        # Calculate the team strength difference
        unified_data['team_strength_diff'] = unified_data['team_x_strength'] - unified_data['opponent_team_strength']
        unified_data = unified_data.drop(columns=['opponent_team_strength'])
        #remove columns with missing values
        unified_data = unified_data.dropna(axis=1)
        
        #drop specific columns
        # Drop specified columns
        columns_to_drop = [
            "expected_goals_conceded", "team", "starts", "expected_goals", "prev_avg_goals_conceded", "xP"
        ]
        unified_data = unified_data.drop(columns=columns_to_drop, errors='ignore')

        # Rename `team_x` to `team`
        unified_data.rename(columns={"team_x": "team"}, inplace=True)

        unified_data['team_avg_points'] = unified_data.groupby('team')['points'].transform('mean')
        unified_data['team_total_goals'] = unified_data.groupby('team')['goals_scored'].transform('sum')
        unified_data['season_stage'] = pd.cut(unified_data['GW'], bins=[0, 10, 25, 38], labels=['Early', 'Mid', 'Late'])
        
        df = unified_data.copy()
        df = pd.DataFrame(df)
        
        # Step 1: Filter data from the 2018-2019 season onwards
        df = df[df['season_x'].isin(['2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2023-24', '2024-25'])]
        df.rename(columns={"total_points": "points"})
        df['was_home'] = df['was_home'].map({True: 1, False: 0})

        df["points"] = df["points"].apply(lambda x: max(x, 0))
        df['points_after'] = df.groupby('element')['points'].shift(-1)
        df['points_after'] = df['points_after'].fillna(0)
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
        df['start_hour'] = df['kickoff_time'].dt.hour
        #drop kick off time
        df = df.drop(columns=['kickoff_time'])
        #split into four sets for differnt positions
        df_gk = df[df['position'] == 'GK']
        df_def = df[df['position'] == 'DEF']
        df_mid = df[df['position'] == 'MID']
        df_fwd = df[df['position'] == 'FWD']
        gk_vars = ['transfers_balance', 'clean_sheets', 'points', 'bps', 'season_x',
       'transfers_in', 'name', 'fixture', 'saves', 'transfers_out', 'round',
       'value', 'position', 'own_goals', 'red_cards', 'goals_scored',
       'element', 'creativity', 'ict_index', 'was_home', 'team_h_score',
       'penalties_saved', 'opponent_team', 'yellow_cards', 'team', 'assists',
       'goals_conceded', 'threat', 'GW', 'influence', 'team_a_score',
       'minutes', 'penalties_missed', 'bonus', 'selected', 'rolling_bps',
       'rolling_clean_sheets', 'cumulative_points', 'cumulative_minutes',
       'next_opponent_team', 'cumulative_goals_conceded', 'match_count',
       'avg_goals_conceded', 'team_x_strength', 'team_avg_points',
       'team_total_goals', 'season_stage', 'points_after', 'start_hour']

        # Apply similar changes to other variable lists
        def_vars = gk_vars
        mid_vars = gk_vars
        fwd_vars = gk_vars
        
        #apply the vars to the dataframes, for start hour, get it from the kick off time first before applying it, ignore team_x_encoded as it is not in the dataframes
        df_gk = df_gk[gk_vars]
        df_def = df_def[def_vars]
        df_mid = df_mid[mid_vars]
        df_fwd = df_fwd[fwd_vars]

        #store in self.data
        self.data["positions"]["gk"] = df_gk
        self.data["positions"]["def"] = df_def
        self.data["positions"]["mid"] = df_mid
        self.data["positions"]["fwd"] = df_fwd
        
    
    def encode_positions(self):
        
        team_encoder = LabelEncoder()
        opp_team_encoder = LabelEncoder()
        season_stage_encoder = LabelEncoder()
        
        # fit the encoders during data prep
        self.data["positions"]["gk"]["team"] = team_encoder.fit_transform(self.data["positions"]["gk"]["team"])
        self.data["positions"]["def"]["team"] = team_encoder.transform(self.data["positions"]["def"]["team"])
        self.data["positions"]["mid"]["team"] = team_encoder.transform(self.data["positions"]["mid"]["team"])
        self.data["positions"]["fwd"]["team"] = team_encoder.transform(self.data["positions"]["fwd"]["team"])
        
        self.data["positions"]["gk"]["opponent_team"] = opp_team_encoder.fit_transform(self.data["positions"]["gk"]["opponent_team"])
        self.data["positions"]["def"]["opponent_team"] = opp_team_encoder.transform(self.data["positions"]["def"]["opponent_team"])
        self.data["positions"]["mid"]["opponent_team"] = opp_team_encoder.transform(self.data["positions"]["mid"]["opponent_team"])
        self.data["positions"]["fwd"]["opponent_team"] = opp_team_encoder.transform(self.data["positions"]["fwd"]["opponent_team"])
        
        
        self.data["positions"]["gk"]["season_stage"] = season_stage_encoder.fit_transform(self.data["positions"]["gk"]["season_stage"])       
        self.data["positions"]["def"]["season_stage"] = season_stage_encoder.transform(self.data["positions"]["def"]["season_stage"])
        self.data["positions"]["mid"]["season_stage"] = season_stage_encoder.transform(self.data["positions"]["mid"]["season_stage"])
        self.data["positions"]["fwd"]["season_stage"] = season_stage_encoder.transform(self.data["positions"]["fwd"]["season_stage"])
        
        
        #store the encoders
        self.encoders["team_encoder"] = team_encoder
        self.encoders["opp_team_encoder"] = opp_team_encoder
        self.encoders["season_stage_encoder"] = season_stage_encoder
        
    def get_last_round(self):
        def get_last_round_data(df):
            # Sort by player (element) and round (descending)
            df = df.sort_values(by=['element', 'round'], ascending=[True, False])
            # Keep only the latest gameweek per player
            last_round_data = df.groupby('element').head(1)
            return last_round_data

        # Ensure `season_x` exists and is correctly formatted
        for position in ["gk", "def", "mid", "fwd"]:
            if "season_x" not in self.data["positions"][position].columns:
                raise ValueError(f"Missing 'season_x' in {position} data.")
            self.data["positions"][position]["season_x"] = self.data["positions"][position]["season_x"].astype(str)

        # Filter data for the 2024-25 season
        for position in ["gk", "def", "mid", "fwd"]:
            position_data = self.data["positions"][position]
            filtered_data = position_data[position_data["season_x"] == "2024-25"]
            self.last_round[position] = get_last_round_data(filtered_data)

        print("✅ Successfully extracted last round data for all positions.")


        
    def extra_pos_process(self):
        
        self.data["positions"]["gk"].drop(columns=["season_x"], inplace=True)
        self.data["positions"]["def"].drop(columns=["season_x"], inplace=True)
        self.data["positions"]["mid"].drop(columns=["season_x"], inplace=True)
        self.data["positions"]["fwd"].drop(columns=["season_x"], inplace=True)
        
    
    def encode_name(self):
        name_encoder = LabelEncoder()
        #combine all unique names across all positions
        
        all_names = pd.concat([
            self.data["positions"]["gk"]["name"],
            self.data["positions"]["def"]["name"],
            self.data["positions"]["mid"]["name"],
            self.data["positions"]["fwd"]["name"]
        ])
        name_encoder.fit(all_names)
        
        self.data["positions"]["gk"]["name"] = name_encoder.transform(self.data["positions"]["gk"]["name"])
        self.data["positions"]["def"]["name"] = name_encoder.transform(self.data["positions"]["def"]["name"])
        self.data["positions"]["mid"]["name"] = name_encoder.transform(self.data["positions"]["mid"]["name"])
        self.data["positions"]["fwd"]["name"] = name_encoder.transform(self.data["positions"]["fwd"]["name"])
        
        self.encoders["name_encoder"] = name_encoder
        
    def get_gk_last_round(self):
        return self.last_round["gk"]
    
    def get_def_last_round(self):
        
        return self.last_round["def"]
    
    def get_mid_last_round(self):
        
        return self.last_round["mid"]
    
    def get_fwd_last_round(self):
        
        return self.last_round["fwd"]
    
    def get_gk_data(self):
            
            return self.data["positions"]["gk"]
    
    def get_def_data(self):
            
            return self.data["positions"]["def"]

    def get_mid_data(self):

            return self.data["positions"]["mid"]
        
    def get_fwd_data(self):
                
                return self.data["positions"]["fwd"]
            
    def get_encoders(self):
            
            return self.encoders
 