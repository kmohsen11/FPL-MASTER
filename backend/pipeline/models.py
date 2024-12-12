import os
import time
import numpy as np
import pandas as pd 
import json
import tensorflow as tf
import pickle

import joblib
# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Deep Learning Libraries
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import save_model


#data_prep.py 
from data_prep import DataHandler

class FantasyFootballPredictor:
    def __init__(self, output_dir='predictions'):
        """
        Initialize the predictor with data handling and configuration
        
        Args:
            output_dir (str): Directory to save model and prediction outputs
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.data_handler = DataHandler()
        self.results = {
            pos: {
                model: {"MAE": 0, "MSE": 0, "R2": 0, "Model": None}
                for model in ['multiple linear regression', 'random forest', 'xgboost', 'rnn']
            }
            for pos in ['gk', 'def', 'mid', 'fwd']
        }
        
        # Placeholders for data and encoders
        self.gk_data = None
        self.def_data = None
        self.mid_data = None
        self.fwd_data = None
        self.encoders = None

    def prepare_data(self):
        """Comprehensive data preparation method."""
        print("Preparing data...")
        
        # Load and process data
        self.data_handler.fill_general_data()
        self.data_handler.fetch_new_data()
        self.data_handler.fill_team_mapping()
        self.data_handler.merge_opponent()
        self.data_handler.process_unified_data()
        self.data_handler.encode_positions()
        self.data_handler.get_last_round()
        self.data_handler.extra_pos_process()
        self.data_handler.encode_name()
        # Get data for different positions
        self.gk_data = self.data_handler.get_gk_data()
        self.def_data = self.data_handler.get_def_data()
        self.mid_data = self.data_handler.get_mid_data()
        self.fwd_data = self.data_handler.get_fwd_data()

        # Add position column
        self.gk_data["position"] = 1
        self.def_data["position"] = 2
        self.mid_data["position"] = 3
        self.fwd_data["position"] = 4

        # Preprocessing: Drop NaN values and encode categorical columns
        for df in [self.gk_data, self.def_data, self.mid_data, self.fwd_data]:
            df.dropna(inplace=True)
            
            # Encode 'next_opponent_team'
            df['next_opponent_team'] = df['next_opponent_team'].astype(str)
            label_encoder = LabelEncoder()
            df['next_opponent_team_encoded'] = label_encoder.fit_transform(df['next_opponent_team'])
            df.drop(columns=['next_opponent_team'], inplace=True)
        
        # Log-transform 'points_after' for all positions
        for position_data, position_name in [
            (self.gk_data, "GK"),
            (self.def_data, "DEF"),
            (self.mid_data, "MID"),
            (self.fwd_data, "FWD")
        ]:
            if 'points_after' in position_data.columns:
                position_data['points_after_log'] = np.log1p(position_data['points_after'])
                print(f"{position_name} points_after_log stats:")
                print(position_data['points_after_log'].describe())
            else:
                print(f"'points_after' column missing for {position_name}.")

        self.encoders = self.data_handler.get_encoders()
        
        print("Data preparation completed.")


    def train_multivar_linear_model(self, data, position):
        """Train Multiple Linear Regression Model"""
        # Drop target variables and ensure all features are numeric
        X = data.drop(['points_after', 'points_after_log'], axis=1)
        X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
        y = data['points_after_log']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions and evaluate
        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)

        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        return model, mae, mse, r2


    def train_rf_model(self, data, position):
        """Train Random Forest Model"""
        X = data.drop(['points_after', 'points_after_log'], axis=1)
        y = data['points_after_log']
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)

        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        return model, mae, mse, r2

    def train_xgb_model(self, data, position):
        """Train XGBoost Model"""
        X = data.drop(['points_after', 'points_after_log'], axis=1)
        y = data['points_after_log']
        X = X.select_dtypes(include=[np.number])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)
        y_test_original = np.expm1(y_test)

        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        return model, mae, mse, r2

    def train_rnn_model(self, data, position):
        """Train RNN (LSTM) Model"""
        X = data.drop(['points_after', 'points_after_log'], axis=1)
        y = data['points_after_log']
        X = X.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
        X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'{self.output_dir}/best_model_{position}.keras', monitor='val_loss', save_best_only=True)

        model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log.flatten())
        y_test_original = np.expm1(y_test)

        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        return model, mae, mse, r2

    def train_models(self):
        """Train models for each position"""
        positions_data = [
            ('gk', self.gk_data),
            ('def', self.def_data),
            ('mid', self.mid_data),
            ('fwd', self.fwd_data)
        ]

        model_save_paths = {
            'random forest': "{output_dir}/best_rf_model_{position}.pkl",
            'xgboost': "{output_dir}/best_xgb_model_{position}.json",
            'rnn': "{output_dir}/best_rnn_model_{position}.keras",
            'multiple linear regression': "{output_dir}/best_lr_model_{position}.pkl"
        }

        for position, data in positions_data:
            print(f"Training models for {position.upper()} position...")

            model_functions = [
                ('multiple linear regression', self.train_multivar_linear_model),
                ('random forest', self.train_rf_model),
                ('xgboost', self.train_xgb_model),
                ('rnn', self.train_rnn_model)
            ]

            for model_name, train_func in model_functions:
                print(f"Training {model_name} for {position.upper()}...")

                # Train the model and collect metrics
                model, mae, mse, r2 = train_func(data, position)

                # Update results dictionary
                self.results[position][model_name]["MAE"] = mae
                self.results[position][model_name]["MSE"] = mse
                self.results[position][model_name]["R2"] = r2
                self.results[position][model_name]["Model"] = model

                # Save the model in the appropriate format
                save_path = model_save_paths[model_name].format(output_dir=self.output_dir, position=position.upper())

                if model_name == 'random forest' or model_name == 'multiple linear regression':
                    with open(save_path, 'wb') as f:
                        pickle.dump(model, f)

                elif model_name == 'xgboost':
                    model.save_model(save_path)

                elif model_name == 'rnn':
                    model.save(save_path)

                print(f"Saved {model_name} for {position.upper()} at {save_path}")

        print("\nAll models have been trained and saved successfully!")

    def find_best_model(self, metric='R2'):
        """Find the best model for each position based on a metric"""
        best_models = {}
        for position, models in self.results.items():
            best_model = max(models.items(), key=lambda x: x[1][metric] if metric == 'R2' else -x[1][metric])
            best_models[position] = {'Model': best_model[0], 'Value': best_model[1][metric]}
        return best_models

    def predict_and_decode(self, position, model_info, file, team_encoder, name_encoder, opponent_encoder, season_encoder):
        """Predict and decode results for a specific position"""
        data = file.copy()
        
        # Drop unused columns
        unused_columns = ['season_x']
        for col in unused_columns:
            if col in data.columns:
                data.drop(columns=[col], inplace=True)

        # Add missing columns
        if 'name' not in data.columns:
            data['name'] = "Unknown"
        if 'position' not in data.columns:
            data['position'] = position

        # Encode categorical features
        data['next_opponent_team_encoded'] = team_encoder.transform(data['next_opponent_team'].astype(str))
        data['name_encoded'] = name_encoder.transform(data['name'].astype(str))

        # Prepare features
        features = data.drop(columns=['name', 'points_after_log', 'points_after', 'position', 'next_opponent_team'], errors='ignore')
        features = features.fillna(0).astype(float)

        results = []
        for model_name, model in model_info:
            if model_name == "RNN":
                aligned_features = np.expand_dims(self.align_features(features, model, model_name).values, axis=1)
            else:
                aligned_features = self.align_features(features, model, model_name)

            # Predict using the model
            if model_name == "RNN":
                predictions = model.predict(aligned_features).flatten()
            else:
                predictions = model.predict(aligned_features).flatten()
            
            predictions = np.expm1(predictions)
            predictions = np.maximum(predictions, 0)

            temp_df = data.copy()
            temp_df['predicted_points_after'] = predictions
            temp_df['model'] = model_name
            
            if 'next_opponent_team_encoded' in data.columns:
                temp_df['decoded_team'] = team_encoder.inverse_transform(data['next_opponent_team_encoded'])
            if 'name_encoded' in data.columns:
                temp_df['decoded_name'] = name_encoder.inverse_transform(data['name_encoded'])

            results.append(temp_df[['decoded_name', 'decoded_team', 'predicted_points_after', 'model']])

        combined_results = pd.concat(results)
        output_file = os.path.join(self.output_dir, f'{position}_predictions.csv')
        combined_results.to_csv(output_file, index=False)
        
        return output_file
    # Add the helper align_features method to the class
    def align_features(self, features, model, model_name):
        """Align features with those used during model training."""
        if model_name == "RNN":
            expected_features = model.input_shape[-1]
            current_features = features.shape[1]

            if current_features != expected_features:
                

                # Add or drop features as necessary
                if current_features < expected_features:
                    for i in range(expected_features - current_features):
                        features[f"dummy_feature_{i}"] = 0  # Add dummy columns
                else:
                    features = features.iloc[:, :expected_features]  # Drop extra columns

        else:
            expected_features = model.feature_names_in_
            missing_features = set(expected_features) - set(features.columns)
            for feature in missing_features:
                features[feature] = 0  # Add missing features
            features = features[expected_features]  # Align the order

        return features
    def predict_all(self):
        """Predict for all positions"""
        print("Generating predictions...")
        
        # Get last round data
        last_round_data = {
            'gk': self.data_handler.get_gk_last_round(),
            'def': self.data_handler.get_def_last_round(),
            'mid': self.data_handler.get_mid_last_round(),
            'fwd': self.data_handler.get_fwd_last_round()
        }

        # Find best models
        best_models = self.find_best_model('R2')
        
        prediction_files = {}
        for position, data in last_round_data.items():
            best_model_name = best_models[position]['Model']
            best_model = self.results[position][best_model_name]['Model']
            
            output_file = self.predict_and_decode(
                position, 
                [(best_model_name, best_model)], 
                data, 
                self.encoders['team_encoder'], 
                self.encoders['name_encoder']
            )
            
            prediction_files[position] = output_file
        
        return prediction_files
    
class PredictionExporter:
    def __init__(self, prediction_files, output_dir='predictions'):
        """
        Prepare predictions for backend consumption
        
        Args:
            prediction_files (dict): Dictionary of prediction file paths for each position
            output_dir (str): Directory to save backend-ready prediction files
        """
        self.prediction_files = prediction_files
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_backend_predictions(self):
        """
        Process prediction files and create backend-friendly formats
        
        Returns:
            dict: Processed prediction data ready for backend use
        """
        backend_predictions = {}

        for position, file_path in self.prediction_files.items():
            # Read the prediction CSV
            df = pd.read_csv(file_path)
            
            # Group by player name and team, aggregate predictions
            grouped_predictions = df.groupby(['decoded_name', 'decoded_team']).agg({
                'predicted_points_after': ['mean', 'std', 'min', 'max']
            }).reset_index()
            
            # Flatten the multi-level column names
            grouped_predictions.columns = [
                'player_name', 'team', 
                'predicted_points_mean', 
                'predicted_points_std', 
                'predicted_points_min', 
                'predicted_points_max'
            ]
            
            # Add position information
            grouped_predictions['position'] = position.upper()
            
            # Convert to dictionary for easy JSON serialization
            backend_predictions[position] = grouped_predictions
            
