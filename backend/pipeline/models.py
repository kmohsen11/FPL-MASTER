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
from pipeline.data_prep import DataHandler


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
                position_data['points_after_sqrt'] = np.sign(position_data['points_after']) * np.sqrt(np.abs(position_data['points_after']))
                print(f"{position_name} points_after_sqrt stats:")
                print(position_data['points_after_sqrt'].describe())
            else:
                print(f"'points_after' column missing for {position_name}.")

        self.encoders = self.data_handler.get_encoders()
        
        print("Data preparation completed.")


    def train_multivar_linear_model(self, data, position):
        """Train Multiple Linear Regression Model with Debugging and Feature Validation"""
        
        print(f"Training Multiple Linear Regression for {position.upper()}...\n")
        
        # Ensure target variables exist
        if 'points_after' not in data.columns or 'points_after_sqrt' not in data.columns:
            raise ValueError(f"Missing target columns in {position} data. Columns found: {data.columns}")

        # Drop target variables and keep only numeric features
        X = data.drop(['points_after', 'points_after_sqrt'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
        y = data['points_after_sqrt']

        # Debug: Check feature consistency
        print(f"Feature columns for {position.upper()} model training:\n{X.columns.tolist()}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=None, random_state=42
        )

        # Debug: Check data distribution
        print(f"Train Data Distribution for {position.upper()}:\n", y_train.describe())
        print(f"Test Data Distribution for {position.upper()}:\n", y_test.describe())

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Ensure feature alignment during prediction
        if set(X_train.columns) != set(X_test.columns):
            missing_features = set(X_train.columns) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(X_train.columns)
            print(f"⚠️ Feature mismatch detected in {position.upper()}:\n"
                f"Missing: {missing_features}\nExtra: {extra_features}")

            # Add missing columns with zero values
            for feature in missing_features:
                X_test[feature] = 0

            # Ensure order consistency
            X_test = X_test[X_train.columns]

        # Make predictions
        y_pred_sqrt = model.predict(X_test)
        y_pred = np.sign(y_pred_sqrt) * (y_pred_sqrt ** 2)  # Reverse transformation
        y_test_original = np.sign(y_test) * (y_test ** 2)  # Reverse for evaluation

        # Debug: Check prediction outputs
        print(f"Actual vs. Predicted (Square Root Transformed) for {position.upper()}:\n",
            list(zip(y_test[:5], y_pred_sqrt[:5])))
        print(f"Actual vs. Predicted (Original Scale) for {position.upper()}:\n",
            list(zip(y_test_original[:5], y_pred[:5])))

        # Evaluate the model
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        print(f"✅ {position.upper()} Model Performance:\n"
            f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}\n")

        return model, mae, mse, r2




    
    def train_rf_model(self, data, position):
        """Train Random Forest Model with Debugging and Feature Validation"""
        
        print(f"Training Random Forest Model for {position.upper()}...\n")
        
        # Ensure target variables exist
        if 'points_after' not in data.columns or 'points_after_sqrt' not in data.columns:
            raise ValueError(f"Missing target columns in {position} data. Columns found: {data.columns}")

        # Drop target variables and keep only numeric features
        X = data.drop(['points_after', 'points_after_sqrt'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        y = data['points_after_sqrt']

        # Debug: Check feature consistency
        print(f"Feature columns for {position.upper()} model training:\n{X.columns.tolist()}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=None, random_state=42
        )

        # Debug: Check data distribution
        print(f"Train Data Distribution for {position.upper()}:\n", y_train.describe())
        print(f"Test Data Distribution for {position.upper()}:\n", y_test.describe())

        # Train the model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Ensure feature alignment during prediction
        if set(X_train.columns) != set(X_test.columns):
            missing_features = set(X_train.columns) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(X_train.columns)
            print(f"⚠️ Feature mismatch detected in {position.upper()}:\n"
                f"Missing: {missing_features}\nExtra: {extra_features}")

            # Add missing columns with zero values
            for feature in missing_features:
                X_test[feature] = 0

            # Ensure order consistency
            X_test = X_test[X_train.columns]

        # Make predictions
        y_pred_sqrt = model.predict(X_test)
        y_pred = np.sign(y_pred_sqrt) * (y_pred_sqrt ** 2)  # Reverse transformation
        y_test_original = np.sign(y_test) * (y_test ** 2)  # Reverse for evaluation

        # Debug: Check prediction outputs
        print(f"Actual vs. Predicted (Square Root Transformed) for {position.upper()}:\n",
            list(zip(y_test[:5], y_pred_sqrt[:5])))
        print(f"Actual vs. Predicted (Original Scale) for {position.upper()}:\n",
            list(zip(y_test_original[:5], y_pred[:5])))

        # Evaluate the model
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        print(f"✅ {position.upper()} Model Performance:\n"
            f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}\n")

        return model, mae, mse, r2


    def train_xgb_model(self, data, position):
        """Train XGBoost Model with Debugging and Feature Validation"""
        
        print(f"Training XGBoost Model for {position.upper()}...\n")
        
        # Ensure target variables exist
        if 'points_after' not in data.columns or 'points_after_sqrt' not in data.columns:
            raise ValueError(f"Missing target columns in {position} data. Columns found: {data.columns}")

        # Drop target variables and keep only numeric features
        X = data.drop(['points_after', 'points_after_sqrt'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])
        y = data['points_after_sqrt']

        # Debug: Check feature consistency
        print(f"Feature columns for {position.upper()} model training:\n{X.columns.tolist()}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=None, random_state=42
        )

        # Debug: Check data distribution
        print(f"Train Data Distribution for {position.upper()}:\n", y_train.describe())
        print(f"Test Data Distribution for {position.upper()}:\n", y_test.describe())

        # Train the model
        model = XGBRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Ensure feature alignment during prediction
        if set(X_train.columns) != set(X_test.columns):
            missing_features = set(X_train.columns) - set(X_test.columns)
            extra_features = set(X_test.columns) - set(X_train.columns)
            print(f"⚠️ Feature mismatch detected in {position.upper()}:\n"
                f"Missing: {missing_features}\nExtra: {extra_features}")

            # Add missing columns with zero values
            for feature in missing_features:
                X_test[feature] = 0

            # Ensure order consistency
            X_test = X_test[X_train.columns]

        # Make predictions
        y_pred_sqrt = model.predict(X_test)
        y_pred = np.sign(y_pred_sqrt) * (y_pred_sqrt ** 2)  # Reverse transformation
        y_test_original = np.sign(y_test) * (y_test ** 2)  # Reverse for evaluation

        # Debug: Check prediction outputs
        print(f"Actual vs. Predicted (Square Root Transformed) for {position.upper()}:\n",
            list(zip(y_test[:5], y_pred_sqrt[:5])))
        print(f"Actual vs. Predicted (Original Scale) for {position.upper()}:\n",
            list(zip(y_test_original[:5], y_pred[:5])))

        # Evaluate the model
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        print(f"✅ {position.upper()} Model Performance:\n"
            f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}\n")

        return model, mae, mse, r2

    
    def train_rnn_model(self, data, position):
        """Train RNN (LSTM) Model with Proper Scaling and Debugging"""

        print(f"Training RNN Model for {position.upper()}...\n")

        # Ensure target variables exist
        if 'points_after' not in data.columns or 'points_after_sqrt' not in data.columns:
            raise ValueError(f"Missing target columns in {position} data. Columns found: {data.columns}")

        # Drop target variables and keep only numeric features
        X = data.drop(['points_after', 'points_after_sqrt'], axis=1, errors='ignore')
        X = X.select_dtypes(include=[np.number])  # Keep only numeric columns
        y = data['points_after_sqrt']

        # Debug: Check feature consistency
        print(f"Feature columns for {position.upper()} model training:\n{X.columns.tolist()}")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True, stratify=None, random_state=42
        )

        # Debug: Check data distribution before scaling
        print(f"Train Data Distribution for {position.upper()}:\n", y_train.describe())
        print(f"Test Data Distribution for {position.upper()}:\n", y_test.describe())

        # Apply StandardScaler properly (fit only on training, transform on both)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Save scaler for later use in predictions
        joblib.dump(scaler, f"{self.output_dir}/scaler_{position}.pkl")

        # Dynamically determine number of features
        num_features = X_train_scaled.shape[1]

        # Ensure correct input shape for LSTM (batch_size, timesteps=1, features)
        if num_features < 1:
            raise ValueError(f"Error: No valid numerical features found for {position.upper()} model training!")

        X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, num_features))
        X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, num_features))

        # Debugging print statements to verify shape before training
        print(f"✅ Corrected Input Shape for Training {position.upper()}: {X_train_scaled.shape}")
        print(f"✅ Corrected Input Shape for Prediction {position.upper()}: {X_test_scaled.shape}")

        # Define the RNN (LSTM) model
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(1, num_features)),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])

        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        # Model checkpointing and early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint(f'{self.output_dir}/best_model_{position}.keras', monitor='val_loss', save_best_only=True)

        # Train the model
        model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=100,
            batch_size=32,
            callbacks=[early_stopping, checkpoint],
            verbose=0
        )

        # Try to predict and handle errors
        try:
            y_pred_sqrt = model.predict(X_test_scaled).flatten()
        except Exception as e:
            print(f"⚠️ Prediction Error for {position.upper()}: {e}")
            return model, None, None, None

        # Reverse transformation (square root to original scale)
        y_pred = np.sign(y_pred_sqrt) * (y_pred_sqrt ** 2)
        y_test_original = np.sign(y_test) * (y_test ** 2)

        # Evaluate the model
        mae = mean_absolute_error(y_test_original, y_pred)
        mse = mean_squared_error(y_test_original, y_pred)
        r2 = r2_score(y_test_original, y_pred)

        print(f"✅ {position.upper()} RNN Model Performance:\n"
            f"MAE: {mae:.4f}, MSE: {mse:.4f}, R2 Score: {r2:.4f}\n")

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

    def find_best_model(self, metric='MAE'):
        """Find the best model for each position based on the specified metric (default: MAE)."""
        best_models = {}
        for position, models in self.results.items():
            if metric == 'MAE':
                # Select the model with the lowest MAE
                best_model = min(models.items(), key=lambda x: x[1]['MAE'] if x[1]['MAE'] is not None else float('inf'))
            else:
                # Default to maximizing R2 score (useful for debugging)
                best_model = max(models.items(), key=lambda x: x[1]['R2'] if x[1]['R2'] is not None else float('-inf'))

            best_models[position] = {'Model': best_model[0], 'Value': best_model[1][metric]}
        
        return best_models


    def predict_and_decode(self, position, model_info, file, team_encoder, name_encoder, opponent_encoder, season_encoder):
        """Predict and decode results for a specific position."""
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
        data['next_opponent_team'] = data['next_opponent_team'].astype(str)

        # ✅ Fix: Handle unseen labels in `team_encoder`
        known_classes = set(team_encoder.classes_)  # Get known labels from encoder
        data['next_opponent_team_encoded'] = data['next_opponent_team'].apply(
            lambda x: team_encoder.transform([x])[0] if x in known_classes else -1
        )

        # ✅ Fix: Handle unseen player names in `name_encoder`
        known_names = set(name_encoder.classes_)
        data['name_encoded'] = data['name'].apply(
            lambda x: name_encoder.transform([x])[0] if x in known_names else -1
        )

        # Prepare features
        features = data.drop(columns=['name', 'points_after_log', 'points_after', 'position', 'next_opponent_team'], errors='ignore')

        # ✅ Ensure all categorical and boolean values are converted before float conversion
        features = features.replace({'True': 1, 'False': 0})  
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
                data['next_opponent_team_encoded'] = data['next_opponent_team_encoded'].fillna(-1).astype(int)

                # Decode only known values
                temp_df['decoded_team'] = data['next_opponent_team_encoded'].apply(
                    lambda x: team_encoder.inverse_transform([x])[0] if x in team_encoder.classes_ else "Unknown"
                )

            if 'name_encoded' in data.columns:
                temp_df['decoded_name'] = name_encoder.inverse_transform(data['name_encoded'])

            results.append(temp_df[['decoded_name', 'decoded_team', 'predicted_points_after', 'model']])

        combined_results = pd.concat(results)
        output_file = os.path.join(self.output_dir, f'{position}_predictions.csv')
        combined_results.to_csv(output_file, index=False)

        return output_file
    def align_features(self, features, model, model_name):
        """Align features with those used during model training."""

        print(f"Aligning features for {model_name} model...")

        if model_name == "RNN":
            expected_features = model.input_shape[-1]  # Get expected feature count
            current_features = features.shape[1] if len(features.shape) > 1 else 0

            if current_features != expected_features:
                print(f"⚠️ Feature mismatch detected in RNN model: "
                    f"Expected {expected_features}, but got {current_features}.")

                # Add missing features as dummy columns
                while features.shape[1] < expected_features:
                    features[f"dummy_feature_{features.shape[1]}"] = 0  

                # Drop extra features if too many
                features = features.iloc[:, :expected_features]

                print(f"✅ Adjusted features for RNN: Now {features.shape[1]} features.")

            # ✅ Ensure features are **3D for LSTM** (batch_size, timesteps=1, features)
            aligned_features = features.values.reshape((features.shape[0], 1, features.shape[1]))
            print(f"✅ Final RNN input shape: {aligned_features.shape}")  # Debugging

        else:
            # For non-RNN models (XGBoost, RF, Linear Regression)
            expected_features = model.feature_names_in_ if hasattr(model, "feature_names_in_") else features.columns
            missing_features = set(expected_features) - set(features.columns)
            extra_features = set(features.columns) - set(expected_features)

            # Add missing columns with zeros
            for feature in missing_features:
                features[feature] = 0

            # Drop extra columns
            features = features.reindex(columns=expected_features, fill_value=0)

            print(f"✅ Features aligned successfully for {model_name}.")

            aligned_features = features.values

        return aligned_features




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
                self.encoders['name_encoder'], 
                self.encoders['opp_team_encoder'],  # ✅ Fix: Pass the opponent encoder
                self.encoders['season_stage_encoder']  # ✅ Fix: Pass the season stage encoder
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
            
