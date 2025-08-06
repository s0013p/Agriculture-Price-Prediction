

import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

class SarimaXGBoostEnsembleModel:
    def __init__(self):
        self.sarima_model = None
        self.sarima_fit = None  # Store the fitted model here
        self.xgb_model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            objective='reg:squarederror',
            eval_metric='rmse'
        )
        self.scaler = StandardScaler()
        self.trained = False
        self.feature_columns = ['Min_Price', 'Max_Price']
        self.target_column = 'Modal_Price'
        self.date_column = 'Arrival_Date'
        self.features_importance = {}
        self.ensemble_weights = {'sarima': 0.6, 'xgboost': 0.4}  # Default weights
        
    def preprocess_data(self, df):
        """Preprocess the data for modeling."""
        try:
            # Create a copy of dataframe
            data = df.copy()
            
            # Ensure date column is datetime
            data[self.date_column] = pd.to_datetime(data[self.date_column])
            
            # Sort by date
            data = data.sort_values(by=self.date_column)
            
            # Convert price columns to numeric
            for col in [self.target_column] + self.feature_columns:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Drop missing values
            data = data.dropna(subset=[self.target_column])
            
            # Create additional features
            data['price_ratio'] = data['Max_Price'] / data['Min_Price']
            data['price_diff'] = data['Max_Price'] - data['Min_Price']
            
            # Include weather features if available
            weather_features = ['temperature', 'visibility', 'wind_speed', 'clouds']
            for feature in weather_features:
                if feature in data.columns:
                    data[feature] = pd.to_numeric(data[feature], errors='coerce')
                    self.feature_columns.append(feature)
            
            # Include CPI if available
            if 'CPI' in data.columns:
                data['CPI'] = pd.to_numeric(data['CPI'], errors='coerce')
                self.feature_columns.append('CPI')
            
            # Fill missing values with forward and backward fill
            for col in self.feature_columns:
                if col in data.columns:
                    data[col] = data[col].ffill().bfill()
            
            # Add time-based features
            data['month'] = data[self.date_column].dt.month
            data['day_of_week'] = data[self.date_column].dt.dayofweek
            data['day_of_year'] = data[self.date_column].dt.dayofyear
            
            # Add lag features
            for lag in [1, 3, 7]:
                data[f'price_lag_{lag}'] = data[self.target_column].shift(lag)
                
            # Add rolling statistics
            for window in [3, 7, 14]:
                data[f'price_rolling_mean_{window}'] = data[self.target_column].rolling(window=window).mean()
                data[f'price_rolling_std_{window}'] = data[self.target_column].rolling(window=window).std()
            
            # Drop rows with NaN values created by lag features
            data = data.dropna()
            
            return data
            
        except Exception as e:
            print(f"Error in preprocessing: {str(e)}")
            return None
    
    def train(self, df):
        """Train the ensemble model."""
        try:
            # Preprocess data
            data = self.preprocess_data(df)
            if data is None or len(data) < 30:  # Need sufficient data for training
                print("Insufficient data for training")
                return False
            
            # Set aside most recent data for validation
            train_size = int(len(data) * 0.8)
            train_data = data.iloc[:train_size]
            val_data = data.iloc[train_size:]
            
            # Train SARIMA model
            # Use a basic seasonal model for weekly patterns
            self.sarima_model = SARIMAX(
                train_data[self.target_column],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            # Store the fitted model
            self.sarima_fit = self.sarima_model.fit(disp=False)
            
            # Prepare data for XGBoost
            X_train = train_data[self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                         'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                         'price_rolling_mean_7', 'price_rolling_std_7']]
            y_train = train_data[self.target_column]
            
            X_val = val_data[self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                     'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                     'price_rolling_mean_7', 'price_rolling_std_7']]
            y_val = val_data[self.target_column]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train XGBoost model
            eval_set = [(X_train_scaled, y_train), (X_val_scaled, y_val)]
            self.xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
            
            # Get feature importance
            feature_names = X_train.columns
            self.features_importance = dict(zip(feature_names, self.xgb_model.feature_importances_))
            
            # Optimize ensemble weights using validation data
            self._optimize_weights(val_data)
            
            self.trained = True
            return True
            
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return False
    
    def _optimize_weights(self, val_data):
        """Optimize the weights for SARIMA and XGBoost models."""
        try:
            # Get validation features
            X_val = val_data[self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                   'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                   'price_rolling_mean_7', 'price_rolling_std_7']]
            X_val_scaled = self.scaler.transform(X_val)
            
            # Get predictions from both models
            # FIXED: Use the fitted model's forecast method
            sarima_pred = self.sarima_fit.forecast(steps=len(val_data))
            xgb_pred = self.xgb_model.predict(X_val_scaled)
            
            # Try different weight combinations to minimize MAPE
            best_mape = float('inf')
            best_weights = {'sarima': 0.5, 'xgboost': 0.5}
            
            for sarima_weight in np.arange(0.1, 1.0, 0.1):
                xgb_weight = 1 - sarima_weight
                ensemble_pred = sarima_weight * sarima_pred + xgb_weight * xgb_pred
                
                mape = mean_absolute_percentage_error(val_data[self.target_column], ensemble_pred)
                
                if mape < best_mape:
                    best_mape = mape
                    best_weights = {'sarima': sarima_weight, 'xgboost': xgb_weight}
            
            self.ensemble_weights = best_weights
            print(f"Optimized ensemble weights: SARIMA={best_weights['sarima']:.2f}, XGBoost={best_weights['xgboost']:.2f}")
            
        except Exception as e:
            print(f"Error in weight optimization: {str(e)}")
            # Fall back to default weights
            self.ensemble_weights = {'sarima': 0.6, 'xgboost': 0.4}
    
    def predict(self, prediction_date_str):
        """Generate predictions for the specified date and few days ahead."""
        if not self.trained:
            print("Model is not trained yet")
            return pd.DataFrame()
        
        try:
            prediction_date = datetime.strptime(prediction_date_str, '%Y-%m-%d')
            
            # Create dataframe for 5 days ahead
            future_dates = [prediction_date + timedelta(days=i) for i in range(5)]
            predictions_df = pd.DataFrame({
                'date': future_dates,
                'predicted_price': 0.0,
                'lower_bound': 0.0,
                'upper_bound': 0.0
            })
            
            # In the predict method where you're getting the SARIMA forecast
            sarima_mean = self.sarima_fit.forecast(steps=5)
            # Convert to a list or numpy array to index by position instead of by Pandas index
            sarima_mean_values = sarima_mean.values if hasattr(sarima_mean, 'values') else np.array(sarima_mean)

            # Then use sarima_mean_values instead of sarima_mean when indexing
            feature_dict['price_lag_1'] = sarima_mean_values[0]
            sarima_conf_int = self.sarima_fit.get_forecast(steps=5).conf_int(alpha=0.05)
            
            # For XGBoost, we need features for future dates
            # Get the last known values from the fitted scaler
            # This is a more reliable approach than before
            feature_means = self.scaler.mean_[:len(self.feature_columns)]
            
            xgb_predictions = []
            
            for i, future_date in enumerate(future_dates):
                # Create feature vector for XGBoost - start with basic features
                feature_dict = {}
                
                # Set basic price features using means from training data
                for j, col in enumerate(self.feature_columns):
                    feature_dict[col] = feature_means[j]
                
                # Add derived features
                feature_dict['price_ratio'] = feature_dict['Max_Price'] / feature_dict['Min_Price'] if 'Min_Price' in feature_dict and feature_dict['Min_Price'] != 0 else 1.0
                feature_dict['price_diff'] = feature_dict['Max_Price'] - feature_dict['Min_Price'] if 'Min_Price' in feature_dict and 'Max_Price' in feature_dict else 0.0
                
                # Update time features
                feature_dict['month'] = future_date.month
                feature_dict['day_of_week'] = future_date.weekday()
                feature_dict['day_of_year'] = future_date.timetuple().tm_yday
                
                # For lag and rolling features
                if i == 0:
                    # Use SARIMA prediction for first day
                    feature_dict['price_lag_1'] = sarima_mean[0]
                    feature_dict['price_lag_3'] = sarima_mean[0]  # Simplified
                    feature_dict['price_lag_7'] = sarima_mean[0]  # Simplified
                    feature_dict['price_rolling_mean_3'] = sarima_mean[0]
                    feature_dict['price_rolling_mean_7'] = sarima_mean[0]
                    feature_dict['price_rolling_mean_14'] = sarima_mean[0]
                    feature_dict['price_rolling_std_3'] = 0.1 * sarima_mean[0]
                    feature_dict['price_rolling_std_7'] = 0.1 * sarima_mean[0]
                    feature_dict['price_rolling_std_14'] = 0.1 * sarima_mean[0]
                else:
                    # Use previous predictions
                    feature_dict['price_lag_1'] = xgb_predictions[i-1]
                    
                    if i >= 3:
                        feature_dict['price_lag_3'] = xgb_predictions[i-3]
                    else:
                        feature_dict['price_lag_3'] = sarima_mean[0]
                        
                    if i >= 7:
                        feature_dict['price_lag_7'] = xgb_predictions[i-7]
                    else:
                        feature_dict['price_lag_7'] = sarima_mean[0]
                    
                    # Update rolling features
                    prev_preds = xgb_predictions[:i]
                    
                    # For 3-day window
                    if len(prev_preds) >= 3:
                        feature_dict['price_rolling_mean_3'] = np.mean(prev_preds[-3:])
                        feature_dict['price_rolling_std_3'] = np.std(prev_preds[-3:]) if len(prev_preds) > 1 else 0.1 * feature_dict['price_rolling_mean_3']
                    else:
                        feature_dict['price_rolling_mean_3'] = np.mean(prev_preds + [sarima_mean[0]] * (3 - len(prev_preds)))
                        feature_dict['price_rolling_std_3'] = 0.1 * feature_dict['price_rolling_mean_3']
                    
                    # For 7-day window
                    if len(prev_preds) >= 7:
                        feature_dict['price_rolling_mean_7'] = np.mean(prev_preds[-7:])
                        feature_dict['price_rolling_std_7'] = np.std(prev_preds[-7:]) if len(prev_preds) > 1 else 0.1 * feature_dict['price_rolling_mean_7']
                    else:
                        feature_dict['price_rolling_mean_7'] = np.mean(prev_preds + [sarima_mean[0]] * (7 - len(prev_preds)))
                        feature_dict['price_rolling_std_7'] = 0.1 * feature_dict['price_rolling_mean_7']
                    
                    # For 14-day window (simplified since we're only predicting 5 days)
                    feature_dict['price_rolling_mean_14'] = feature_dict['price_rolling_mean_7']  # Simplification
                    feature_dict['price_rolling_std_14'] = feature_dict['price_rolling_std_7']    # Simplification
                
                # Prepare feature vector for scaling
                feature_list = []
                for col in self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                'price_rolling_mean_7', 'price_rolling_std_7']:
                    if col in feature_dict:
                        feature_list.append(feature_dict[col])
                    else:
                        # For any missing features, use 0 or another reasonable default
                        feature_list.append(0.0)
                
                # Scale features and predict
                scaled_features = self.scaler.transform([feature_list])
                xgb_pred = self.xgb_model.predict(scaled_features)[0]
                xgb_predictions.append(xgb_pred)
            
            # Combine predictions using ensemble weights
            for i in range(5):
                sarima_pred = sarima_mean_values[i]
                xgb_pred = xgb_predictions[i]
                
                # Weighted ensemble prediction
                ensemble_pred = (self.ensemble_weights['sarima'] * sarima_pred + 
                            self.ensemble_weights['xgboost'] * xgb_pred)
                
                # Calculate confidence intervals
                lower_bound = sarima_conf_int.iloc[i, 0] * self.ensemble_weights['sarima'] + xgb_pred * (1 - 0.1) * self.ensemble_weights['xgboost']
                upper_bound = sarima_conf_int.iloc[i, 1] * self.ensemble_weights['sarima'] + xgb_pred * (1 + 0.1) * self.ensemble_weights['xgboost']
                
                predictions_df.loc[i, 'predicted_price'] = ensemble_pred
                predictions_df.loc[i, 'lower_bound'] = lower_bound
                predictions_df.loc[i, 'upper_bound'] = upper_bound
            
            return predictions_df
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return pd.DataFrame()
        
    def backtesting(self, df):
        """Perform backtesting on historical data."""
        try:
            # Process the data
            data = self.preprocess_data(df)
            if data is None or len(data) < 30:
                print("Insufficient data for backtesting")
                return None
            
            # Use the last 20% for testing
            test_size = int(len(data) * 0.2)
            train_data = data.iloc[:-test_size]
            test_data = data.iloc[-test_size:]
            
            # Train SARIMA on training data
            sarima_model = SARIMAX(
                train_data[self.target_column],
                order=(1, 1, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarima_fit = sarima_model.fit(disp=False)
            
            # FIXED: Get SARIMA predictions for test period using the fitted model
            sarima_pred = sarima_fit.forecast(steps=len(test_data))
            
            # Train XGBoost on training data
            X_train = train_data[self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                       'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                       'price_rolling_mean_7', 'price_rolling_std_7']]
            y_train = train_data[self.target_column]
            
            X_test = test_data[self.feature_columns + ['price_ratio', 'price_diff', 'month', 'day_of_week', 
                                                     'price_lag_1', 'price_lag_3', 'price_lag_7',
                                                     'price_rolling_mean_7', 'price_rolling_std_7']]
            y_test = test_data[self.target_column]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost
            xgb_model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                objective='reg:squarederror',
                eval_metric='rmse'
            )
            eval_set = [(X_train_scaled, y_train)]
            xgb_model.fit(X_train_scaled, y_train, eval_set=eval_set, verbose=False)
            
            # Get XGBoost predictions
            xgb_pred = xgb_model.predict(X_test_scaled)
            
            # Compute ensemble predictions
            ensemble_pred = (self.ensemble_weights['sarima'] * sarima_pred + 
                            self.ensemble_weights['xgboost'] * xgb_pred)
            
            # Calculate metrics
            sarima_rmse = np.sqrt(mean_squared_error(y_test, sarima_pred))
            xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
            ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            
            sarima_mae = mean_absolute_error(y_test, sarima_pred)
            xgb_mae = mean_absolute_error(y_test, xgb_pred)
            ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
            
            sarima_r2 = r2_score(y_test, sarima_pred)
            xgb_r2 = r2_score(y_test, xgb_pred)
            ensemble_r2 = r2_score(y_test, ensemble_pred)
            
            # Calculate MAPE
            sarima_mape = mean_absolute_percentage_error(y_test, sarima_pred) * 100
            xgb_mape = mean_absolute_percentage_error(y_test, xgb_pred) * 100
            ensemble_mape = mean_absolute_percentage_error(y_test, ensemble_pred) * 100
            
            # Create results dictionary
            results = {
                'test_dates': test_data[self.date_column],
                'actual_prices': y_test.values,
                'sarima_predictions': sarima_pred,
                'xgb_predictions': xgb_pred,
                'ensemble_predictions': ensemble_pred,
                'metrics': {
                    'rmse': round(ensemble_rmse, 2),
                    'mae': round(ensemble_mae, 2),
                    'r2': round(ensemble_r2, 4),
                    'mape': round(ensemble_mape, 2),
                    'sarima_rmse': round(sarima_rmse, 2),
                    'xgb_rmse': round(xgb_rmse, 2),
                    'sarima_r2': round(sarima_r2, 4),
                    'xgb_r2': round(xgb_r2, 4),
                    'sarima_mape': round(sarima_mape, 2),
                    'xgb_mape': round(xgb_mape, 2)
                }
            }
            
            print(f"Backtesting Results:")
            print(f"Ensemble RMSE: {ensemble_rmse:.2f}")
            print(f"Ensemble MAE: {ensemble_mae:.2f}")
            print(f"Ensemble R²: {ensemble_r2:.4f}")
            print(f"Ensemble MAPE: {ensemble_mape:.2f}%")
            
            return results
            
        except Exception as e:
            print(f"Error in backtesting: {str(e)}")
            return None


