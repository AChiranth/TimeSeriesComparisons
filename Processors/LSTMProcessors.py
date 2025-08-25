import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from neuralforecast.auto import AutoLSTM
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.core import NeuralForecast
from neuralforecast.models import LSTM
from datetime import datetime, timedelta
import plotly.graph_objects as go
import matplotlib.colors as mcolors
from neuralforecast.losses.pytorch import MQLoss
import optuna

from pathlib import Path



class FixedModelLSTMProcessor:
    def __init__(self, overall_df, dates):
        self.overall_df = overall_df 
        self.overall_df_value_col = "value"
        self.dates = dates
        self.dfs = []
        
        self.nf = None
        
        self.forecasts = []
        self.plotting_df = pd.DataFrame()
        
        self.maes = []
        self.mses = []
        self.mapes = []
        self.nmses = []
        
        self.metrics_df = pd.DataFrame(columns = ["Reference Date", "unique_id", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(columns = ["Reference Date", "Target End Date", "unique_id", "Quantile", "Prediction"])
        
        self.color_mapping = {}
    
    def create_training_dfs(self, value_col):
        self.overall_df_value_col = value_col
        self.overall_df['y'] = self.overall_df.groupby('unique_id', group_keys = False)['y'] \
            .apply(lambda col: col.interpolate(limit_direction='both'))
        for date in self.dates:
            df = self.overall_df[self.overall_df["ds"] <= pd.Timestamp(date)].copy()
            self.dfs.append(df)
        
    
    def create_fixed_model(self, h, freq, model_name, level = [], config = None, save = False):
        #Creating AutoLSTM model and predicting with hyperparameter tuning by optuna backend. This is based upon the first training dataframe
        
        #Default config
        if config == None:
            config = self.config_wrapper(index = 0, h = h)
        
        #Checking if model has already been loaded in and fit
        if not self.nf:
            if not level:
                self.nf = NeuralForecast(models = [AutoLSTM(h = h, backend = "optuna", config = config, search_alg = optuna.samplers.TPESampler())], freq = freq)
                self.nf.fit(df = self.dfs[0])
            else:
                self.nf = NeuralForecast(models = [AutoLSTM(h = h, backend = "optuna", loss = MQLoss(level = level), config = config, search_alg = optuna.samplers.TPESampler())], 
                                         freq = freq)
                self.nf.fit(df = self.dfs[0])
            if save:
                here = Path(__file__).resolve().parent
                base_dir = here.parent / "AutoLSTM Models" / "fixed_models"
                model_path = base_dir / model_name

                model_path.mkdir(parents=True, exist_ok=True)
                self.nf.save(path=str(model_path), overwrite=True)
                

        for i in range(len(self.dfs)):
            df = self.dfs[i]
            y_hat = self.nf.predict(df = df)

            y_hat.sort_values(by = ["ds", "unique_id"], inplace = True)
            self.forecasts.append(y_hat)
            
    def config_wrapper(self, index, h):
        def config_LSTM(trial):
            input_len = self.dfs[index].groupby("unique_id").size().min()
            max_inp = max(8, input_len - h - 1)
            return {
            "input_size": trial.suggest_int("input_size", 8, max_inp),
            "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", [32, 64, 128]),
            "encoder_n_layers": trial.suggest_int("encoder_n_layers", 1, 3),
            "context_size": trial.suggest_int("context_size", 1, h),
            "decoder_hidden_size": trial.suggest_categorical("decoder_hidden_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "random_seed": trial.suggest_int("random_seed", 1, 99999),
            "max_steps": 1000
            }
    
        return config_LSTM
    
    def load_fixed_model(self, path):
        self.nf = NeuralForecast.load(path = path)
        
    def generate_color_map(self, columns, cmap_name = "viridis"):
        intervals = set()
        for col in columns:
            if 'median' in col:
                continue
            if col in ("unique_id", "ds"):
                continue
            parts = col.split('-')
            number = parts[-1]
            intervals.add(number)
        
        intervals = sorted(intervals, key=int)
        cmap = plt.cm.get_cmap(cmap_name)
        
        n = len(intervals)
        if n > 0:
            half = np.linspace(0.2, 0.45, n // 2, endpoint=True)[::-1]  # lower intervals
            upper = np.linspace(0.55, 0.8, n - n // 2, endpoint=True)  # higher intervals
            sample_points = np.concatenate([half, upper])
        else:
            sample_points = np.array([])  # just median
        
        median_value = np.median(sample_points)
        
        color_mapping = {}
        color_mapping['median'] = mcolors.to_hex(cmap(median_value))  # center of the colormap
        for interval, point in zip(intervals, sample_points):
            color_mapping[interval] = mcolors.to_hex(cmap(point))
    
        return color_mapping
    
    def create_graph(self, unique_id):
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
        
        #Create color map for various confidence bands, only if levels are present
        if len(self.forecasts[0].columns) > 3:
            self.color_mapping = self.generate_color_map(columns = self.forecasts[0].columns)
                
        
        for i in range(len(self.forecasts)):
            #Plot the overall Real Data
            
            first_forecast_date = self.forecasts[i].iloc[0]["ds"]
            final_forecast_date = self.forecasts[i].iloc[-1]["ds"]
            
            sliced_fc = self.forecasts[i][self.forecasts[i]["unique_id"] == unique_id]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = sliced_df["ds"], y = sliced_df["y"], mode = "lines", name = "Real Data"))

            
            for col in self.forecasts[i].columns:
                #Plot his first
                if "hi" in col:
                    number = col[-2:]
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping[number])))
            
            for col in self.forecasts[i].columns:
                #Lows will go to corresponding his
                if "lo" in col:
                    number = col[-2:]
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             fill = "tonexty", fillcolor = self.color_mapping[number], line = dict(color = self.color_mapping[number])))
            
            for col in self.forecasts[i].columns:
                #Median gets plotted last
                if "median" in col:
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping["median"])))
            
            #Case for if confidence interval not present
            for col in self.forecasts[i].columns:
                if col == "AutoLSTM":
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y =sliced_fc["AutoLSTM"], mode = "lines", name = "AutoLSTM"))
            
            
            fig.update_layout(title = f"Fixed Parameter LSTM Predictions, {unique_id, self.dates[i]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
            fig.show()
            
    def calculate_metrics(self, unique_id):
        
        col_string = "AutoLSTM"
        
        if "AutoLSTM-median" in self.forecasts[0].columns:
            col_string = "AutoLSTM-median"
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
      
        maes = []
        mses = []
        mapes = []
        nmses = []
        
        for i in range(len(self.forecasts)):
            
            first_forecast_date = self.forecasts[i].iloc[0]["ds"]
            final_forecast_date = self.forecasts[i].iloc[-1]["ds"]
            
            sliced_fc = self.forecasts[i][self.forecasts[i]["unique_id"] == unique_id]
            
            mae = mean_absolute_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            mape = mean_absolute_percentage_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            mse = mean_squared_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            nmse = mse/np.var(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"])
            
            maes.append(mae)
            mses.append(mse)
            mapes.append(mape)
            nmses.append(nmse)
        
        return (maes, mses, mapes, nmses)
    
    
    def create_metrics_df(self):
        for unique_id in self.overall_df["unique_id"].unique():
            maes, mses, mapes, nmses = self.calculate_metrics(unique_id)
            for i in range(len(self.dates)):
                self.metrics_df.loc[len(self.metrics_df)] = [self.dates[i], unique_id, maes[i], mses[i], mapes[i], nmses[i]]
        
            
    def create_display_df(self):
        for i in range(len(self.forecasts)):
            for index, row in self.forecasts[i].iterrows():
                reference_date = self.dates[i]
                target_end_date = row[1]
                unique_id = row[0]
                
                for col in self.forecasts[i].columns:
                    value = self.forecasts[i].loc[index, col]
                    if "lo" in col:
                        number = int (col[-2:])
                        alpha = 1 - (number / 100)
                        quantile = alpha / 2
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]

                    if col == "AutoLSTM" or "median" in col:
                        quantile = 0.5
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]

                    elif "hi" in col:
                        number = int (col[-2:])
                        alpha = 1 - (number / 100)
                        quantile = 1 - (alpha / 2)
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
                
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "unique_id", "Quantile"], inplace = True)
        
        

class UpdatingModelLSTMProcessor:
    def __init__(self, overall_df, dates):
        self.overall_df = overall_df
        self.overall_df_value_col = "value"
        self.dates = dates
        self.dfs = []
        
        self.nfs = []
        
        self.forecasts = []
        self.plotting_df = pd.DataFrame()
        
        self.maes = []
        self.mses = []
        self.mapes = []
        self.nmses = []
        
        self.metrics_df = pd.DataFrame(columns = ["Reference Date", "unique_id", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(columns = ["Reference Date", "Target End Date", "unique_id", "Quantile", "Prediction"])
    
    def create_training_dfs(self, value_col):
        self.overall_df_value_col = value_col
        self.overall_df['y'] = self.overall_df.groupby('unique_id', group_keys = False)['y'] \
            .apply(lambda col: col.interpolate(limit_direction='both'))
        for date in self.dates:
            df = self.overall_df[self.overall_df["ds"] <= pd.Timestamp(date)]
            self.dfs.append(df)
    
    def create_models(self, h, freq, model_names, level = [], config = None, save = False):
        if not self.nfs:
            for i in range(len(self.dfs)):
                #Default config
                if config == None:
                    config = self.config_wrapper(index = i, h = h)
                if not level:   
                    nf = NeuralForecast(models = [AutoLSTM(h = h, backend = "optuna", config = config, search_alg = optuna.samplers.TPESampler())], freq = freq)
                    nf.fit(df = self.dfs[i])
                else:
                    nf = NeuralForecast(models = [AutoLSTM(h = h, backend = "optuna", loss = MQLoss(level = level), config = config, search_alg = optuna.samplers.TPESampler())], 
                                        freq = freq)
                    nf.fit(df = self.dfs[i])
                
                self.nfs.append(nf)
                if save:
                    
                    here = Path(__file__).resolve().parent
                    base_dir = here.parent / "AutoLSTM Models" / "updating_models"
                    model_path = base_dir / model_names[i]

                    model_path.mkdir(parents=True, exist_ok=True)
                    self.nfs[i].save(path=str(model_path), overwrite=True)

        
        for i in range(len(self.dfs)):
            y_hat = self.nfs[i].predict(df = self.dfs[i])
            y_hat.sort_values(by = ["ds", "unique_id"], inplace = True)
            self.forecasts.append(y_hat)
            
    def config_wrapper(self, index, h):
        def config_LSTM(trial):
            input_len = self.dfs[index].groupby("unique_id").size().min()
            max_inp = max(8, input_len - h - 1)
            return {
            "input_size": trial.suggest_int("input_size", 8, max_inp),
            "encoder_hidden_size": trial.suggest_categorical("encoder_hidden_size", [32, 64, 128]),
            "encoder_n_layers": trial.suggest_int("encoder_n_layers", 1, 3),
            "context_size": trial.suggest_int("context_size", 1, h),
            "decoder_hidden_size": trial.suggest_categorical("decoder_hidden_size", [32, 64, 128]),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
            "random_seed": trial.suggest_int("random_seed", 1, 99999),
            "max_steps": 1000
            }
    
        return config_LSTM
    
    def load_models(self, paths):
        for i in range(len(paths)):
            nf = NeuralForecast.load(path = paths[i])
            self.nfs.append(nf)
    
        
    def generate_color_map(self, columns, cmap_name = "viridis"):
        intervals = set()
        for col in columns:
            if 'median' in col:
                continue
            if col in ("unique_id", "ds"):
                continue
            parts = col.split('-')
            number = parts[-1]
            intervals.add(number)
        
        intervals = sorted(intervals, key=int)
        cmap = plt.cm.get_cmap(cmap_name)
        
        n = len(intervals)
        if n > 0:
            half = np.linspace(0.2, 0.45, n // 2, endpoint=True)[::-1]  # lower intervals
            upper = np.linspace(0.55, 0.8, n - n // 2, endpoint=True)  # higher intervals
            sample_points = np.concatenate([half, upper])
        else:
            sample_points = np.array([])  # just median
        
        median_value = np.median(sample_points)
        
        color_mapping = {}
        color_mapping['median'] = mcolors.to_hex(cmap(median_value))  # center of the colormap
        for interval, point in zip(intervals, sample_points):
            color_mapping[interval] = mcolors.to_hex(cmap(point))
    
        return color_mapping
    
    def create_graph(self, unique_id):
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
        
        #Create color map for various confidence bands, only if levels are present
        if len(self.forecasts[0].columns) > 3:
            self.color_mapping = self.generate_color_map(columns = self.forecasts[0].columns)
                
        
        for i in range(len(self.forecasts)):
            #Plot the overall Real Data
            first_forecast_date = self.forecasts[i].iloc[0]["ds"]
            final_forecast_date = self.forecasts[i].iloc[-1]["ds"]
            
            sliced_fc = self.forecasts[i][self.forecasts[i]["unique_id"] == unique_id]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = sliced_df["ds"], y = sliced_df["y"], mode = "lines", name = "Real Data"))

            
            for col in self.forecasts[i].columns:
                #Plot his first
                if "hi" in col:
                    number = col[-2:]
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping[number])))
            
            for col in self.forecasts[i].columns:
                #Lows will go to corresponding his
                if "lo" in col:
                    number = col[-2:]
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             fill = "tonexty", fillcolor = self.color_mapping[number], line = dict(color = self.color_mapping[number])))
            
            for col in self.forecasts[i].columns:
                #Median gets plotted last
                if "median" in col:
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping["median"])))
            
            #Case for if confidence interval not present
            for col in self.forecasts[i].columns:
                if col == "AutoLSTM":
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc["AutoLSTM"], mode = "lines", name = "AutoLSTM"))
            
            
            fig.update_layout(title = f"Updating Parameter LSTM Predictions, {unique_id, self.dates[i]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
            fig.show()
        
    def calculate_metrics(self, unique_id):
        
        col_string = "AutoLSTM"
        
        if "AutoLSTM-median" in self.forecasts[0].columns:
            col_string = "AutoLSTM-median"
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
      
        maes = []
        mses = []
        mapes = []
        nmses = []
        
        for i in range(len(self.forecasts)):
            
            first_forecast_date = self.forecasts[i].iloc[0]["ds"]
            final_forecast_date = self.forecasts[i].iloc[-1]["ds"]
            
            sliced_fc = self.forecasts[i][self.forecasts[i]["unique_id"] == unique_id]
            
            mae = mean_absolute_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            mape = mean_absolute_percentage_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            mse = mean_squared_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
            nmse = mse/np.var(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"])
            
            maes.append(mae)
            mses.append(mse)
            mapes.append(mape)
            nmses.append(nmse)
        
        return (maes, mses, mapes, nmses)
    
    
    def create_metrics_df(self):
        for unique_id in self.overall_df["unique_id"].unique():
            maes, mses, mapes, nmses = self.calculate_metrics(unique_id)
            for i in range(len(self.dates)):
                self.metrics_df.loc[len(self.metrics_df)] = [self.dates[i], unique_id, maes[i], mses[i], mapes[i], nmses[i]]
        
            
    def create_display_df(self):
        for i in range(len(self.forecasts)):
            for index, row in self.forecasts[i].iterrows():
                reference_date = self.dates[i]
                target_end_date = row[1]
                unique_id = row[0]
                
                for col in self.forecasts[i].columns:
                    value = self.forecasts[i].loc[index, col]
                    if "lo" in col:
                        number = int (col[-2:])
                        alpha = 1 - (number / 100)
                        quantile = alpha / 2
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]

                    if col == "AutoLSTM" or "median" in col:
                        quantile = 0.5
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]

                    elif "hi" in col:
                        number = int (col[-2:])
                        alpha = 1 - (number / 100)
                        quantile = 1 - (alpha / 2)
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
                
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "unique_id", "Quantile"], inplace = True)
            
