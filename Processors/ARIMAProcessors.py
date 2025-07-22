import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from neuralforecast.auto import AutoLSTM
from neuralforecast.tsdataset import TimeSeriesDataset

from datetime import datetime, timedelta

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoCES, AutoETS
from statsforecast.arima import arima_string
import plotly.graph_objects as go
import matplotlib.colors as mcolors

from pathlib import Path


class FixedARIMAProcessor:
    def __init__(self, overall_df, dates):
        self.overall_df = overall_df
        self.overall_df_value_col = "value"
        self.dates = dates
        
        self.dfs = []
        self.sf = None
        self.forecast = None
        self.plotting_df = pd.DataFrame()
        
        self.mae = None
        self.mse = None
        self.mape = None
        self.nmse = None
        
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
    
    def create_fixed_model(self, h, freq, season_length, model_name, level = [], save = False):
        if not self.sf:
            self.sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq = freq)
            self.sf.fit(self.dfs[0])
            if save:
                here = Path(__file__).resolve().parent
                models_dir = here.parent / "AutoARIMA Models" / "fixed_models"
                models_dir.mkdir(parents=True, exist_ok=True)
                self.sf.save(path = models_dir / f"{model_name}.pkl")
        
        start_date = datetime.strptime(self.dates[0], "%Y-%m-%d")
        ending_date = datetime.strptime(self.dates[-1], "%Y-%m-%d") + timedelta(weeks = h)

        prediction_horizon = abs((ending_date - start_date).days) // 7
        
        if not level:
            self.forecast = self.sf.predict(h = prediction_horizon)
        else:
            self.forecast = self.sf.predict(h = prediction_horizon, level = level)
        
        self.forecast.sort_values(by = ["ds", "unique_id"], inplace = True)
        
    def load_model(self, path):
        self.sf = StatsForecast.load(path = path)
    
    def generate_color_map(self, columns, cmap_name = "viridis"):
        intervals = set()
        for col in columns:
            if col in ("ds", "AutoARIMA", "unique_id"):
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
        color_mapping['AutoARIMA'] = mcolors.to_hex(cmap(median_value))  # center of the colormap
        for interval, point in zip(intervals, sample_points):
            color_mapping[interval] = mcolors.to_hex(cmap(point))
    
        return color_mapping
    
    
    def create_graph(self, unique_id):
        
        first_forecast_date = self.forecast.iloc[0]["ds"]
        final_forecast_date = self.forecast.iloc[-1]["ds"]
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
        sliced_fc = self.forecast[self.forecast["unique_id"] == unique_id]
        
        if len(sliced_fc.columns) > 3:
            self.color_mapping = self.generate_color_map(columns = sliced_fc.columns)
    
    
        fig = go.Figure()
        fig.add_trace(go.Scatter(x = sliced_df["ds"], y = sliced_df["y"], mode = "lines", name = "Real Data"))
        
        for col in self.forecast.columns:
            if "hi" in col:
                number = col[-2:]
                fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col,
                                         line = dict(color = self.color_mapping[number])))
                                         
        for col in self.forecast.columns:
            if "lo" in col:
                number = col[-2:]
                fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col,
                                         fill = "tonexty", fillcolor = self.color_mapping[number], line = dict(color = self.color_mapping[number])))
            
        for col in self.forecast.columns:
            if col == "AutoARIMA":
                fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping["AutoARIMA"])))
        
        fig.update_layout(title = f"Fixed Parameter ARIMA Predictions, {unique_id, self.dates[0]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
        fig.show()
        
        
    def create_metrics(self, unique_id):
        col_string = "AutoARIMA"
        first_forecast_date = self.forecast.iloc[0]["ds"]
        final_forecast_date = self.forecast.iloc[-1]["ds"]
        
        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
        sliced_fc = self.forecast[self.forecast["unique_id"] == unique_id]
        
        mae = mean_absolute_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
        mape = mean_absolute_percentage_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
        mse = mean_squared_error(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"], sliced_fc[col_string])
        nmse = mse/np.var(sliced_df[(sliced_df["ds"] >= first_forecast_date) & (sliced_df["ds"] <= final_forecast_date)]["y"])
        
        return (mae, mse, mape, nmse)

    def create_metrics_df(self):
        for unique_id in self.overall_df["unique_id"].unique():
            mae, mse, mape, nmse = self.create_metrics(unique_id)
            self.metrics_df.loc[len(self.metrics_df)] = [self.dates[0], unique_id, mae, mse, mape, nmse]
        
        self.metrics_df.sort_values(by = ["Reference Date", "unique_id"], inplace = True)
    
    def create_display_df(self):
        for index, row in self.forecast.iterrows():
            reference_date = self.dates[0]
            target_end_date = row[1]
            unique_id = row[0]
            
            for col in self.forecast.columns:
                value = self.forecast.loc[index, col]
                if "lo" in col:
                    number = int (col[-2:])
                    alpha = 1 - (number / 100)
                    quantile = alpha / 2
                    self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
                
                if col == "AutoARIMA":
                    quantile = 0.5
                    self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
                
                elif "hi" in col:
                    number = int (col[-2:])
                    alpha = 1 - (number / 100)
                    quantile = 1 - (alpha / 2)
                    self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
        
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "unique_id", "Quantile"], inplace = True)
        

        
        
class UpdatingARIMAProcessor:
    def __init__(self, overall_df, dates):
        self.overall_df = overall_df
        self.overall_df_value_col = "value"
        self.dates = dates
        
        self.dfs = []
        self.sfs = []
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
            df = self.overall_df[self.overall_df["ds"] <= pd.Timestamp(date)]
            self.dfs.append(df)
            
    def create_models(self, h, freq, season_length, model_names, level = [], save = False):
        
        if not self.sfs:
            for i in range(len(self.dfs)):
                sf = StatsForecast(models=[AutoARIMA(season_length=season_length)], freq = freq)
                sf.fit(self.dfs[i])
                self.sfs.append(sf)
                if save:
                    here = Path(__file__).resolve().parent
                    models_dir = here.parent / "AutoARIMA Models" / "updating_models"
                    models_dir.mkdir(parents=True, exist_ok=True)
                    sf.save(models_dir / f"{model_names[i]}.pkl")

      
        for i in range(len(self.dfs)):
            fc = pd.DataFrame()
            if not level:
                fc = self.sfs[i].predict(h = h)
            else:
                fc = self.sfs[i].predict(h = h, level = level)
            
            fc.sort_values(by = ["ds", "unique_id"], inplace = True)
            self.forecasts.append(fc)
    
    def load_models(self, paths):
        for i in range(len(paths)):
            sf = StatsForecast.load(path = paths[i])
            self.sfs.append(sf)
    
    def generate_color_map(self, columns, cmap_name = "viridis"):
        intervals = set()
        for col in columns:
            if col in ("AutoARIMA", "ds", "unique_id"):
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
        color_mapping['AutoARIMA'] = mcolors.to_hex(cmap(median_value))  # center of the colormap
        for interval, point in zip(intervals, sample_points):
            color_mapping[interval] = mcolors.to_hex(cmap(point))
    
        return color_mapping
    
    
    
    
    def create_graph(self, unique_id):

        sliced_df = self.overall_df[self.overall_df["unique_id"] == unique_id]
 
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
                if col == "AutoARIMA":
                    fig.add_trace(go.Scatter(x = sliced_fc["ds"], y = sliced_fc[col], mode = "lines", name = col, 
                                             line = dict(color = self.color_mapping["AutoARIMA"])))
                    
            fig.update_layout(title = f"Updating Parameter ARIMA Predictions, {unique_id, self.dates[i]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
            fig.show()
        
    def calculate_metrics(self, unique_id):
        col_string = "AutoARIMA"
        
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

                    if col == "AutoARIMA":
                        quantile = 0.5
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]

                    elif "hi" in col:
                        number = int (col[-2:])
                        alpha = 1 - (number / 100)
                        quantile = 1 - (alpha / 2)
                        self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, unique_id, quantile, value]
                
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "unique_id", "Quantile"], inplace = True)