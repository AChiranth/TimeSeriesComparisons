import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.core import NeuralForecast
from neuralforecast.models import TimeLLM

from datetime import datetime, timedelta

import plotly.graph_objects as go
import matplotlib.colors as mcolors

from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForCausalLM

import optuna


def objective_wrapper(df, config, window_size, h, prompt, freq, train_prop = 0.7, stride = 1):
    def objective(trial):
        #Split dataframe into train_df, val_x, and val_y
        train_size = int(len(df) * train_prop)
        train_df = df.iloc[:train_size]
        val_x = df.iloc[train_size:-h]
        val_y = df.iloc[train_size + window_size:]
        
        #Create windowed dataframes
        windowed_train_df = create_windowed_df(train_df, input_size=window_size, stride = stride)
        windowed_val_x = create_windowed_df(val_x, input_size=window_size, stride = stride)
        windowed_val_y = create_windowed_df(val_y, input_size=h, stride = stride)
        
        hyperparameters = {}
        
        #Iterate through config for hyperparameter search space
        for param, val in config.items():
            #Just one integer
            if type(val) == int:
                hyperparameters[param] = val
            else:
                if val["type"] == "categorical":
                    hyperparameters[param] = trial.suggest_categorical(param, val["values"])
                elif val["type"] == "integer":
                    hyperparameters[param] = trial.suggest_int(param, val["values"][0], val["values"][1])
                elif val["type"] == "float":
                    hyperparameters[param] = trial.suggest_float(param, val["values"][0], val["values"][1])

        
        
        nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'gpt2-medium', d_llm = 1024, prompt_prefix = prompt, **hyperparameters)], freq = freq)
        nf.fit(df = windowed_train_df)
        y_pred = nf.predict(df = windowed_val_x)
        
        return mean_absolute_error(windowed_val_y["y"], y_pred["TimeLLM"])
    
    return objective

def create_windowed_df(df, input_size, stride = 1):
    
    windows = []
    for i in range(0, len(df) - input_size + 1, stride):
        window = df.iloc[i:i+input_size].copy()
        window["unique_id"] = f"window_{i:03d}"
        windows.append(window)
    
    return pd.concat(windows, ignore_index=True)



class FixedModelTimeLLMProcessor():
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
        
        self.metrics_df = pd.DataFrame(columns = ["Reference Date", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(columns = ["Reference Date", "Target End Date", "Quantile", "Prediction"])
        
        self.color_mapping = {}
        
        
        self.llm_config = AutoConfig.from_pretrained("distilgpt2")
        self.llm_tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    
    def create_training_dfs(self, value_col):
        self.overall_df_value_col = value_col
        for date in self.dates:
            df = self.overall_df.loc[:date].copy()
            df["ds"] = df.index
            df["unique_id"] = "series_1"
            df.rename(columns = {value_col: "y"}, inplace=True)
            self.dfs.append(df)
    
    def create_fixed_model(self, h, freq, model_name, prompt, window_size = 16, config = {}, save = False, tune = True, sliding = False, train_prop = 0.7, stride = 1):
        
        
        if not self.nf:
            if tune and not sliding:
                obj = objective_wrapper(df = self.dfs[0], h = h, freq = freq, prompt = prompt, window_size = window_size, config = config, train_prop = train_prop, stride = stride)
                study = optuna.create_study(direction="minimize")
                study.optimize(obj, n_trials=10)

                self.nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'gpt2-medium', d_llm = 1024,
                                                           prompt_prefix = prompt, 
                                                           **study.best_params
                                                          )], freq = freq)
                self.nf.fit(df = self.dfs[0])
            if not tune and not sliding:
                self.nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'gpt2-medium', d_llm = 1024,
                                                           prompt_prefix = prompt, 
                                                           input_size = 16, windows_batch_size = 64,
                                                           inference_windows_batch_size = 64
                                                          )], freq = freq)
                self.nf.fit(df = self.dfs[0])
            if save:
                self.nf.save(path=f'TimeLLM/fixed_models/{model_name}/', overwrite=True)
        
        for i in range(len(self.dfs)):
            if not sliding:
                df = self.dfs[i]
                y_hat = self.nf.predict(df = df)

                y_hat.set_index("ds", inplace = True)
                y_hat.drop(columns = "unique_id", inplace = True)
                self.forecasts.append(y_hat)
            else:
                sliding_df = self.dfs[0].copy()
                forecast = pd.DataFrame()
                for i in range(h):
                    self.nf = NeuralForecast(models = [TimeLLM(h = 1, llm = 'gpt2-medium', d_llm = 1024,
                                                           prompt_prefix = prompt, 
                                                           input_size = 16, windows_batch_size = 64,
                                                           inference_windows_batch_size = 64
                                                          )], freq = "W-SAT")
                
                    self.nf.fit(sliding_df)
                    fc = self.nf.predict(df = sliding_df)
                    del self.nf
                    
                    y_hat = fc.copy()
                    y_hat.set_index("ds", inplace = True)
                    y_hat.drop(columns = "unique_id", inplace = True)
                    forecast = pd.concat([forecast, y_hat])
                    
                    fc = fc.rename(columns = {"TimeLLM": "y"})
                    fc["date"] = fc["ds"].copy()
                    fc.set_index("date", inplace = True)
                    
                    sliding_df = pd.concat([sliding_df, fc])
                    sliding_df = sliding_df.iloc[1:].copy()
                    
                
                self.forecasts.append(forecast)

    def load_fixed_model(self, path):
        self.nf = NeuralForecast.load(path = path)

    def create_graph(self):
        for i in range(len(self.forecasts)):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = self.overall_df.index, y = self.overall_df[self.overall_df_value_col], mode = "lines", name = "Real Data"))
            fig.add_trace(go.Scatter(x = self.forecasts[i].index, y = self.forecasts[i]["TimeLLM"], mode = "lines", name = "TimeLLM"))
            fig.update_layout(title = f"Fixed Parameter TimeLLM Predictions, {self.dates[i]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
            fig.show()
    
    def calculate_metrics(self):
        for i in range(len(self.forecasts)):
            mae = mean_absolute_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            mse = mean_squared_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            mape = mean_absolute_percentage_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            nmse = mse/np.var(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index])
            
            self.maes.append(mae)
            self.mses.append(mse)
            self.mapes.append(mape)
            self.nmses.append(nmse)
            
    def create_metrics_df(self):
        for i in range(len(self.dates)):
            self.metrics_df.loc[len(self.metrics_df)] = [self.dates[i], self.maes[i], self.mses[i], self.mapes[i], self.nmses[i]]
    
    def create_display_df(self):
        for i in range(len(self.forecasts)):
            for index, row in self.forecasts[i].iterrows():
                reference_date = self.dates[i]
                target_end_date = index
                value = self.forecasts[i].loc[target_end_date, "TimeLLM"]
                self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, 0.5, value]
        
        
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "Quantile"], inplace = True)     
        
        

class UpdatingModelTimeLLMProcessor:
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
        
        self.metrics_df = pd.DataFrame(columns = ["Reference Date", "MAE", "MSE", "MAPE", "NMSE"])
        self.display_df = pd.DataFrame(columns = ["Reference Date", "Target End Date", "Quantile", "Prediction"])
        
    def create_training_dfs(self, value_col):
        self.overall_df_value_col = value_col
        for date in self.dates:
            df = self.overall_df.loc[:date].copy()
            df['ds'] = df.index
            df["unique_id"] = "series_1"
            df = df.rename(columns = {value_col: "y"})
            self.dfs.append(df)
    
    def create_models(self, h, freq, model_names, prompt, window_size, config = {}, save = False, train_prop = 0.7, stride = 1):
        if not self.nfs:
            for i in range(len(self.dfs)):
                obj = objective_wrapper(df = self.dfs[i], h = h, freq = freq, prompt = prompt, window_size = window_size, config = config, train_prop = train_prop, stride = stride)
                study = optuna.create_study(direction="minimize")
                study.optimize(obj, n_trials=10)
                nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'gpt2-medium', d_llm = 1024,
                                                       prompt_prefix = prompt, 
                                                       **study.best_params
                                                      )], freq = freq)
                nf.fit(df = self.dfs[i])
                
                self.nfs.append(nf)
                if save:
                    nf.save(path=f'TimeLLM/updating_models/{model_names[i]}/', overwrite = True)
        
        for i in range(len(self.dfs)):
            y_hat = self.nfs[i].predict(df = self.dfs[i])
            y_hat.set_index("ds", inplace = True)
            y_hat.drop(columns = "unique_id", inplace = True)
            self.forecasts.append(y_hat)
    
    def load_models(self, paths):
        for i in range(len(paths)):
            nf = NeuralForecast.load(path = paths[i])
            self.nfs.append(nf)
    
    def create_graph(self):
        for i in range(len(self.forecasts)):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x = self.overall_df.index, y = self.overall_df[self.overall_df_value_col], mode = "lines", name = "Real Data"))
            fig.add_trace(go.Scatter(x = self.forecasts[i].index, y = self.forecasts[i]["TimeLLM"], mode = "lines", name = "TimeLLM"))
            fig.update_layout(title = f"Fixed Parameter TimeLLM Predictions, {self.dates[i]}", xaxis_title = "Date", yaxis_title = "Count", hovermode = "x")
            fig.show()

    def calculate_metrics(self):
        for i in range(len(self.forecasts)):
            mae = mean_absolute_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            mse = mean_squared_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            mape = mean_absolute_percentage_error(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index], self.forecasts[i].iloc[:, 0])
            nmse = mse/np.var(self.overall_df[self.overall_df_value_col].loc[self.forecasts[i].index])
            
            self.maes.append(mae)
            self.mses.append(mse)
            self.mapes.append(mape)
            self.nmses.append(nmse)
            
    def create_metrics_df(self):
        for i in range(len(self.dates)):
            self.metrics_df.loc[len(self.metrics_df)] = [self.dates[i], self.maes[i], self.mses[i], self.mapes[i], self.nmses[i]]
    
    def create_display_df(self):
        for i in range(len(self.forecasts)):
            for index, row in self.forecasts[i].iterrows():
                reference_date = self.dates[i]
                target_end_date = index
                value = self.forecasts[i].loc[target_end_date, "TimeLLM"]
                self.display_df.loc[len(self.display_df)] = [reference_date, target_end_date, 0.5, value]
        
        
        self.display_df.sort_values(by = ["Reference Date", "Target End Date", "Quantile"], inplace = True)