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


def objective_wrapper(df, config, h, prompt, freq):
    def objective(trial):
        #Split dataframe into train_df and val_df
        #Fix to rolling window for training df and validation df
        train_df = df.iloc[:-1*h]
        val_df = df.loc[-1*h:]
        
        hyperparameters = {}
        
        #Iterate through config for hyperparameter search space
        for param, val in config.items():
            if type(val) == list:
                if type(val[0]) == int:
                    hyperparameters[param] = trial.suggest_int(param, val[0], val[1])
                if type(val[0]) == float:
                    hyperparameters[param] = trial.suggest_float(param, val[0], val[1])
            if type(val == int):
                hyperparameters[param] = val
        
        
        nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'openai-community/gpt2', prompt_prefix = prompt, **hyperparameters)], freq = freq)
        nf.fit(df = train_df)
        y_pred = nf.predict(df = val_df)
        
        return mean_absolute_error(y_pred, val_df["y"])
    
    return objective








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
    
    def create_fixed_model(self, h, freq, model_name, prompt, config = {}, save = False):
        
        
        if not self.nf:
            obj = objective_wrapper(df = self.dfs[0], h = h, freq = freq, prompt = prompt, config = config)
            study = optuna.create_study(direction="minimize")
            study.optimize(obj, n_trials=10)
            
            self.nf = NeuralForecast(models = [TimeLLM(h = h, llm = 'openai-community/gpt2', 
                                                       prompt_prefix = prompt, 
                                                       **study.best_params
                                                      )], freq = freq)
            self.nf.fit(df = self.dfs[0])
            if save:
                self.nf.save(path=f'TimeLLM/fixed_models/{model_name}/', overwrite=True)
        
        for i in range(len(self.dfs)):
            df = self.dfs[i]
            y_hat = self.nf.predict(df = df)

            y_hat.set_index("ds", inplace = True)
            y_hat.drop(columns = "unique_id", inplace = True)
            self.forecasts.append(y_hat)
        

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
            df = self.overall_df.loc[:date]
            df['ds'] = df.index
            df["unique_id"] = "series_1"
            df = df.rename(columns = {value_col: "y"})
            self.dfs.append(df)
    
    def create_models(self, h, freq, model_names, prompt):
        if not self.nfs:
            for i in range(len(self.dfs)):
                nf = NeuralForecast(models = [TimeLLM(h = h, input_size = 3 * h, llm = 'openai-community/gpt2', 
                                                       prompt_prefix = prompt, 
                                                       batch_size = 1, windows_batch_size = 32, inference_windows_batch_size = 32
                                                      )], freq = freq)
                nf.fit(df = self.dfs[i])
                
                self.nfs.append(nf)
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