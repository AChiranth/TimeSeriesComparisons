## Project Description

Pipelines for AutoARIMA, AutoLSTM, AutoCES, and AutoETS modeling using Nixtla. User must specify type of model and fixed or updating model. Whole workflow of slicing training data frames, model optimization and parameter saving, model plotting, and metric calculation can be handled with a few function calls.

## Method Description

### create_training_dfs(self, value_col)

**Description**

Creates one training data frame for each date provided at object instantiation. The provided date corresponds to the last date in the training dataframe (i.e. reference date).

**Parameters**
| Name       | Type         | Description                                 |
|------------|--------------|---------------------------------------------|
| `value_col`   | `string`| Value column in dataframe provided at instantiation                      |

### create_fixed_model(self, h, freq, season_length, model_name, level = []) and create_models(self, h, freq, season_length, model_names, level = []) ###

**Description**

Creates a fixed model or multiple models (depending on type of object) and saves file with model parameters.

**Parameters**

| Name       | Type         | Description                                 |
|------------|--------------|---------------------------------------------|
| `h`   | `int`| Prediction horizon for each model                     |
| `freq` | `string`      | Frequency of data  |
| `season_length` | `int`      | Data's length of season  |
| `model_name` | `string`      | For a fixed model object, the name of the saved model file |
| `model_names` | `List[string]`      | For a updating model object, the name of the saved model files |
| `level` | `List[int]`      | Confidence intervals that will create a probabilistic forecast. A probabilistic forecast will generate 2 corresponding quantiles  |

### load_model(self, path) and load_model(self, paths) ###

**Description**

Given a path/paths, load a model with existing tuned parameters into an object.

**Parameters**
| Name       | Type         | Description                                 |
|------------|--------------|---------------------------------------------|
| `path`   | `string`| Path to folder of .pkl and .ckpt files (for LSTM models) or just a .pkl file (all other models). Used for a fixed model        |
| `paths`   | `List[string]`| List of paths corresponding to multiple models. Used for an updating model.|

### create_graph(self) ###

**Description**

Plot the training data, predicted data, and possible quantiles using Plotly. Will return five different plots for an updating model and 

### calculate_metrics(self) and create_metrics(self) ###

**Description**

Calculate Mean Absolute Error, Mean Squared Error, Mean Absolute Percentage Error, and Normalized Mean Squared Error for various forecasts. 

