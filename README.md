# CME ML Explorer

This is an app for ML methods for CME arrival time and arrival speed predictions, using various gradient boosting algorithms namely [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html), [LightGBM](https://lightgbm.readthedocs.io/en/latest/), [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/stable/).  The database used for the work is from [Napolitano et. al. 2021](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2021SW002925). The interactive interface is using the Dash interactive Python framework developed by [Plotly](https://plot.ly/).

Dash abstracts away all of the technologies and protocols required to build an interactive web-based application and is a simple and effective way to bind a user interface around your Python code. To learn more check out our [documentation](https://plot.ly/dash).

Try out the [app here](http://cme-pdbm-ml.herokuapp.com/).



## Getting Started

### Using the App

This app lets you interactively explore different ML regression models for probabilistic drag-based models. [Read more about drag based Models here](https://www.swsc-journal.org/articles/swsc/full_html/2018/01/swsc170019/swsc170019.html).

The **Select Dataset** dropdown lets you select the dataset (PDBM-  Napolitano et. al 2021)

The app is divided into two tabs the **Explore dataset** and the **Experiment dataset**. 

The **Explore dataset** allows one to interact with the dataset and perform some exploratory analysis including two scatter plots with the option to choose the X-axis and Y-axis for the scatter plot. The HALO status is transformed from a categorical variable to a numeric variable, with the corresponding keys as follows: 

| Halo Value  | Halo Options|
| :---        |    :----:   |
| 0           | Full Halo (LASCO_da > 270) |
| 1           | Half Halo (LASCO_da > 180) |
| 2           | Partial Halo (LASC_da > 90)|
| 3           | No Halo (otherwise)        |

The Cross-correlation plot for the entire dataset is also presented with the hover text showing the pairwise Pearson's cross-correlation coefficient. The  **CME select** dropdown allows the user to select one cme from the database to have a look at the properties of the CME.  

The **Experiment dataset** tab is where the user can run various Gradient boosting ML models on the dataset. The ratio of the train/test dataset for all the models is .75. 
The **Input features** dropdown lets the user choose the features from the dataset that the user wants to use as input features for the model. By default, the $LASCO_v$ and $v_r$ is selected as the input features.

The **Target features** dropdown lets the user select the target feature for the CME namely the $Arrival_v$ and the $Transit time$. The models work with either one or both features enabled. 

The **Select Model** dropdown lets the user select different types of Gradient boosting models. 

The various parameters of the model selected and trained are also displayed as a table on the top right part of the experiment tab. 

Two plots one for the feature importance and the comparison between the prediction and the ground truth for the trained model are also shown. 


## About the app
### How does it work?
This app is fully written in python using Dash + scikit-learn/LightGBM/CatBoost/XGBoost. All the components are used as input parameters for scikit-learn or NumPy functions, which then generates a model concerning the parameters you changed. The model is then used to perform predictions that are displayed as a scatter plot. The combination of those two libraries lets you quickly write high-level, concise code.

## Built With
* [Dash](https://dash.plot.ly/) - Main server and interactive components
* [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots
* [Scikit-Learn](http://scikit-learn.org/stable/documentation.html) - Run the regression algorithms and generate datasets
* [Gradient Boosting](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html)
* [LightGBM](https://lightgbm.readthedocs.io/en/latest/)
* [CatBoost](https://catboost.ai/)
* [XGBoost](https://xgboost.readthedocs.io/en/stable/)



## Authors
* **Ajay Tiwari** -[@ajeytiwary](https://github.com/ajeytiwary)


## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments
This project is supported by the European Union's Horizon 2020 research and innovation program under grant agreement No. 824064 ([ESCAPE] (https://projectescape.eu/))
## Screenshots
