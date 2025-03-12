# Explain Winning Strategies in Sports

This repository contains the code used for the paper *An Explainability Approach for Identifying Winning Strategies in Rugby Union*

### Usage 

1. **Install Python 3.10**

2. **Install dependencies** : for simplicity, execute the following command 
  
```
pip install -r requirements.txt
```
3. **Modify your dataset** so it contains the following columns :
* `home_team` and `away_team` : unique identifiers for home and away team respectively,
* either a binary `home_win` or a `diff_score` (computed as home score - away score, mandatory for ELO computation),
* optionally, a `split` with the following values :
  * 0 for game set aside for ELO initialization,
  * 1 for games in the training set
  * 2 for games in the validation set
  * 3 for games in the test set
  * if no `split` column is available, please specify a tuple with the share of each set (4 coordinates required)

4. **Modify `data_boxscore/constants.py`** :
  * `DATA_PATH` and `OUTPUT_PATH` 
  * `excluded_columns` : the columns you want to hide from the model (e.g. number of tries scored, `diff_score`, ...)
  * optionally :
    * `features_standard` : the features to be scaled using `StandardScaler`, 
    * `features_minmax` : the features to be scaled using `MinMaxScaler`, 
    * `features_robust` : the features to be scaled using `RobustScaler`, 
    * `features_perc` : the features to be scaled being simply divided by 100 (percentages), 
    * `features_no_scaling` : the features that do not require to be scaled
    * if all those lists are empty, `StandardScaler` will be used as default.

5. **Empirical model comparison** with user-defined budget : either using `models.ipynb` or `run.sh` and `data_boxscore/main.py`

6. **Explanation** using `explanations.ipynb`. Please change the model defined in the notebook considering the results of step 5. 

### Citation

<span style= 'color:orange'>to be added </span>

### Data availability

The data cannot be directly uploaded to an open platform due to contractual agreements with AIA Enterprise.
