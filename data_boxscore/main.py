import os
import pandas as pd
import numpy as np
from .data import load_dataframes
from .models import compare_models_w_hpo
from .constants import OUTPUT_PATH, features_minmax, features_no_scaling, features_perc, features_robust, features_standard, excluded_columns


filename = 'data.csv'

n_iter_ml = 200
n_iter_ann = 50
k_elo = 20

if __name__ == '__main__' :
    _, X_train, y_train, X_val, y_val, X_test, y_test, _, _ = load_dataframes(filename=filename,                                                                              
                                                                                    use_ELO = True,
                                                                                    k_elo = k_elo,
                                                                                    features_standard = features_standard, 
                                                                                    features_minmax = features_minmax, 
                                                                                    features_robust = features_robust, 
                                                                                    features_perc = features_perc,
                                                                                    features_no_scaling = features_no_scaling, 
                                                                                    team = None,
                                                                                    season = None,
                                                                                    league = None,
                                                                                    excluded_columns=excluded_columns)
    c_df = compare_models_w_hpo(X_train, y_train, X_val, y_val, X_test, y_test, 
                                cv = 0, 
                                n_iter_ml = n_iter_ml, 
                                n_iter_ann= n_iter_ann)
    path = os.path.join(OUTPUT_PATH, 'model_comparison.csv')
    c_df.to_csv(path)