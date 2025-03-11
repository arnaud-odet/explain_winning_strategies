import pandas as pd
import numpy as np

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer

def do_nothing(X):
    return X

def percentage(X):
    return X/100

class IdentityTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features
    
class PercentageTransformer(FunctionTransformer):
    def get_feature_names_out(self, input_features=None):
        return input_features

def make_custom_preproc(standard_features = [], 
                        minmax_features = [], 
                        robust_features = [], 
                        perc_features=[], 
                        no_scaling_features = []):


    num_std_transformer = make_pipeline(StandardScaler())
    num_mm_transformer = make_pipeline(MinMaxScaler())
    num_rob_transformer = make_pipeline(RobustScaler())

    col_transf = make_column_transformer(
        (num_std_transformer, standard_features),
        (num_mm_transformer, minmax_features),
        (num_rob_transformer, robust_features),
        (IdentityTransformer(func=do_nothing, validate=False),no_scaling_features),
        (PercentageTransformer(func=percentage, validate=False),perc_features),
        remainder='drop'
    )

    preprocessor = make_pipeline(col_transf)
    
    return preprocessor

def feature_check(df, all_standard, all_minmax, all_robust, all_perc, all_no_sc):
    standard_features = []
    minmax_features = []
    robust_features = []
    no_scaling_features = []
    perc_features = []

    for feature in df.columns:
        if feature in all_standard:
            standard_features.append(feature)
        elif feature in all_minmax:
            minmax_features.append(feature)
        elif feature in all_robust:
            robust_features.append(feature)
        elif feature in all_perc:
            perc_features.append(feature)
        elif feature in all_no_sc:
            no_scaling_features.append(feature)
    
    return standard_features, minmax_features, robust_features, perc_features, no_scaling_features