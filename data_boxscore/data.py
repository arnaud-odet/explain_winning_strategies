import pandas as pd
import numpy as np
import os

from .preprocess import make_custom_preproc, feature_check
from .elo import compute_ELO
from .constants import DATA_PATH


def custom_query_df(df, team:str = None, season:str=None, league:str = None):
    query_df = df.copy()
    if team is not None :
        query_df = query_df[[home ==team or away ==team for home,away in zip(query_df['home_team'],query_df['away_team'])]]
    if season is not None :
        query_df = query_df[query_df['season']==season]
    if league is not None :
        query_df = query_df[query_df['league']==league]
    return query_df.reset_index(drop=True)


def prepare_dataset(gdf, 
                    features_standard: list=[], 
                    features_minmax: list=[], 
                    features_robust: list=[], 
                    features_perc: list=[],
                    features_no_scaling: list=[],
                    team:str = None, 
                    season:str=None, 
                    league:str = None,
                    split:tuple=(0.1,0.6,0.15,0.15),
                    excluded_columns:list = []):

        
    if 'split' not in gdf.columns :
        n = gdf.shape[0]
        eps = 0.1
        inds_split = np.array([n * sum(split[:i+1]) - eps for i in range(len(split)-1)])
        split_col = [np.searchsorted(np.array(inds_split), i) for i in range(n)] 
        gdf['split'] = split_col

    df = gdf.copy()

    for col in excluded_columns:
        try :
            df.drop(columns = col, inplace = True)
        except :
            pass

    if features_standard == [] and features_minmax == [] and features_robust==[] and features_perc == [] and features_no_scaling == [] :
        features_standard = df.drop(columns = ['split', 'home_win']).select_dtypes(include=['number']).columns

    standard_features, minmax_features, robust_features, perc_features, no_scaling_features = feature_check(
        df,
        features_standard,
        features_minmax,
        features_robust,
        features_perc,
        features_no_scaling)
    
    preprocessor = make_custom_preproc(standard_features=standard_features, 
                                       minmax_features=minmax_features, 
                                       robust_features=robust_features, 
                                       perc_features=perc_features,
                                       no_scaling_features=no_scaling_features)
    train_df = df.query("split == 1")
    y_train = train_df['home_win']
    val_df = df.query("split == 2")
    y_val = val_df['home_win']
    test_df = df.query("split == 3")
    y_test = test_df['home_win']
    
    X_train = preprocessor.fit_transform(train_df)
    X_train = pd.DataFrame(X_train, columns=preprocessor.get_feature_names_out())
    columns_name_dict = {i: i[i.find("__") + 2 :] for i in X_train.columns}
    X_train.rename(columns=columns_name_dict, inplace=True)
    
    X_val = preprocessor.transform(val_df)
    X_val = pd.DataFrame(X_val, columns = preprocessor.get_feature_names_out())
    X_val.rename(columns = columns_name_dict, inplace=True) 
    X_test = preprocessor.transform(test_df)
    X_test = pd.DataFrame(X_test, columns = preprocessor.get_feature_names_out())
    X_test.rename(columns = columns_name_dict, inplace=True) 
    
    if team is not None or season is not None or league is not None :
        query_df = custom_query_df(df, team = team, season = season, league = league)
    else : 
        query_df = test_df.copy()  
    X_query = preprocessor.transform(query_df)
    y_query = query_df['home_win']
    X_query = pd.DataFrame(X_query, columns = preprocessor.get_feature_names_out())
    X_query.rename(columns = columns_name_dict, inplace=True)     
    
    return X_train, y_train, X_val, y_val, X_test, y_test, X_query, y_query
    
    
def load_dataframes(filename:str, 
                    use_ELO:bool=False, 
                    k_elo:int = 20,
                    features_standard: list=[], 
                    features_minmax: list=[], 
                    features_robust: list=[], 
                    features_perc: list=[],
                    features_no_scaling: list=[],
                    team:str = None, 
                    season:str=None, 
                    league:str = None,
                    split:tuple = (0.1,0.6,0.15,0.15),
                    excluded_columns = []):
    
    
    file_path = os.path.join(DATA_PATH,filename)
    gdf = pd.read_csv(file_path, index_col=0)
    if use_ELO :
        gdf = compute_ELO(gdf, k_elo)   
    
    if 'split' not in gdf.columns and (len(split) != 4 or sum(split) != 1) :
        raise ValueError('Please specify a split column if your game dataframe or a split tupe with 4 values (elo_ignored_games_share, train_share, val_share, test_share) summing to 1')
    
    X_train, y_train, X_val, y_val, X_test, y_test, X_query, y_query = prepare_dataset(gdf, 
                                                                                    features_standard = features_standard, 
                                                                                    features_minmax = features_minmax, 
                                                                                    features_robust = features_robust, 
                                                                                    features_perc = features_perc,
                                                                                    features_no_scaling = features_no_scaling,                                                                                       
                                                                                    team = team, 
                                                                                    season = season, 
                                                                                    league = league,
                                                                                    split=split,
                                                                                    excluded_columns = excluded_columns)
    
    return gdf, X_train, y_train, X_val, y_val, X_test, y_test, X_query, y_query
    