import numpy as np
import pandas as pd
import shap
from .data import custom_query_df

def inverse_transform_SHAP(shap_values, original_X):
    for i, c in enumerate(original_X.columns):
        shap_values.feature_names[i] = c   
    shap_values.data = original_X.values
    return shap_values

def shap_query(df:pd.DataFrame, X_query ,shap_explainer ,team:str=None, season:str=None, league:str=None):
    
    query_df = custom_query_df(df, team = team, season=season, league=league)
    
    sv_query = shap_explainer(X_query)
    info_cols = ['game_id','home_team', 'away_team'] if 'game_id' in query_df.columns else ['home_team', 'away_team']
    shap_query_df = query_df[info_cols].merge(pd.DataFrame(sv_query.values[:,:,1], columns = sv_query.feature_names), right_index=True,left_index=True)
    if team is not None :
        # Switching SHAP values for away games 
        column_renamer_team_df = {}    
        for ind, row in shap_query_df.iterrows():
            if row['away_team'] == team :
                tmp_row = row.copy()
                for col in sv_query.feature_names :
                    if col[:5] == 'home_':
                        shap_query_df.loc[ind, col] = -tmp_row['away_'+col[5:]]
                        column_renamer_team_df[col] = 'team_' + col[5:]
                    elif col[:5] == 'away_':
                        shap_query_df.loc[ind, col] = -tmp_row['home_'+col[5:]]
                        column_renamer_team_df[col] = 'opponent_' + col[5:]

        shap_query_df.rename(columns = column_renamer_team_df, inplace= True) 
    return shap_query_df