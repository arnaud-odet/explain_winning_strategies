import pandas as pd
import numpy as np

def ELO_exchange(elo_a, elo_b, diff_score, k_elo):
    """
    This function takes on two ELO ratings and the score difference of a game and returns the exchanged ELO_points
    """
    elo_diff = elo_a - elo_b
    proba_win_a = 1 / ( 1 + 10 ** (-elo_diff / 400) )
    
    if diff_score > 0 :
        result_a = 1
    elif diff_score <0 :
        result_a = 0
    else :
        result_a = 0.5
    
    return  k_elo * (result_a - proba_win_a)

def compute_ELO(gdf:pd.DataFrame, k_elo:int = 20, return_elo_end:bool = False):
    elo_diffs = []
    elo_dict =  {team : 0 for team in pd.concat([gdf['home_team'], gdf['away_team']]).drop_duplicates()}
    for index, row in gdf.iterrows():
        elo_h = elo_dict[row['home_team']]
        elo_a = elo_dict[row['away_team']]
        elo_diff = elo_h - elo_a
        elo_diffs.append(elo_diff)
        elo_exch = ELO_exchange(elo_h, elo_a, diff_score=row['diff_score'], k_elo = k_elo)
        elo_dict[row['home_team']] += elo_exch
        elo_dict[row['away_team']] -= elo_exch
    gdf['ELO_diff_before'] = elo_diffs
    if return_elo_end :
        return gdf, elo_dict
    else :
        return gdf
    
def compute_kl_divergence(p, q, bins=100, epsilon=1e-10):
    """
    Compute KL divergence between two (n,1) numpy arrays by histogramming.
    
    Args:
        p: First array of shape (n,1)
        q: Second array of shape (n,1)
        bins: Number of bins for histogram
        epsilon: Small constant to avoid division by zero
        
    Returns:
        KL divergence from p to q
    """
    # Ensure arrays are flattened for histogramming
    p = p.flatten()  # shape: (n,)
    q = q.flatten()  # shape: (n,)
    
    # Find min and max across both arrays to use same bins
    min_val = min(p.min(), q.min())
    max_val = max(p.max(), q.max())
    
    # Create histogram for both arrays using the same bins
    p_hist, bin_edges = np.histogram(p, bins=bins, range=(min_val, max_val), density=True)
    q_hist, _ = np.histogram(q, bins=bins, range=(min_val, max_val), density=True)
    
    # Add small epsilon to avoid log(0) and division by zero
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon
    
    # Normalize to ensure they are valid probability distributions
    p_hist = p_hist / np.sum(p_hist)  # shape: (bins,)
    q_hist = q_hist / np.sum(q_hist)  # shape: (bins,)
    
    # Compute KL divergence: KL(p||q) = Σ p(i) * log(p(i)/q(i))
    kl_div = np.sum(p_hist * np.log(p_hist / q_hist))
    
    return kl_div