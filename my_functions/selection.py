import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance


def feature_corr(df, cmap="YlGnBu",table=False):
    ## Create a correlation table
    corr = df.corr()
    ## Plot a correlation heatmap
    sns.heatmap(corr, xticklabels = corr.columns,yticklabels = corr.columns, cmap = cmap)
    ## Unstack the correlation matrix
    corr_df = corr.unstack().reset_index()
    ## Rename the columns
    corr_df.columns = ["feature_1","feature_2", "correlation"]
    ## Sort the correlation in descending order
    corr_df.sort_values(by="correlation",ascending=False, inplace=True)
    ## Remove self correlations
    corr_df = corr_df[corr_df["feature_1"] != corr_df["feature_2"]]
    ## Display the correlation table
    if table:
        return corr_df

def feature_perm(X,y,model):

    ## Fit the model
    model().fit(X, y)
    # Perform permutation
    permutation_score = permutation_importance(model, X, y, n_repeats=10)
    ## Unstack the results and store them into a DataFrame
    importance_df = pd.DataFrame(np.vstack((X.columns,permutation_score.importances_mean)).T)
    ## Rename the columns
    importance_df.columns=["feature","score decrease"]
    ## Order by importance
    importance_df.sort_values(by="score decrease", ascending = False)
    ## Display the importance table
    return importance_df
