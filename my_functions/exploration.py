import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def numerical(df):
    '''
    Explore the distribution of the numerical features of a given dataframe
    and also gives an idea of the outliers with boxplots.
    '''

    ## Select columns, excluding objects
    input_num = df.select_dtypes(exclude=["object"])
    ## Store the column names in a variable
    num_features = input_num.columns
    ## Loop over all the selected columns to generate the plots
    for num_feature in num_features:
        ## Generate the subplots
        fig, ax = plt.subplots(1, 2, figsize=(15,5))
        ## Generate the distribution plot
        ax[0].set_title(f"Distribution of {num_feature}")
        sns.histplot(data=df, x=num_feature, kde=True, ax = ax[0])
        ## Generate the boxplot
        ax[1].set_title(f"Boxplot of {num_feature}")
        sns.boxplot(data=df, x=num_feature, ax = ax[1])

def categorical(df):
    '''
    Explore the different categorical features of a given dataframe
    '''
    ## Select columns, excluding objects
    input_cat = df.select_dtypes(exclude=["float64","int64"])
    ## Store the column names in a variable
    features = input_cat.columns
    ## Loop over all the selected columns to display value_counts
    for feature in features:
        print(df[feature].value_counts())


def null(df):
    return df.isnull().sum().sort_values(ascending=False) / len(df)
